"""US-007 edge-count acceptance harness: an agent config must earn >= half the reference's edges.

Drives the REAL ``build_pipeline`` against a tiny REAL ``rs.build_fullmap_db`` redb (the
offline recipe from ``tests/test_e2e_smoke.py``) over the committed ``tests/fixtures/edgecount``
payload and asserts the agent-config fixture clears ``REFERENCE_EDGE_FRACTION`` of the richer
reference-config fixture's KGX edge count. The fixtures encode the US-006 improved-agent shape
(multi-section, correct sheet + row_slice, ``explode_by``/``prioritize`` breadth, paired
``effect_size`` + ``effect_type`` annotations); the reference is deliberately richer (all three
payload sheets) so the fraction is meaningful, and an intentionally-impoverished single-section
no-``explode_by`` config MUST fail the gate (the regression this harness exists to catch).

An env-gated test runs the same gate against REAL PMC artifacts: set ``TABLASSERT_PMC_COMPARE``
to a JSON array of four paths —
``["<agent-config>", "<reference-config>", "<payload>", "<fullmap-redb>"]`` (runbook in
``docs/agent.md``); unset, it skips with a printed reason. Everything offline is hermetic and
fast; all artifacts land in ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.cli import build_pipeline
from tablassert.ingests import to_yaml
from tablassert.progress import PipelineProgress

FIXTURE_DIR: Path = Path(__file__).parent / "fixtures" / "edgecount"
PAYLOAD: Path = FIXTURE_DIR / "payload.xlsx"

REFERENCE_EDGE_FRACTION: float = 0.5
"""Minimum fraction of the reference edge count an agent config must emit to be accepted."""

ENV_PMC_COMPARE: str = "TABLASSERT_PMC_COMPARE"

AGENT_CONFIG: str = "agent_config.yaml"
REFERENCE_CONFIG: str = "reference_config.yaml"
POOR_CONFIG: str = "agent_config_poor.yaml"

# The tiny real redb's entity sets (name -> CURIE). Every payload entity resolves through
# these; anything else is dropped by strict resolution.
DISEASES: list[tuple[str, str]] = [
    ("type 2 diabetes", "MONDO:1"),
    ("Crohn's disease", "MONDO:2"),
    ("ulcerative colitis", "MONDO:3"),
    ("rheumatoid arthritis", "MONDO:4"),
    ("asthma", "MONDO:5"),
    ("coronary artery disease", "MONDO:6"),
    ("chronic kidney disease", "MONDO:7"),
    ("Alzheimer disease", "MONDO:8"),
    ("Parkinson disease", "MONDO:9"),
    ("systemic lupus erythematosus", "MONDO:10"),
    ("psoriasis", "MONDO:11"),
    ("multiple sclerosis", "MONDO:12"),
]
SYSTEMS: list[tuple[str, str]] = [
    ("cardiovascular system", "UBERON:1"),
    ("nervous system", "UBERON:2"),
    ("digestive system", "UBERON:3"),
    ("respiratory system", "UBERON:4"),
    ("immune system", "UBERON:5"),
    ("musculoskeletal system", "UBERON:6"),
    ("endocrine system", "UBERON:7"),
    ("urinary system", "UBERON:8"),
]
LOCI: list[tuple[str, str]] = [
    ("FTO", "HGNC:101"),
    ("IL23R", "HGNC:102"),
    ("NOD2", "HGNC:103"),
    ("TCF7L2", "HGNC:104"),
    ("HLA-DRB1", "HGNC:105"),
    ("APOE", "HGNC:106"),
    ("PTPN22", "HGNC:107"),
    ("TNF", "HGNC:108"),
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _synonym_row(curie: str, preferred_name: str, category: str) -> dict[str, Any]:
    return {
        "curie": curie,
        "preferred_name": preferred_name,
        "names": [preferred_name.lower(), preferred_name],
        "types": [category],
        "taxa": ["NCBITaxon:9606"],
    }


def _class_row(curie: str) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": curie}]}


def _build_real_redb(root: Path) -> Path:
    """Tiny REAL fullmap redb registering every disease, organ system, and locus in the payload."""
    root.mkdir(parents=True, exist_ok=True)
    synonyms: list[dict[str, Any]] = [
        *[_synonym_row(curie, name, "Disease") for name, curie in DISEASES],
        *[_synonym_row(curie, name, "AnatomicalEntity") for name, curie in SYSTEMS],
        *[_synonym_row(curie, name, "Gene") for name, curie in LOCI],
    ]
    classes: list[dict[str, Any]] = [_class_row(curie) for _, curie in [*DISEASES, *SYSTEMS, *LOCI]]
    classes_path: Path = _write_jsonl(root / "classes.ndjson", classes)
    synonyms_path: Path = _write_jsonl(root / "synonyms.ndjson", synonyms)
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes_path], [synonyms_path])
    return output


def _rig(artifact_dir: Path, infores_id: str) -> dict[str, Any]:
    """Minimal VALID ``rig:`` dict (same contract as the conftest ``rig_factory`` fixture)."""
    return {
        "source_info": {
            "infores_id": infores_id,
            "terms_of_use_info": {"license_name": "CC0 1.0 Universal"},
            "data_access_locations": ["Edge-count fixture - https://example.org/data"],
            "source_status": "unknown",
        },
        "ingest_info": {"utility": "US-007 edge-count acceptance.", "scope": "Offline fixture comparison."},
        "provenance_info": {"contributions": ["Test author - code author"]},
        "artifact_base_url": f"https://example.org/{infores_id.removeprefix('infores:')}",
        "artifact_base_path": str(artifact_dir),
    }


def _load_config_with_absolute_local(config_file: Path, payload: Path) -> dict[str, Any]:
    """Load a committed {template, sections} config, pointing every section at ``payload``."""
    cfg: dict[str, Any] = yaml.safe_load(config_file.read_text())
    for section in cfg["sections"]:
        section["source"]["local"] = str(payload)
    return cfg


def _build_and_count_edges(root: Path, fullmap: Path, config: dict[str, Any], name: str) -> int:
    """Run the six-stage ``build_pipeline`` for one table config and count emitted KGX edges."""
    root.mkdir(parents=True, exist_ok=True)
    table: Path = root / f"{name.lower()}_table.yaml"
    to_yaml(table, config)
    artifact_dir: Path = root / name.lower()
    graph: Path = root / f"{name.lower()}_graph.yaml"
    graph_config: dict[str, Any] = {
        "name": name,
        "version": "0.0.1",
        "tables": [str(table)],
        "fullmap": str(fullmap),
        "rig": _rig(artifact_dir, f"infores:{name.lower()}"),
    }
    to_yaml(graph, graph_config)
    build_pipeline(graph, PipelineProgress(total_stages=6))
    edges_path: Path = artifact_dir / f"{name}_0.0.1.edges.ndjson"
    assert edges_path.is_file(), f"build emitted no edges file: {edges_path}"
    return sum(1 for line in edges_path.read_text().splitlines() if line.strip())


def _assert_fraction(agent_edges: int, reference_edges: int) -> None:
    assert agent_edges >= REFERENCE_EDGE_FRACTION * reference_edges, (
        f"agent edges {agent_edges} below {REFERENCE_EDGE_FRACTION} x reference edges {reference_edges}"
    )


def _pmc_compare_paths(spec: str) -> tuple[Path, Path, Path, Path]:
    """Parse ``TABLASSERT_PMC_COMPARE`` as a JSON array of exactly four non-empty path strings.

    A JSON array — not the legacy colon-separated form — so paths may themselves contain ``:``
    (POSIX) or carry Windows drive-letter prefixes. Every entry is ``expanduser().resolve()``d.
    Fails loudly via ``pytest.fail`` on any malformed value: a bad gate input must surface as an
    actionable error, never as a silent skip or fallback.
    """
    try:
        parsed: Any = json.loads(spec)
    except ValueError as exc:
        pytest.fail(f"{ENV_PMC_COMPARE} must be a JSON array of four path strings; got invalid JSON {spec!r}: {exc}")
    if not isinstance(parsed, list):
        pytest.fail(f"{ENV_PMC_COMPARE} must be a JSON array of four path strings; got non-array JSON {spec!r}")
    if len(parsed) != 4:
        pytest.fail(
            f"{ENV_PMC_COMPARE} must contain exactly four paths "
            f"(agent config, reference config, payload, fullmap redb); got {len(parsed)} in {spec!r}"
        )
    paths: list[Path] = []
    for entry in parsed:
        if not isinstance(entry, str) or not entry.strip():
            pytest.fail(f"{ENV_PMC_COMPARE} entries must be non-empty path strings; got {entry!r} in {spec!r}")
        paths.append(Path(entry).expanduser().resolve())
    return (paths[0], paths[1], paths[2], paths[3])


@pytest.fixture(scope="module")
def edge_counts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, int]:
    """Build all three fixture configs once against one tiny real redb; return their edge counts."""
    root: Path = tmp_path_factory.mktemp("edgecount")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)  # compile_graph/STORE are cwd-relative; keep the repo tree clean
        (root / ".tablassert" / "store").mkdir(parents=True)
        fullmap: Path = _build_real_redb(root / "fullmap")
        counts: dict[str, int] = {}
        for key, filename in (("agent", AGENT_CONFIG), ("reference", REFERENCE_CONFIG), ("poor", POOR_CONFIG)):
            config: dict[str, Any] = _load_config_with_absolute_local(FIXTURE_DIR / filename, PAYLOAD)
            counts[key] = _build_and_count_edges(root, fullmap, config, f"EDGECOUNT_{key.upper()}_KG")
    return counts


def test_edgecount_fixtures_encode_the_expected_shapes() -> None:
    """Committed fixtures exist and carry the harness's required shapes (no build needed)."""
    assert PAYLOAD.is_file()
    agent: dict[str, Any] = yaml.safe_load((FIXTURE_DIR / AGENT_CONFIG).read_text())
    reference: dict[str, Any] = yaml.safe_load((FIXTURE_DIR / REFERENCE_CONFIG).read_text())
    poor: dict[str, Any] = yaml.safe_load((FIXTURE_DIR / POOR_CONFIG).read_text())
    # Improved-agent shape: multi-section, correct sheet + row_slice, paired effect annotations.
    assert len(agent["sections"]) >= 2
    for section in agent["sections"]:
        assert section["source"]["sheet"]
        assert section["source"]["row_slice"] == [2, "auto"]
        targets: set[str] = {str(annotation["annotation"]) for annotation in section["annotations"]}
        assert {"effect_size", "effect_type"} <= targets
    # The reference is richer (more sections over more sheets) so the fraction is meaningful.
    assert len(reference["sections"]) > len(agent["sections"])
    # The negative control is single-section with no explode_by.
    assert len(poor["sections"]) == 1
    assert "explode_by" not in poor["sections"][0]["statement"]["object"]


def test_reference_is_strictly_richer_than_agent(edge_counts: dict[str, int]) -> None:
    """Both configs emit edges and the reference emits strictly more (meaningful fraction)."""
    assert edge_counts["agent"] > 0
    assert edge_counts["reference"] > edge_counts["agent"]


def test_agent_config_clears_reference_fraction(edge_counts: dict[str, int]) -> None:
    """The improved-agent fixture earns >= REFERENCE_EDGE_FRACTION of the reference edge count."""
    _assert_fraction(edge_counts["agent"], edge_counts["reference"])


def test_poor_agent_config_fails_reference_fraction(edge_counts: dict[str, int]) -> None:
    """The impoverished fixture MUST fall below the gate: the regression this harness catches."""
    with pytest.raises(AssertionError, match="below"):
        _assert_fraction(edge_counts["poor"], edge_counts["reference"])


def test_pmc_compare_spec_parses_four_paths() -> None:
    """A well-formed JSON array of four paths parses into expanded, resolved Paths.

    US-010: pins the success contract of the gate input so a future format tweak cannot silently
    redefine what the four entries (agent config, reference config, payload, fullmap redb) mean;
    also proves ``~`` expansion and ``resolve()`` are applied to every entry.
    """
    entries: list[str] = [".tablassert/a.yaml", "./legacy/b.yaml", "./downloads/table.xlsx", "data/fullmap.redb"]
    assert _pmc_compare_paths(json.dumps(entries)) == tuple(Path(entry).expanduser().resolve() for entry in entries)
    tilde: tuple[Path, Path, Path, Path] = _pmc_compare_paths(json.dumps(["~/agent.yaml", "b.yaml", "c.xlsx", "d.redb"]))
    assert tilde[0] == Path("~/agent.yaml").expanduser().resolve()


def test_pmc_compare_spec_accepts_colons_and_drive_letters() -> None:
    """Paths containing ``:`` (POSIX) or Windows drive letters parse cleanly.

    The whole reason US-010 exists: ``spec.split(":")`` rejected both classes of valid path
    (CodeRabbit finding on PR #105). This locks in the fix so a future "simplify back to
    ``split(':')``" cannot regress it.
    """
    entries: list[str] = ["/home/ci/odd:dir/agent.yaml", "C:/Users/ci/reference.yaml", "relative:with:colons/table.xlsx", "D:/data/fullmap.redb"]
    parsed: tuple[Path, Path, Path, Path] = _pmc_compare_paths(json.dumps(entries))
    assert parsed == tuple(Path(entry).expanduser().resolve() for entry in entries)
    assert [path.name for path in parsed] == ["agent.yaml", "reference.yaml", "table.xlsx", "fullmap.redb"]


def test_pmc_compare_spec_rejects_legacy_colon_form() -> None:
    """The pre-US-010 colon-separated form is rejected loudly, never silently split.

    Operators upgrading from the old format must get an actionable ``pytest.fail`` naming the env
    var and echoing their value, so they fix the export instead of wondering why the real PMC
    comparison silently skips.
    """
    legacy: str = "a.yaml:b.yaml:c.xlsx:d.redb"
    with pytest.raises(pytest.fail.Exception, match=ENV_PMC_COMPARE):
        _pmc_compare_paths(legacy)


@pytest.mark.parametrize(
    "spec",
    [
        "{}",
        '"a.yaml:b.yaml:c.xlsx:d.redb"',
        '["a.yaml", "b.yaml", "c.xlsx"]',
        '["a.yaml", "b.yaml", "c.xlsx", "d.redb", "e.yaml"]',
        '["a.yaml", "b.yaml", "c.xlsx", 4]',
        '["a.yaml", "", "c.xlsx", "d.redb"]',
        '["a.yaml", "   ", "c.xlsx", "d.redb"]',
        '["a.yaml", null, "c.xlsx", "d.redb"]',
    ],
)
def test_pmc_compare_spec_rejects_wrong_shapes(spec: str) -> None:
    """Every malformed variant fails loudly, naming the env var and echoing the offending value.

    US-010 acceptance criterion: a silent default would let a mistyped gate quietly skip the real
    PMC comparison, defeating the whole acceptance harness — so each wrong shape gets an explicit
    ``pytest.fail`` instead of a fallback.
    """
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _pmc_compare_paths(spec)
    message: str = str(excinfo.value)
    assert ENV_PMC_COMPARE in message
    assert spec in message


def test_docs_pmc_compare_example_is_valid_json() -> None:
    """Drift guard: the docs/agent.md runbook example is valid JSON the parser accepts.

    Operators copy-paste the documented command verbatim; if the docs example drifts from the
    parser's JSON-array contract they only discover it as a ``pytest.fail`` at run time. Parsing
    the documented value here keeps docs/agent.md and ``_pmc_compare_paths`` in lockstep.
    """
    doc: str = (Path(__file__).parent.parent / "docs" / "agent.md").read_text()
    match: re.Match[str] | None = re.search(r"TABLASSERT_PMC_COMPARE='(\[[^]]*\])'", doc)
    assert match is not None, "docs/agent.md lost the TABLASSERT_PMC_COMPARE JSON-array example"
    example: str = match.group(1)
    entries: Any = json.loads(example)
    assert isinstance(entries, list)
    assert len(entries) == 4
    assert all(isinstance(entry, str) and entry.strip() for entry in entries)
    _pmc_compare_paths(example)


def test_real_pmc_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REAL PMC comparison, gated on ``TABLASSERT_PMC_COMPARE``; skips with a reason when unset.

    ``TABLASSERT_PMC_COMPARE`` is a JSON array of four paths —
    ``["<agent-config>", "<reference-config>", "<payload>", "<fullmap-redb>"]`` — e.g. the
    agent's accepted config vs the converted legacy reference config over the downloaded payload,
    both built against the project fullmap (runbook: ``docs/agent.md``).
    """
    spec: str | None = os.environ.get(ENV_PMC_COMPARE)
    if not spec:
        reason: str = (
            f"set {ENV_PMC_COMPARE} to a JSON array of four paths — "
            f'["<agent-config>", "<reference-config>", "<payload>", "<fullmap-redb>"] — '
            "to run the real PMC comparison (runbook: docs/agent.md)"
        )
        print(reason)
        pytest.skip(reason)

    agent_config_path, reference_config_path, payload_path, fullmap_path = _pmc_compare_paths(spec)
    for candidate in (agent_config_path, reference_config_path, payload_path, fullmap_path):
        assert candidate.is_file(), f"{ENV_PMC_COMPARE} path does not exist: {candidate}"

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    agent_edges: int = _build_and_count_edges(
        tmp_path / "agent", fullmap_path, _load_config_with_absolute_local(agent_config_path, payload_path), "PMC_AGENT_KG"
    )
    reference_edges: int = _build_and_count_edges(
        tmp_path / "reference", fullmap_path, _load_config_with_absolute_local(reference_config_path, payload_path), "PMC_REFERENCE_KG"
    )
    print(f"real PMC comparison: agent edges={agent_edges} reference edges={reference_edges}")
    _assert_fraction(agent_edges, reference_edges)
