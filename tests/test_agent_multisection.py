"""Tests for W3 multi-section: one config per paper, multiple sections, per-section files/URLs.

The pure tests (``validate_table_config``, ``map_coverage`` aggregation, ``propose_config_edit``
per-section) run in the BASE environment (no ``[agent]`` extra); the coverage tests reuse the tiny REAL
redb recipe (``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871). The supervisor test drives a real
``CodeAgent`` OFFLINE via ``FakeModel`` and calls ``pytest.importorskip("smolagents")`` so it skips
cleanly without the extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import ConfigRecord, make_fake_model, map_coverage, propose_config_edit, run_supervisor, validate_table_config
from tablassert.biolink import Categories

ORGANISM_TAXON: str = Categories.ORGANISM_TAXON.value  # pyright: ignore[reportAttributeAccessIssue]
GENE: str = Categories.GENE.value


# --------------------------------------------------------------------------- #
# Offline fixtures: tiny REAL redb + multi-section config builders
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _synonym_row(curie: str, preferred_name: str, names: list[str], category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


@pytest.fixture
def redb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Offline real redb + an isolated cwd (``.tablassert/store`` mirrors the e2e recipe)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    root: Path = tmp_path / "fullmap"
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [_class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [_synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "brca1"], "Gene"), _synonym_row("HGNC:6871", "MAPK1", ["MAPK1", "mapk1"], "Gene")],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


def _write_table(tmp_path: Path, name: str, text: str) -> Path:
    table: Path = tmp_path / name
    table.write_text(text)
    return table


def _section(local: Path, *, subject: str = "A", obj: str = "B") -> dict[str, Any]:
    """One section: column subject/object, its OWN source (local + url), associated_with."""
    return {
        "source": {"kind": "text", "local": str(local), "url": [f"https://example.com/{local.name}"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": subject},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": obj},
        },
    }


def _multi_cfg(*sections: dict[str, Any]) -> dict[str, Any]:
    """A multi-section table config: shared template provenance, NO source in the template."""
    return {"template": {"provenance": {"repo": "PMC", "publication": "PMC1"}}, "sections": list(sections)}


# --------------------------------------------------------------------------- #
# validate_table_config — the multi-section final-answer gate (PURE; base env)
# --------------------------------------------------------------------------- #


def test_validate_table_config_accepts_multi_section(tmp_path: Path) -> None:
    """A {template, sections} config with ALL sections valid passes the gate."""
    t1: Path = _write_table(tmp_path, "s1.tsv", "brca1\tmapk1\n")
    t2: Path = _write_table(tmp_path, "s2.tsv", "mapk1\tbrca1\n")
    assert validate_table_config(yaml.safe_dump(_multi_cfg(_section(t1), _section(t2)), sort_keys=False)) is True


def test_validate_table_config_rejects_one_bad_section(tmp_path: Path) -> None:
    """One INVALID section (missing source) among valid ones => the WHOLE config is rejected (validate ALL)."""
    t1: Path = _write_table(tmp_path, "s1.tsv", "brca1\tmapk1\n")
    bad: dict[str, Any] = {
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "value", "encoding": "X"},
        }
    }
    cfg: dict[str, Any] = _multi_cfg(_section(t1), bad)  # second section has no source
    assert validate_table_config(yaml.safe_dump(cfg, sort_keys=False)) is False


def test_validate_table_config_single_section_backcompat(tmp_path: Path) -> None:
    """A bare single section and a {template: {...}} config remain valid (one-section cases)."""
    t1: Path = _write_table(tmp_path, "s1.tsv", "brca1\tmapk1\n")
    bare: dict[str, Any] = {**_section(t1), "provenance": {"repo": "PMC", "publication": "PMC1"}}
    assert validate_table_config(yaml.safe_dump(bare, sort_keys=False)) is True
    assert validate_table_config(yaml.safe_dump({"template": bare}, sort_keys=False)) is True


@pytest.mark.parametrize("cfg", ["just a string", "[1, 2]", "template: {}\nsections: []\n", "::: not yaml"])
def test_validate_table_config_never_raises(cfg: str) -> None:
    """Non-mapping / empty-sections / invalid YAML all return False, never raise."""
    assert validate_table_config(cfg) is False


# --------------------------------------------------------------------------- #
# map_coverage — multi-section aggregation (PURE; needs the real redb)
# --------------------------------------------------------------------------- #


def test_map_coverage_multi_section_aggregates(tmp_path: Path, redb: Path) -> None:
    """Two measurable sections aggregate: overall = MEAN, min = weakest, measured True, union unresolved."""
    good: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\n")  # both resolve -> 1.0
    half: Path = _write_table(tmp_path, "half.tsv", "brca1\tzzznotreal\n")  # object unresolved -> 0.5
    result: dict[str, Any] = map_coverage(_multi_cfg(_section(good), _section(half)), fullmap=redb, workdir=tmp_path)

    assert result["measured"] is True
    assert result["overall"] == pytest.approx(0.75)  # mean(1.0, 0.5)
    assert result["min"] == pytest.approx(0.5)
    sections = result["sections"]
    assert isinstance(sections, list)
    assert len(sections) == 2
    unresolved = result["unresolved"]
    assert isinstance(unresolved, list)
    assert "zzznotreal" in unresolved  # union across sections
    # multi-section: top-level per_column is empty (per-column lives under each section)
    assert result["per_column"] == {}


def test_map_coverage_unmeasurable_section_counts_zero(tmp_path: Path, redb: Path) -> None:
    """One measurable (1.0) + one UNMEASURABLE section => overall 0.5 (mean), measured False, NOT empty.

    An unmeasurable section contributes 0.0 (never a false perfect); ``measured`` is True iff EVERY section
    measured, so a partially-measurable config reports measured=False while still surfacing its aggregate.
    """
    good: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\n")
    missing: Path = tmp_path / "definitely_missing.tsv"  # never written -> frame unreproducible
    result: dict[str, Any] = map_coverage(_multi_cfg(_section(good), _section(missing)), fullmap=redb, workdir=tmp_path)

    assert result["measured"] is False  # not EVERY section measured
    assert result["overall"] == pytest.approx(0.5)  # mean(1.0, 0.0) — unmeasurable counts as 0.0
    assert result["min"] == pytest.approx(0.0)
    sections = result["sections"]
    assert isinstance(sections, list)
    assert len(sections) == 2


def test_map_coverage_fully_unmeasurable_returns_empty(tmp_path: Path, redb: Path) -> None:
    """ALL sections unmeasurable => the 4-key empty result (back-compat exact shape, never a false score)."""
    m1: Path = tmp_path / "missing1.tsv"
    m2: Path = tmp_path / "missing2.tsv"
    result: dict[str, Any] = map_coverage(_multi_cfg(_section(m1), _section(m2)), fullmap=redb, workdir=tmp_path)
    assert result == {"overall": 0.0, "measured": False, "per_column": {}, "unresolved": []}


# --------------------------------------------------------------------------- #
# propose_config_edit — per-section editing (PURE; base env)
# --------------------------------------------------------------------------- #


def _multi_report(*per_section: dict[str, Any]) -> dict[str, Any]:
    """A multi-section coverage report: ``sections`` aligned positionally with the config's sections."""
    return {"overall": 0.5, "min": 0.0, "measured": True, "sections": list(per_section), "unresolved": ["g__Bacteroides"]}


def _taxonomic_per_column() -> dict[str, Any]:
    return {"subject": {"coverage": 0.0, "total": 1, "resolved": 0, "unresolved": ["g__Bacteroides"], "method": "column"}}


def _clean_per_column() -> dict[str, Any]:
    return {"subject": {"coverage": 1.0, "total": 1, "resolved": 1, "unresolved": [], "method": "column"}}


def test_propose_multi_section_edits_only_unresolved_section(tmp_path: Path) -> None:
    """Only the section with unresolved terms is edited; the clean section + template are untouched."""
    t1: Path = _write_table(tmp_path, "s1.tsv", "g__Bacteroides\tmapk1\n")
    t2: Path = _write_table(tmp_path, "s2.tsv", "brca1\tmapk1\n")
    cfg: dict[str, Any] = _multi_cfg(_section(t1), _section(t2))
    original: str = yaml.safe_dump(cfg, sort_keys=False)
    report: dict[str, Any] = _multi_report(
        {"overall": 0.0, "measured": True, "per_column": _taxonomic_per_column(), "unresolved": ["g__Bacteroides"]},
        {"overall": 1.0, "measured": True, "per_column": _clean_per_column(), "unresolved": []},
    )

    edited, rationale = propose_config_edit(cfg, report)

    assert validate_table_config(edited) is True
    assert edited != original
    parsed: dict[str, Any] = yaml.safe_load(edited)
    # Section 0 (unresolved taxonomic) gained the organism prioritization...
    assert ORGANISM_TAXON in parsed["sections"][0]["statement"]["subject"]["prioritize"]
    assert GENE in parsed["sections"][0]["statement"]["subject"]["avoid"]
    # ...section 1 (clean) is untouched (no prioritize added)...
    assert "prioritize" not in parsed["sections"][1]["statement"]["subject"]
    # ...and the shared template provenance is never edited.
    assert parsed["template"] == {"provenance": {"repo": "PMC", "publication": "PMC1"}}
    assert "g__Bacteroides" in rationale


def test_propose_multi_section_idempotent(tmp_path: Path) -> None:
    """Re-proposing on the edited multi-section config does not grow the per-section knob lists."""
    t1: Path = _write_table(tmp_path, "s1.tsv", "g__Bacteroides\tmapk1\n")
    t2: Path = _write_table(tmp_path, "s2.tsv", "brca1\tmapk1\n")
    report: dict[str, Any] = _multi_report(
        {"overall": 0.0, "measured": True, "per_column": _taxonomic_per_column(), "unresolved": ["g__Bacteroides"]},
        {"overall": 1.0, "measured": True, "per_column": _clean_per_column(), "unresolved": []},
    )
    edited, _ = propose_config_edit(_multi_cfg(_section(t1), _section(t2)), report)
    edited2, _ = propose_config_edit(edited, report)

    first: dict[str, Any] = yaml.safe_load(edited)["sections"][0]["statement"]["subject"]
    second: dict[str, Any] = yaml.safe_load(edited2)["sections"][0]["statement"]["subject"]
    assert second["prioritize"] == first["prioritize"]
    assert second["prioritize"].count(ORGANISM_TAXON) == 1


def test_propose_multi_section_no_safe_edit(tmp_path: Path) -> None:
    """When no section has unresolved terms, the original config is returned with a 'no safe edit' note."""
    t1: Path = _write_table(tmp_path, "s1.tsv", "brca1\tmapk1\n")
    cfg: dict[str, Any] = _multi_cfg(_section(t1))
    original: str = yaml.safe_dump(cfg, sort_keys=False)
    report: dict[str, Any] = _multi_report({"overall": 1.0, "measured": True, "per_column": _clean_per_column(), "unresolved": []})
    edited, rationale = propose_config_edit(cfg, report)
    assert edited == original
    assert "no safe edit" in rationale


def test_propose_multi_section_validation_failure_returns_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed multi-section edit that FAILS validate_table_config returns the ORIGINAL config."""
    monkeypatch.setattr("tablassert.agent.validate_table_config", lambda *args, **kwargs: False)
    t1: Path = _write_table(tmp_path, "s1.tsv", "g__Bacteroides\tmapk1\n")
    cfg: dict[str, Any] = _multi_cfg(_section(t1))
    original: str = yaml.safe_dump(cfg, sort_keys=False)
    report: dict[str, Any] = _multi_report(
        {"overall": 0.0, "measured": True, "per_column": _taxonomic_per_column(), "unresolved": ["g__Bacteroides"]}
    )
    edited, rationale = propose_config_edit(cfg, report)
    assert edited == original
    assert "failed schema validation" in rationale


# --------------------------------------------------------------------------- #
# Supervisor — ONE multi-section config per paper (needs the [agent] extra)
# --------------------------------------------------------------------------- #


def test_supervisor_one_multisection_config_per_paper(tmp_path: Path, redb: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The supervisor yields ONE multi-section config per paper, with distinct per-section sources + URLs.

    The inner FakeModel returns a 2-section config (each section its own table); the supervisor builds it
    (both sections resolve -> MAPPED), writes ONE best config retaining both sections, records per-section
    coverages, and presents candidate tables to the agent as ``local -> url``.
    """
    pytest.importorskip("smolagents")
    import tablassert.agent as agent_mod

    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")

    # Two tables under a prefix-shaped dir so public_url(prefix, name) yields a sensible link.
    prefix_dir: Path = tmp_path / "downloads" / "PMC1" / "PMC1.1"
    prefix_dir.mkdir(parents=True)
    t1: Path = prefix_dir / "s1.tsv"
    t1.write_text("brca1\tmapk1\nbrca1\tmapk1\n")
    t2: Path = prefix_dir / "s2.tsv"
    t2.write_text("mapk1\tbrca1\nmapk1\tbrca1\n")

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        return [t1, t2]

    monkeypatch.setattr(agent_mod, "fetch_pmc_article", fake_fetch)

    multi_cfg: dict[str, Any] = _multi_cfg(_section(t1), _section(t2))
    multi_yaml: str = yaml.safe_dump(multi_cfg, sort_keys=False)

    # Spy on the task to confirm tables are presented as local -> url.
    captured: dict[str, str] = {}
    real_build_agent = agent_mod.build_agent

    def spy_build_agent(*args: object, **kwargs: object) -> object:
        agent = real_build_agent(*args, **kwargs)

        class _Spy:
            def run(self, task: str) -> object:
                captured["task"] = task
                return agent.run(task)  # pyright: ignore[reportAttributeAccessIssue]

        return _Spy()

    monkeypatch.setattr(agent_mod, "build_agent", spy_build_agent)

    result: dict[str, Any] = run_supervisor(
        ["PMC1"],
        fullmap=redb,
        build_model_factory=lambda: make_fake_model(final_yaml=multi_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
    )

    rec: ConfigRecord = result["records"]["PMC1"]  # pyright: ignore[reportIndexIssue]
    assert rec.status == "MAPPED"

    # ONE best config per paper, retaining BOTH sections with distinct per-section sources.
    assert rec.best_config_path is not None
    best: dict[str, Any] = yaml.safe_load(Path(rec.best_config_path).read_text())
    assert "sections" in best
    assert len(best["sections"]) == 2
    locals_: list[str] = [str(section["source"]["local"]) for section in best["sections"]]
    assert str(t1) in locals_
    assert str(t2) in locals_

    # Per-section coverages recorded (both sections fully resolve -> [1.0, 1.0]).
    assert rec.section_coverages == [1.0, 1.0]

    # The task presents each candidate table as local -> url (prefix-derived public URL).
    task: str = captured["task"]
    assert str(t1) in task
    assert str(t2) in task
    assert "source.url" in task
    assert "pmc-oa-opendata.s3.amazonaws.com/PMC1.1/s1.tsv" in task
