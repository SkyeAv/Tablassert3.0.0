"""Targeted branch-coverage tests for tablassert.agent (US-013).

These exercise reachable-but-otherwise-uncovered branches so the optional ``[agent]`` module stays
above the 95% coverage bar WITHOUT weakening any assertion: the ``_require`` missing-package raise,
the step-callback context trim, the Reflexion-with-fullmap re-score path, the GEPA ``TypeError`` /
compile-failure fallbacks, the judge parse/prompt helpers, the propose noise/chemical heuristics, and
the fetch error branches. Genuinely live-only seams (real ``urlopen``) carry a precise ``# pragma: no
cover`` in agent.py instead. dspy-dependent tests use ``importorskip``; an autouse fixture keeps any
agent/dspy run hermetically offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tablassert.agent import (
    _default_gepa_program,
    _parse_judge_scores,
    _require,
    make_step_callback,
    propose_config_edit,
    reflexion_improve,
    run_gepa,
    validate_section,
)


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep any agent/dspy run hermetically offline (HF telemetry blocks on network)."""
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


def _section_cfg(subject: dict[str, Any], object_: dict[str, Any], predicate: str = "associated_with") -> str:
    """A schema-valid Section config YAML with the given subject/object encodings."""
    return yaml.safe_dump(
        {
            "source": {"kind": "text", "local": "./t.tsv", "url": ["https://e.com/t.tsv"], "delimiter": "\t"},
            "statement": {"subject": subject, "predicate": predicate, "object": object_},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        }
    )


# --------------------------------------------------------------------------- #
# _require missing-package raise (otherwise only hit when the extra is absent)
# --------------------------------------------------------------------------- #


def test_require_raises_for_missing_package() -> None:
    """_require raises an actionable ImportError naming the extra for a package that is not installed."""
    with pytest.raises(ImportError, match=r"tablassert\[agent\]"):
        _require("definitely_not_a_real_package_xyz_123")


# --------------------------------------------------------------------------- #
# step_callback context trimming
# --------------------------------------------------------------------------- #


def test_step_callback_trims_old_observations() -> None:
    """Observations older than the last two steps are trimmed when > 4000 chars (token saving)."""
    metrics: dict[str, object] = {}
    cb = make_step_callback(metrics)
    old = SimpleNamespace(observations="x" * 5000)
    mid = SimpleNamespace(observations="y" * 10)
    recent = SimpleNamespace(observations="z" * 10)
    agent = SimpleNamespace(memory=SimpleNamespace(steps=[old, mid, recent]))
    step = SimpleNamespace(error=None, tool_calls=[], observations="ok", token_usage=None, is_final_answer=False)
    cb(step, agent)
    assert str(old.observations).startswith("[trimmed observation:")  # only steps[:-2] is trimmed
    assert mid.observations == "y" * 10  # the last two are preserved
    assert recent.observations == "z" * 10


def test_step_callback_trim_never_raises_on_odd_memory() -> None:
    """A missing/odd memory shape makes trimming a safe no-op (the callback never raises)."""
    metrics: dict[str, object] = {}
    cb = make_step_callback(metrics)
    step = SimpleNamespace(error=None, tool_calls=[], observations="ok", token_usage=None, is_final_answer=False)
    cb(step, SimpleNamespace(memory=None))  # no memory at all
    cb(step, SimpleNamespace(memory=SimpleNamespace(steps="not-a-list")))  # wrong shape
    assert metrics["steps"] == 2


# --------------------------------------------------------------------------- #
# Reflexion with a fullmap (re-score + promote path)
# --------------------------------------------------------------------------- #


def _tiny_redb(root: Path) -> Path:
    """Real redb: brca1 -> HGNC:1100, mapk1 -> HGNC:6871 (e2e recipe)."""
    from tablassert import rs

    root.mkdir(parents=True, exist_ok=True)

    def jl(p: Path, rows: list[dict[str, Any]]) -> Path:
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    classes = jl(root / "classes.ndjson", [{"id": "HGNC:1100", "equivalent_identifiers": [{"identifier": "NCBIGene:672"}]}])
    synonyms = jl(
        root / "synonyms.ndjson",
        [
            {"curie": "HGNC:1100", "preferred_name": "BRCA1", "names": ["BRCA1", "brca1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]},
            {"curie": "HGNC:6871", "preferred_name": "MAPK1", "names": ["MAPK1", "mapk1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]},
        ],
    )
    output = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


def test_reflexion_with_fullmap_rescores_and_promotes(tmp_path: Path) -> None:
    """With a fullmap, reflexion_improve rebuilds the edited config and promotes a strict improvement.

    The subject column 'g__brca1' is unresolved (coverage 0.5); propose_config_edit strips the 'g__'
    lineage glue so 'brca1' resolves (coverage 1.0). The fullmap re-score path accepts the improvement.
    """
    db = _tiny_redb(tmp_path / "fullmap")
    table = tmp_path / "d.tsv"
    table.write_text("g__brca1\tmapk1\ng__brca1\tmapk1\n")
    cfg = yaml.safe_dump(
        {
            "source": {"kind": "text", "local": str(table), "url": ["https://e.com/d.tsv"], "delimiter": "\t"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "predicate": "associated_with",
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {"repo": "PMC", "publication": "PMC1"},
        }
    )
    coverage_report = {
        "overall": 0.5,
        "per_column": {"subject": {"coverage": 0.5, "total": 1, "resolved": 0, "unresolved": ["g__brca1"], "method": "column"}},
        "unresolved": ["g__brca1"],
    }
    best, reflections = reflexion_improve(cfg, {"coverage_pct": 0.5, "errors": [], "error_codes": []}, coverage_report, max_reflections=1, fullmap=db)
    assert validate_section(best) is True
    assert reflections
    # The promoted config strips the lineage glue (the edit that raises coverage).
    assert "g__" in yaml.safe_dump(yaml.safe_load(best))  # the regex pattern referencing g__ is present


# --------------------------------------------------------------------------- #
# GEPA fallbacks (importorskip): TypeError constructor fallback + compile-failure branch
# --------------------------------------------------------------------------- #


def test_run_gepa_constructor_typeerror_fallback() -> None:
    """When the optimizer rejects the full kwargs (TypeError), run_gepa falls back to metric-only."""
    pytest.importorskip("dspy")
    seen: dict[str, Any] = {}

    class NarrowGEPA:
        def __init__(self, metric: Any = None, **kwargs: Any) -> None:
            if kwargs:  # simulate an optimizer that only accepts `metric`
                raise TypeError("unexpected kwargs")
            seen["metric"] = metric
            self.gepa_stats: dict[str, Any] = {}

        def compile(self, program: Any, *, trainset: Any = None, **kwargs: Any) -> Any:
            predictor = SimpleNamespace(signature=SimpleNamespace(instructions="OPT"))
            return SimpleNamespace(named_predictors=lambda: [("propose", predictor)])

    result = run_gepa(seed_instructions="SEED", gepa_cls=NarrowGEPA, reflection_lm=SimpleNamespace(), trainset=[])
    assert seen["metric"] is not None  # the metric-only fallback constructed the optimizer
    assert result["optimized_instructions"] == "OPT"


def test_run_gepa_compile_failure_is_captured() -> None:
    """A failing GEPA compile never raises; the error is recorded in stats and the seed is kept."""
    pytest.importorskip("dspy")

    class BrokenGEPA:
        def __init__(self, metric: Any = None, **kwargs: Any) -> None:
            self.gepa_stats: dict[str, Any] = {}

        def compile(self, program: Any, *, trainset: Any = None, **kwargs: Any) -> Any:
            raise RuntimeError("simulated GEPA failure")

    result = run_gepa(seed_instructions="SEED-KEPT", gepa_cls=BrokenGEPA, reflection_lm=SimpleNamespace(), trainset=[])
    assert result["optimized_instructions"] == "SEED-KEPT"  # fell back to the seed
    assert "error" in result["stats"]


def test_default_gepa_program_builds_offline() -> None:
    """The default dspy program (a single Predict over the config signature) builds without network."""
    pytest.importorskip("dspy")
    program = _default_gepa_program("SEED INSTRUCTIONS")
    names = [name for name, _ in program.named_predictors()]
    assert "propose" in names


# --------------------------------------------------------------------------- #
# Judge parse/prompt helpers
# --------------------------------------------------------------------------- #


def test_judge_parse_scores_clamps_and_defaults() -> None:
    """_parse_judge_scores parses 'dim: score' lines, clamps to [0,3], and defaults missing dims to 0."""
    scores = _parse_judge_scores("schema_validity: 3\ncoverage_appropriateness: 2\nqc_pass: 99\ngarbage line without colon")
    assert scores["schema_validity"] == 3.0
    assert scores["coverage_appropriateness"] == 2.0
    assert scores["qc_pass"] == 3.0  # 99 clamped down to 3
    assert scores["efficiency"] == 0.0  # absent -> 0


def test_judge_build_prompt_orders_dimensions() -> None:
    """_build_judge_prompt embeds the rubric + dimensions, reversing order when asked (position debias)."""
    from tablassert.agent import JUDGE_DIMENSIONS, _build_judge_prompt

    marker = "## Dimensions (in this order)"
    forward = _build_judge_prompt("cfg: {}", {"coverage_pct": 0.5}, {"steps": 1})
    reversed_ = _build_judge_prompt("cfg: {}", {"coverage_pct": 0.5}, {"steps": 1}, reverse=True)
    # The first dimension listed AFTER the marker reflects the (possibly reversed) order.
    fwd_first = forward.split(marker, 1)[1].lstrip().splitlines()[0]
    rev_first = reversed_.split(marker, 1)[1].lstrip().splitlines()[0]
    assert fwd_first == f"- {JUDGE_DIMENSIONS[0]}"
    assert rev_first == f"- {JUDGE_DIMENSIONS[-1]}"


# --------------------------------------------------------------------------- #
# Propose noise + chemical-fallback heuristics
# --------------------------------------------------------------------------- #


def test_propose_noise_adds_remove_patterns() -> None:
    """Noise terms ('NA ...', '[...]') add the corresponding remove regex patterns."""
    cfg = _section_cfg({"method": "column", "encoding": "A"}, {"method": "value", "encoding": "CHEBI:41774"})
    report: dict[str, object] = {
        "overall": 0.5,
        "per_column": {"subject": {"coverage": 0.5, "total": 2, "resolved": 1, "unresolved": ["NA value", "[bracketed]"], "method": "column"}},
        "unresolved": ["NA value", "[bracketed]"],
    }
    edited, rationale = propose_config_edit(cfg, report)
    subject = yaml.safe_load(edited)["statement"]["subject"]
    assert "^NA " in subject.get("remove", [])
    assert "\\[.*?\\]" in subject.get("remove", [])
    assert validate_section(edited) is True
    assert "remove" in rationale


def test_propose_chemical_fallback_prioritizes_chemical_entity() -> None:
    """An object column with chemical-looking unresolved terms (and no other signal) prioritizes ChemicalEntity."""
    cfg = _section_cfg({"method": "column", "encoding": "A"}, {"method": "column", "encoding": "B"})
    report: dict[str, object] = {
        "overall": 0.5,
        "per_column": {"object": {"coverage": 0.0, "total": 1, "resolved": 0, "unresolved": ["CHEBI:17196"], "method": "column"}},
        "unresolved": ["CHEBI:17196"],
    }
    edited, rationale = propose_config_edit(cfg, report)
    object_node = yaml.safe_load(edited)["statement"]["object"]
    assert "ChemicalEntity" in object_node.get("prioritize", [])
    assert validate_section(edited) is True
    assert "chemical fallback" in rationale


# --------------------------------------------------------------------------- #
# Fetch error branches (mocked HTTP)
# --------------------------------------------------------------------------- #


def test_fetch_metadata_unreadable_is_not_fatal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unreadable ``.json`` metadata is NON-fatal: license is treated as unknown and files still download."""
    import tablassert.agent as agent_mod

    version_listing = '<?xml version="1.0"?><ListBucketResult><CommonPrefixes><Prefix>PMC1.1/</Prefix></CommonPrefixes></ListBucketResult>'
    object_listing = '<?xml version="1.0"?><ListBucketResult><Contents><Key>PMC1.1/PMC1.1.xml</Key></Contents><Contents><Key>PMC1.1/t.xlsx</Key></Contents></ListBucketResult>'

    def get_text(url: str, *, timeout: int = 120) -> str:
        if "list-type=2" in url and "delimiter=" in url:
            return version_listing
        if "list-type=2" in url:
            return object_listing
        if url.endswith(".json"):
            raise OSError("simulated metadata read failure")  # license becomes unknown, not fatal
        raise AssertionError(f"unexpected text url: {url}")

    monkeypatch.setattr(agent_mod, "_http_get_text", get_text)
    monkeypatch.setattr(agent_mod, "_http_get_bytes", lambda url, *, timeout=120: b"X")
    paths = agent_mod.fetch_pmc_article("PMC1", tmp_path)
    assert sorted(p.name for p in paths) == ["PMC1.1.xml", "t.xlsx"]


def test_fetch_no_tables_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """OA article whose latest version has main text but NO table raises FileNotFoundError (extension gate)."""
    import tablassert.agent as agent_mod

    version_listing = '<?xml version="1.0"?><ListBucketResult><CommonPrefixes><Prefix>PMC2.1/</Prefix></CommonPrefixes></ListBucketResult>'
    object_listing = '<?xml version="1.0"?><ListBucketResult><Contents><Key>PMC2.1/PMC2.1.xml</Key></Contents><Contents><Key>PMC2.1/fig.jpg</Key></Contents></ListBucketResult>'

    def get_text(url: str, *, timeout: int = 120) -> str:
        if "list-type=2" in url and "delimiter=" in url:
            return version_listing
        if "list-type=2" in url:
            return object_listing
        if url.endswith(".json"):
            return '{"is_pmc_openaccess": true, "license_code": "CC-BY"}'
        raise AssertionError(f"unexpected text url: {url}")

    monkeypatch.setattr(agent_mod, "_http_get_text", get_text)
    monkeypatch.setattr(agent_mod, "_http_get_bytes", lambda url, *, timeout=120: b"X")
    with pytest.raises(FileNotFoundError, match="No supplementary tables"):
        agent_mod.fetch_pmc_article("PMC2", tmp_path)


def test_parse_judge_scores_empty_value_line_skipped() -> None:
    """Regression (review fix 5): an empty-value dimension line is skipped, not a fatal IndexError.

    A line like 'schema_validity:' made '"".split()[0]' raise IndexError; only ValueError was suppressed, so
    judge_config's outer try/except caught it and silently downgraded the ENTIRE LLM judge to the offline
    heuristic on any single malformed line. Now IndexError is suppressed too, so one malformed dimension is
    skipped (stays 0.0) and the remaining dimensions still parse.
    """
    scores = _parse_judge_scores("schema_validity:\ncoverage_appropriateness: 2\nqc_pass: 3")
    assert scores["schema_validity"] == 0.0  # empty value -> skipped (default 0.0), no raise
    assert scores["coverage_appropriateness"] == 2.0  # remaining dimensions still parsed
    assert scores["qc_pass"] == 3.0
