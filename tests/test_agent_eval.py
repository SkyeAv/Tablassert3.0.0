"""Tests for US-011 eval harness: metrics + LLM-judge rubric + Reflexion + GEPA + Pareto.

The deterministic metrics, the offline judge heuristic, Reflexion, and the Pareto frontier are
PURE and run in the base environment (no ``[agent]`` extra). The dspy-dependent tests
(``gepa_metric`` returns a ``dspy.Prediction``; ``run_gepa`` wiring) call
``pytest.importorskip("dspy")`` so they skip cleanly without the extra. ``run_gepa`` is exercised
through an INJECTABLE ``gepa_cls`` stub so it terminates fast with NO network — the metric contract
is tested for real, the GEPA wiring via the stub. An autouse fixture disables HuggingFace telemetry
so any agent/dspy run stays hermetically offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tablassert.agent import (
    JUDGE_DIMENSIONS,
    _judge_provenance,
    cost_metric,
    coverage_metric,
    dominates,
    gepa_metric,
    judge_config,
    load_gepa_dataset,
    load_kgx,
    load_optimized_instructions,
    node_edge_f1,
    pareto_frontier,
    qc_pass_rate_metric,
    quality_score,
    reflexion_improve,
    reliability_metric,
    run_gepa,
    save_optimized_instructions,
    validate_section,
    validate_table_config,
)

FIXTURE_DIR: Path = Path(__file__).parent / "agent_fixtures" / "PMC11708054"
SECOND_FIXTURE_DIR: Path = Path(__file__).parent / "agent_fixtures" / "GENE_DISEASE"

# A genuinely valid minimal Section config (used wherever a schema-valid YAML string is needed).
VALID_CFG: str = yaml.safe_dump(
    {
        "source": {"kind": "text", "local": "./t.tsv", "url": "https://e.com/t.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "value", "encoding": "BRCA1"},
            "predicate": "associated_with",
            "object": {"method": "value", "encoding": "TP53"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC0000000"},
    }
)


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep any agent/dspy run hermetically offline (HF telemetry blocks on network)."""
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# Golden fixture
# --------------------------------------------------------------------------- #


def test_fixture_present_and_valid() -> None:
    """The golden fixture exists and its configs are schema-valid (the eval's ground truth)."""
    for name in ("ALAMV6.yaml", "reference_config.yaml", "source_table.csv", "README.md"):
        assert (FIXTURE_DIR / name).is_file(), f"missing fixture file {name}"
    assert validate_section((FIXTURE_DIR / "ALAMV6.yaml").read_text()) is True
    assert validate_section((FIXTURE_DIR / "reference_config.yaml").read_text()) is True
    assert (FIXTURE_DIR / "source_table.csv").read_text().strip()  # non-empty


# --------------------------------------------------------------------------- #
# Deterministic metrics (pure)
# --------------------------------------------------------------------------- #


def test_node_edge_f1_identical_and_disjoint_and_partial() -> None:
    """F1 is 1.0 for identical graphs, 0.0 for disjoint, and fractional for a partial overlap."""
    nodes = [{"id": "A"}, {"id": "B"}]
    edges = [{"subject": "A", "predicate": "p", "object": "B"}]
    identical = node_edge_f1(nodes, edges, nodes, edges)
    assert identical["node_f1"] == 1.0
    assert identical["edge_f1"] == 1.0

    disjoint = node_edge_f1([{"id": "X"}], [{"subject": "X", "predicate": "q", "object": "Y"}], nodes, edges)
    assert disjoint["node_f1"] == 0.0
    assert disjoint["edge_f1"] == 0.0

    # Built has A,B,C nodes; ref has A,B -> node precision 2/3, recall 1.0, f1 in (0,1).
    partial = node_edge_f1([{"id": "A"}, {"id": "B"}, {"id": "C"}], edges, nodes, edges)
    assert partial["node_precision"] == pytest.approx(2 / 3)
    assert partial["node_recall"] == 1.0
    assert 0.0 < partial["node_f1"] < 1.0


def test_quality_score_range_and_validity_gate() -> None:
    """quality_score is in [0,1] and an invalid config hard-gates to 0.0."""
    report = {"coverage_pct": 1.0, "qc_pass_rate": 1.0}
    f1 = {"node_f1": 1.0, "edge_f1": 1.0}
    score = quality_score(VALID_CFG, report, f1)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0)  # valid + full coverage + full qc + full f1
    # Invalid config -> hard gate 0.0 regardless of the (great) report.
    assert quality_score("statement: {}", report, f1) == 0.0


def test_metric_helpers() -> None:
    """coverage/qc/cost/reliability helpers extract the right fields with safe defaults."""
    report = {"coverage_pct": 0.7, "qc_pass_rate": 0.9}
    assert coverage_metric(report) == 0.7
    assert coverage_metric({"overall": 0.4}) == 0.4  # map_coverage shape
    assert coverage_metric({}) == 0.0
    assert qc_pass_rate_metric(report) == 0.9
    assert qc_pass_rate_metric({}) is None
    assert cost_metric({"total_tokens": 100, "steps": 5}) == {"tokens": 100, "steps": 5}
    assert reliability_metric({"failed_tool_calls": 1, "wrong_tool_calls": 2, "redundant_tool_calls": 3, "total_tool_calls": 9}) == {
        "failed": 1,
        "wrong": 2,
        "redundant": 3,
        "total_tool_calls": 9,
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge rubric (offline heuristic + bias helpers)
# --------------------------------------------------------------------------- #


def test_judge_offline_heuristic() -> None:
    """The offline judge scores all 7 semantic dimensions 0-3 with a normalized score in [0,1]."""
    result = judge_config(VALID_CFG, {"coverage_pct": 1.0, "qc_pass_rate": 1.0}, {"steps": 2})
    scores = result["scores"]
    assert set(scores) == set(JUDGE_DIMENSIONS)
    assert all(0 <= v <= 3 for v in scores.values())
    assert 0.0 <= result["normalized"] <= 1.0
    assert result["rationale"]
    # A schema-invalid config scores 0 on schema_validity.
    bad = judge_config("statement: {}", {"coverage_pct": 0.0}, {"steps": 20, "failed_tool_calls": 5})
    assert bad["scores"]["schema_validity"] == 0


def test_judge_with_fake_model_is_debiased() -> None:
    """A provided judge_model is called (both orderings) and yields a debiased dict; never raises."""
    calls: list[str] = []

    def fake_judge(prompt: str) -> str:
        calls.append(prompt)
        return "\n".join(f"{d}: 3" for d in JUDGE_DIMENSIONS)

    result = judge_config(VALID_CFG, {"coverage_pct": 1.0}, {"steps": 1}, judge_model=fake_judge)
    assert set(result["scores"]) == set(JUDGE_DIMENSIONS)
    assert all(v == 3.0 for v in result["scores"].values())
    assert len(calls) == 2  # position debiasing scores both dimension orders


def test_debias_helpers_pure() -> None:
    """Position averaging and verbosity penalty are pure and bounded."""
    from tablassert.agent import _debias_position, _debias_verbosity

    assert _debias_position(2.0, 4.0) == 3.0
    assert _debias_verbosity(0.8, 100, 100) == 0.8  # equal length -> identity
    assert _debias_verbosity(0.8, 300, 100) == pytest.approx(0.8 * 0.95)  # >2x inflation -> mild penalty


# --------------------------------------------------------------------------- #
# Reflexion-style retry (offline)
# --------------------------------------------------------------------------- #


def test_reflexion_improve_offline_schema_valid() -> None:
    """reflexion_improve returns a schema-valid config + non-empty reflections and never raises."""
    cfg = {
        "source": {"kind": "text", "local": "./t.tsv", "url": "https://e.com/t.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "correlated_with",
            "object": {"method": "value", "encoding": "CHEBI:41774"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }
    coverage_report = {
        "overall": 0.5,
        "per_column": {"subject": {"coverage": 0.5, "total": 2, "resolved": 1, "unresolved": ["g__Bacteroides"], "method": "column"}},
        "unresolved": ["g__Bacteroides"],
    }
    best, reflections = reflexion_improve(yaml.safe_dump(cfg), {"coverage_pct": 0.5, "errors": [], "error_codes": []}, coverage_report, fullmap=None)
    assert validate_section(best) is True
    assert reflections  # non-empty
    # Odd input never raises.
    best2, refl2 = reflexion_improve("::: not yaml", {}, {})
    assert isinstance(best2, str)
    assert isinstance(refl2, list)


# --------------------------------------------------------------------------- #
# Pareto frontier (pure)
# --------------------------------------------------------------------------- #


def test_dominates_and_pareto_frontier_knee() -> None:
    """Non-dominated set + knee match a hand-computed multi-objective case."""
    assert dominates({"quality": 1.0, "cost": 5, "wrong_calls": 0}, {"quality": 0.8, "cost": 5, "wrong_calls": 0}) is True
    assert dominates({"quality": 1.0, "cost": 5, "wrong_calls": 0}, {"quality": 1.0, "cost": 3, "wrong_calls": 0}) is False  # b is cheaper
    runs = [
        {"id": "a", "quality": 0.9, "cost": 10, "wrong_calls": 0},  # dominated by b (equal quality, b cheaper)
        {"id": "b", "quality": 0.9, "cost": 5, "wrong_calls": 0},  # non-dominated; quality/cost = 0.18
        {"id": "c", "quality": 0.5, "cost": 2, "wrong_calls": 0},  # non-dominated; quality/cost = 0.25 (the knee)
        {"id": "d", "quality": 0.4, "cost": 8, "wrong_calls": 3},  # dominated by c (worse on every objective)
    ]
    result = pareto_frontier(runs)
    assert set(result["frontier"]) == {"b", "c"}
    assert result["knee"] == "c"  # 0.5/2 = 0.25 beats b's 0.9/5 = 0.18


def test_pareto_knee_quality_per_cost() -> None:
    """The knee maximizes quality-per-unit-cost on the frontier (tie-break fewer wrong calls)."""
    runs = [
        {"id": "cheap", "quality": 0.5, "cost": 1, "wrong_calls": 0},  # ratio 0.5
        {"id": "pricey", "quality": 1.0, "cost": 10, "wrong_calls": 0},  # ratio 0.1
    ]
    result = pareto_frontier(runs)
    assert set(result["frontier"]) == {"cheap", "pricey"}
    assert result["knee"] == "cheap"  # 0.5/1 > 1.0/10


# --------------------------------------------------------------------------- #
# dspy GEPA (importorskip; metric for real, wiring via injectable stub)
# --------------------------------------------------------------------------- #


def test_gepa_metric_returns_prediction() -> None:
    """gepa_metric returns a dspy.Prediction(score in [0,1], feedback text); better input -> higher score."""
    pytest.importorskip("dspy")
    good = gepa_metric(
        {
            "config_yaml": VALID_CFG,
            "report": {"coverage_pct": 1.0, "qc_pass_rate": 1.0, "errors": [], "unresolved": []},
            "f1": {"node_f1": 1.0, "edge_f1": 1.0},
            "metrics": {},
        }
    )
    assert 0.0 <= good.score <= 1.0
    assert good.score > 0.9
    assert isinstance(good.feedback, str)

    bad = gepa_metric(
        {
            "config_yaml": "statement: {}",
            "report": {"coverage_pct": 0.1, "errors": ["bad predicate"], "error_codes": ["section-validation-failed"], "unresolved": ["zzz"]},
            "f1": {"node_f1": 0.0, "edge_f1": 0.0},
            "metrics": {"wrong_tool_calls": 2},
        }
    )
    assert bad.score < good.score
    assert any(token in bad.feedback for token in ("errors", "unresolved", "wrong_calls"))


def test_run_gepa_wiring_offline_via_stub() -> None:
    """run_gepa wires gepa_metric + pareto strategy into GEPA and extracts optimized instructions.

    An injectable StubGEPA stands in for dspy.GEPA so this terminates fast with NO network: it records
    the metric + kwargs it was built with and returns a fake compiled program whose predictor carries a
    known instruction. This proves the wiring (metric contract + pareto strategy + instruction extraction
    + stats pickup) without depending on GEPA's internal LM loop.
    """
    pytest.importorskip("dspy")
    created: dict[str, Any] = {}

    class StubGEPA:
        def __init__(self, metric: Any = None, **kwargs: Any) -> None:
            created["metric"] = metric
            created["kwargs"] = kwargs
            self.gepa_stats = {"rollouts": 1, "candidate_selection_strategy": "pareto"}

        def compile(self, program: Any, *, trainset: Any = None, **kwargs: Any) -> Any:
            predictor = SimpleNamespace(signature=SimpleNamespace(instructions="OPTIMIZED: prioritize OrganismTaxon, avoid Gene"))
            return SimpleNamespace(named_predictors=lambda: [("propose", predictor)])

    result = run_gepa(seed_instructions="SEED", gepa_cls=StubGEPA, reflection_lm=SimpleNamespace(), trainset=[])
    assert created["metric"] is gepa_metric
    assert created["kwargs"]["candidate_selection_strategy"] == "pareto"
    assert result["optimized_instructions"] == "OPTIMIZED: prioritize OrganismTaxon, avoid Gene"
    assert result["optimized_descriptions"].get("propose", "").startswith("OPTIMIZED")
    assert isinstance(result["stats"], dict)
    assert result["stats"].get("rollouts") == 1
    assert result["frontier"] == []


def test_run_gepa_default_example_path() -> None:
    """With trainset=None and no dataset, run_gepa builds a default dspy.Example and still terminates."""
    pytest.importorskip("dspy")

    class StubGEPA:
        def __init__(self, metric: Any = None, **kwargs: Any) -> None:
            self.gepa_stats: dict[str, Any] = {}

        def compile(self, program: Any, *, trainset: Any = None, **kwargs: Any) -> Any:
            # trainset must be a non-empty list of dspy.Example (the default path built one).
            assert trainset, "expected a default example to be built"
            predictor = SimpleNamespace(signature=SimpleNamespace(instructions="OPT"))
            return SimpleNamespace(named_predictors=lambda: [("propose", predictor)])

    result = run_gepa(seed_instructions="SEED", gepa_cls=StubGEPA, reflection_lm=SimpleNamespace())
    assert result["optimized_instructions"] == "OPT"


# --------------------------------------------------------------------------- #
# Offline integration: build the reference KGX from the fixture + score F1
# --------------------------------------------------------------------------- #


def _build_reference_redb(root: Path) -> Path:
    """Tiny REAL redb registering the fixture organisms (OrganismTaxon) + CHEBI:41774 (ChemicalEntity)."""
    from tablassert import rs

    root.mkdir(parents=True, exist_ok=True)

    organisms: list[str] = [
        "Lactobacillus rhamnosus",
        "Bacteroides fragilis",
        "Clostridium sp",
        "Escherichia coli",
        "Faecalibacterium prausnitzii",
        "Bifidobacterium longum",
        "Akkermansia muciniphila",
    ]

    def synonym_row(curie: str, preferred: str, names: list[str], category: str, taxon: str) -> dict[str, Any]:
        return {"curie": curie, "preferred_name": preferred, "names": names, "types": [category], "taxa": [taxon]}

    def class_row(curie: str) -> dict[str, Any]:
        return {"id": curie, "equivalent_identifiers": [{"identifier": curie}]}

    synonyms: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    for i, name in enumerate(organisms, start=100):
        curie = f"NCBITaxon:{i}"
        synonyms.append(synonym_row(curie, name, [name.lower(), name], "OrganismTaxon", "NCBITaxon:1"))
        classes.append(class_row(curie))
    synonyms.append(synonym_row("CHEBI:41774", "13C-tamoxifen", ["chebi:41774", "13c-tamoxifen"], "ChemicalEntity", "NCBITaxon:0"))
    classes.append(class_row("CHEBI:41774"))

    classes_path = root / "classes.ndjson"
    synonyms_path = root / "synonyms.ndjson"
    classes_path.write_text("\n".join(json.dumps(r) for r in classes) + "\n")
    synonyms_path.write_text("\n".join(json.dumps(r) for r in synonyms) + "\n")
    output = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes_path], [synonyms_path], threads=2)
    return output


def test_reference_kgx_builds_and_self_f1(tmp_path: Path) -> None:
    """The golden reference config builds offline against a tiny real redb; self-F1 is 1.0.

    Proves the fixture + load_kgx + node_edge_f1 + build_and_audit integrate end-to-end offline. The
    reference config's relative ``local`` path is rewritten to the absolute fixture CSV so the build
    (which chdir's into its workdir) finds the table.
    """
    from tablassert.agent import build_and_audit

    db = _build_reference_redb(tmp_path / "fullmap")
    cfg: dict[str, Any] = yaml.safe_load((FIXTURE_DIR / "reference_config.yaml").read_text())
    cfg["template"]["source"]["local"] = str(FIXTURE_DIR / "source_table.csv")
    config_yaml = yaml.safe_dump(cfg, sort_keys=False)

    report = build_and_audit(config_yaml, fullmap=db, workdir=tmp_path / "w")
    assert isinstance(report, dict)
    assert report["ok"] is True, f"reference build failed: {report.get('errors')}"
    kgx_path = Path(str(report["kgx_path"]))
    assert report["kgx_path"]
    assert kgx_path.is_file()

    ref_nodes = load_kgx(kgx_path)
    ref_edges = load_kgx(Path(str(report["edges_path"])))
    assert ref_nodes, "reference build produced no nodes (resolution failed?)"
    assert ref_edges, "reference build produced no edges (resolution failed?)"
    self_f1 = node_edge_f1(ref_nodes, ref_edges, ref_nodes, ref_edges)
    assert self_f1["node_f1"] == 1.0
    assert self_f1["edge_f1"] == 1.0


def test_judge_verbosity_debiasing_fires_with_baseline() -> None:
    """Regression (review fix 4): verbosity debiasing actually fires when a real baseline length is given.

    judge_config used to pass the config's OWN length as both config_len and baseline_len, so the ratio was
    always 1.0 and the verbosity penalty never triggered (docs overstated 'verbosity bias mitigation'). Now
    a caller can pass ``baseline_len`` (e.g. the reference config length); a config >2x the baseline incurs the
    5% penalty. With no baseline the config is compared against itself (ratio 1.0, no penalty).
    """

    def fake_judge(prompt: str) -> str:  # pyright: ignore[reportUnusedParameter]
        return "\n".join(f"{d}: 3" for d in JUDGE_DIMENSIONS)  # all dimensions score 3 -> raw normalized 1.0

    long_cfg: str = "x" * 100
    report: dict[str, Any] = {"coverage_pct": 1.0}
    metrics: dict[str, Any] = {"steps": 1}
    penalized: dict[str, Any] = judge_config(long_cfg, report, metrics, judge_model=fake_judge, baseline_len=10)
    unpenalized: dict[str, Any] = judge_config(long_cfg, report, metrics, judge_model=fake_judge)  # baseline defaults to own length
    assert penalized["normalized"] == pytest.approx(0.95)  # 1.0 * 0.95 verbosity penalty (ratio 100/10 > 2)
    assert unpenalized["normalized"] == pytest.approx(1.0)  # ratio 1.0 -> identity
    assert penalized["normalized"] < unpenalized["normalized"]


# --------------------------------------------------------------------------- #
# W6: optimized-instructions persist/load, GEPA dataset, smarter judge, second fixture
# --------------------------------------------------------------------------- #


def test_optimized_instructions_roundtrip(tmp_path: Path) -> None:
    """save_optimized_instructions -> load_optimized_instructions round-trips the prompt + descriptions."""
    out: Path = tmp_path / "optimized_instructions.yaml"
    save_optimized_instructions(out, "OPTIMIZED PROMPT", {"propose": "OPTIMIZED DESC"})
    assert load_optimized_instructions(out) == "OPTIMIZED PROMPT"
    # The persisted mapping also carries the descriptions.
    data: dict[str, Any] = yaml.safe_load(out.read_text())
    assert data["descriptions"] == {"propose": "OPTIMIZED DESC"}


def test_load_optimized_instructions_absent_and_bare(tmp_path: Path) -> None:
    """A missing file -> None; a bare YAML string of instructions loads directly."""
    assert load_optimized_instructions(tmp_path / "nope.yaml") is None
    bare: Path = tmp_path / "bare.yaml"
    bare.write_text("just a prompt string")
    assert load_optimized_instructions(bare) == "just a prompt string"


def test_load_optimized_instructions_unreadable_returns_none(tmp_path: Path) -> None:
    """CodeRabbit: an unreadable file (invalid UTF-8) yields None instead of aborting the run."""
    bad: Path = tmp_path / "bad.yaml"
    bad.write_bytes(b"\xff\xfe\x00not valid utf-8")  # read_text(encoding='utf-8') raises UnicodeDecodeError
    assert load_optimized_instructions(bad) is None


def test_load_gepa_dataset(tmp_path: Path) -> None:
    """load_gepa_dataset reads a YAML list of example dicts, dropping non-dict rows."""
    ds: Path = tmp_path / "dataset.yaml"
    ds.write_text(yaml.safe_dump([{"table_summary": "s1", "coverage_feedback": "c1"}, "not-a-dict", {"table_summary": "s2"}]))
    rows = load_gepa_dataset(ds)
    assert rows == [{"table_summary": "s1", "coverage_feedback": "c1"}, {"table_summary": "s2"}]


def test_judge_provenance_smarter() -> None:
    """W6 smarter heuristic: manual override -> 3, repo+pub -> 3, partial -> 1, none -> 0."""

    def cfg(provenance: dict[str, Any]) -> str:
        return yaml.safe_dump(
            {
                "source": {"kind": "text", "local": "./t.tsv", "url": "https://e.com/t.tsv", "delimiter": "\t"},
                "statement": {
                    "subject": {"method": "value", "encoding": "A"},
                    "predicate": "associated_with",
                    "object": {"method": "value", "encoding": "B"},
                },
                "provenance": provenance,
            }
        )

    assert _judge_provenance(cfg({"repo": "PMC", "publication": "PMC1"})) == 3
    assert _judge_provenance(cfg({"repo": "PMC", "publication": "PMC1", "override": {"upstream_resource_ids": ["infores:x"]}})) == 3
    assert _judge_provenance(cfg({"repo": "PMC"})) == 1  # partial credit (was 0)
    assert _judge_provenance(cfg({})) == 0


def test_second_fixture_present_and_valid() -> None:
    """The second golden fixture (gene~disease, multi-section shape) exists and validates section-by-section."""
    assert (SECOND_FIXTURE_DIR / "reference_config.yaml").is_file()
    assert (SECOND_FIXTURE_DIR / "source_table.csv").is_file()
    text: str = (SECOND_FIXTURE_DIR / "reference_config.yaml").read_text()
    assert validate_table_config(text) is True
    # It is a genuine multi-section config (template + sections), distinct from the PMC fixture.
    parsed: dict[str, Any] = yaml.safe_load(text)
    assert "sections" in parsed
    assert len(parsed["sections"]) >= 1


def test_second_fixture_offline_judge_scores() -> None:
    """The offline heuristic judge scores the second fixture (no judge model) with a sane normalized value."""
    text: str = (SECOND_FIXTURE_DIR / "reference_config.yaml").read_text()
    report: dict[str, Any] = {"coverage_pct": 1.0}
    metrics: dict[str, Any] = {"steps": 2}
    verdict: dict[str, Any] = judge_config(text, report, metrics)  # no judge_model -> offline heuristic
    assert 0.0 <= verdict["normalized"] <= 1.0
    assert verdict["scores"]["schema_validity"] == 3.0  # the fixture is schema-valid
    assert verdict["scores"]["provenance_completeness"] == 3.0  # repo + publication
