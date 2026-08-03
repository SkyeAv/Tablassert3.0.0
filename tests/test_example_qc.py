"""Tests for the example QC scripts (examples/agent/qc/): redaction, allowlist, column derivation,
judge-response validation, and the parseability of the committed QC_REPORT.md config fences."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

QC_DIR = Path(__file__).resolve().parents[1] / "examples" / "agent" / "qc"


def _load(alias: str, script: str = "") -> Any:
    """Load an example QC script as a module (they are run as scripts, not installed).

    ``alias`` is the module name to register; ``script`` is the file stem (defaults to ``alias``)
    so the same script can be loaded under several aliases for test isolation.
    """
    if str(QC_DIR) not in sys.path:
        sys.path.insert(0, str(QC_DIR))  # qc_reviewer imports qc_report from its own directory
    target = QC_DIR / f"{script or alias}.py"
    spec = importlib.util.spec_from_file_location(alias, target)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def qc_mods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    report = _load("qc_report_under_test", "qc_report")
    reviewer = _load("qc_reviewer_under_test", "qc_reviewer")
    # point both at a scratch state dir (module STATE_DIR defaults parse pytest's argv)
    monkeypatch.setattr(report, "STATE_DIR", tmp_path)
    monkeypatch.setattr(reviewer, "STATE_DIR", tmp_path)
    return report, reviewer


def test_redact_paths_normalizes_state_dir_and_blocks_local_paths(qc_mods: tuple[Any, Any], tmp_path: Path) -> None:
    report, _ = qc_mods
    text = f"local={tmp_path}/downloads/PMC1/x.xlsx other=/home/skyeav/Desktop/fullmap url=https://host.gov/PMC1/x.xlsx ratio=2.00/3"
    out = report.redact_paths(text)
    assert "<state-dir>/downloads/PMC1/x.xlsx" in out
    assert "/home/skyeav" not in out
    assert "<local-path>" in out
    assert "https://host.gov/PMC1/x.xlsx" in out  # public URLs survive
    assert "ratio=2.00/3" in out  # bare ratios are not paths


def test_get_table_summary_rejects_out_of_root_paths_without_reading(
    qc_mods: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, reviewer = qc_mods

    def explode(*a: object, **k: object) -> object:
        raise AssertionError("read_table must NOT be called for a path outside the downloads allowlist")

    monkeypatch.setattr("tablassert.agent.read_table", explode)
    outside = {"source": {"local": str(tmp_path / "elsewhere" / "secret.xlsx")}}
    assert "outside the QC downloads dir" in reviewer.get_table_summary({"template": outside})


def test_get_table_summary_reads_paths_inside_downloads(qc_mods: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, reviewer = qc_mods
    table = tmp_path / "downloads" / "PMC1" / "t.csv"
    table.parent.mkdir(parents=True)
    table.write_text("a,b\n1,2\n")

    seen: dict[str, object] = {}

    def fake_read_table(source: object, *, sheet: object = None, max_rows: int = 200, max_cols: int = 40) -> str:
        seen.update({"source": str(source), "sheet": sheet, "max_cols": max_cols})
        return "SUMMARY"

    monkeypatch.setattr("tablassert.agent.read_table", fake_read_table)
    cfg = {"template": {"source": {"local": str(table), "sheet": "data"}}}
    assert reviewer.get_table_summary(cfg) == "SUMMARY"
    assert seen["source"] == str(table.resolve())
    assert seen["sheet"] == "data"


def test_required_max_cols_covers_configured_columns() -> None:
    reviewer = _load("qc_reviewer_cols", "qc_reviewer")
    # subject/object/annotation columns up to T (ordinal 20) must widen past the 12 default
    sec = {
        "source": {"local": "x.xlsx"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "T"},
            "annotations": [{"name": "n", "method": "column", "column": "S"}],
        },
    }
    assert reviewer._required_max_cols(sec) == 20
    # no/few configured columns keep the 12 floor
    assert reviewer._required_max_cols({"statement": {"subject": {"encoding": "A"}}}) == 12
    # fixed-value encodings (CURIEs) are not columns
    assert reviewer._required_max_cols({"statement": {"subject": {"encoding": "CHEBI:9168"}}}) == 12
    # capped at read_table's 40-column ceiling
    assert reviewer._required_max_cols({"statement": {"object": {"encoding": "ZZ"}}}) == 40


def test_get_table_summary_accumulates_all_sections(qc_mods: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, reviewer = qc_mods
    calls: list[str] = []

    def fake_read_table(source: object, **k: object) -> str:
        calls.append(str(source))
        return f"SUMMARY-{len(calls)}"

    monkeypatch.setattr("tablassert.agent.read_table", fake_read_table)
    for pmc_dir in ("d1", "d2"):
        p = tmp_path / "downloads" / pmc_dir
        p.mkdir(parents=True)
        (p / "t.csv").write_text("a\n1\n")
    cfg = {
        "sections": [
            {"source": {"local": str(tmp_path / "downloads" / "d1" / "t.csv")}},
            {"source": {"local": str(tmp_path / "downloads" / "d2" / "t.csv")}},
        ]
    }
    out = reviewer.get_table_summary(cfg)
    assert len(calls) == 2  # every section examined, not just the first readable one
    assert "SUMMARY-1" in out
    assert "SUMMARY-2" in out


@pytest.mark.parametrize(
    "bad",
    [
        {},  # missing dimensions
        "not a dict",
        {"predicate_appropriateness": [1, 2]},  # dimension not a mapping
    ],
)
def test_validate_review_result_rejects_bad_shapes(bad: object) -> None:
    reviewer = _load("qc_reviewer_validate", "qc_reviewer")
    with pytest.raises(reviewer.InvalidReview):
        reviewer.validate_review_result(bad)


def test_validate_review_result_rejects_out_of_range_score_and_quality() -> None:
    reviewer = _load("qc_reviewer_validate2", "qc_reviewer")

    def make(score: object = 2, quality: object = "good") -> dict:
        return {**{d: {"score": score, "problem": "", "suggestion": ""} for d in reviewer._REVIEW_DIMS}, "overall_quality": quality}

    reviewer.validate_review_result(make())  # valid baseline passes
    with pytest.raises(reviewer.InvalidReview):
        reviewer.validate_review_result(make(score=5))
    with pytest.raises(reviewer.InvalidReview):
        reviewer.validate_review_result(make(score=True))  # bool is not a score
    with pytest.raises(reviewer.InvalidReview):
        reviewer.validate_review_result(make(quality="excellent"))


def test_committed_qc_report_yaml_fences_parse() -> None:
    """CI guard: every fenced config block in the committed QC_REPORT.md must be valid YAML
    (a truncated fence would fail safe_load — regression of the 3000-char truncation)."""
    report_md = (Path(__file__).resolve().parents[1] / "examples" / "agent" / "QC_REPORT.md").read_text()
    fences = re.findall(r"```yaml\n(.*?)```", report_md, re.DOTALL)
    assert fences, "QC_REPORT.md should contain fenced config blocks"
    for fence in fences:
        if fence.strip() == "(no config produced)":
            continue
        yaml.safe_load(fence)  # raises on truncated/invalid YAML


def test_committed_qc_artifacts_share_config_hashes() -> None:
    """Each PMC entry in QC_REVIEW.md carries the same config sha256 as QC_REPORT.md."""
    root = Path(__file__).resolve().parents[1] / "examples" / "agent"
    report = (root / "QC_REPORT.md").read_text()
    review = (root / "QC_REVIEW.md").read_text()
    rep_sha = dict(re.findall(r"## (PMC\d+)[^\n]*\n.*?- \*\*config sha256:\*\* `([0-9a-f]+)`", report, re.DOTALL))
    rev_sha = dict(re.findall(r"## (PMC\d+) — (?:\*\*\w+\*\*|review failed[^\n]*?) \(config sha256: `([0-9a-f]+)`\)", review))
    assert rep_sha
    assert rev_sha
    assert rep_sha == rev_sha
