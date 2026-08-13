from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tablassert.cli as cli
import tablassert.study as study


def _write_ndjson(path: Path, lines: list[str]) -> Path:
    """Write raw lines (already JSON-encoded or deliberately malformed) to ``path``."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _records(*records: dict[str, Any]) -> list[str]:
    return [json.dumps(record) for record in records]


def _clean(tmp_path: Path) -> tuple[Path, Path]:
    nodes: Path = _write_ndjson(tmp_path / "g_1.nodes.ndjson", _records({"id": "HGNC:5"}, {"id": "HGNC:6"}))
    edges: Path = _write_ndjson(tmp_path / "g_1.edges.ndjson", _records({"subject": "HGNC:5", "object": "HGNC:6", "predicate": "biolink:related_to"}))
    return nodes, edges


def _checks(violations: list[study.StudyViolation]) -> dict[str, study.StudyViolation]:
    return {violation.check: violation for violation in violations}


def test_clean_files_pass(tmp_path: Path) -> None:
    """Well-formed nodes/edges files produce no violations."""
    nodes, edges = _clean(tmp_path)
    assert study.study_kgx(nodes, edges) == []


def test_duplicate_node_ids(tmp_path: Path) -> None:
    """A node id appearing on more than one line fails the duplicate assertion."""
    nodes: Path = _write_ndjson(tmp_path / "n.ndjson", _records({"id": "HGNC:5"}, {"id": "HGNC:5", "name": "different"}))
    _, edges = _clean(tmp_path)
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, edges))
    assert checks["duplicate-node-ids"].count == 1
    assert checks["duplicate-node-ids"].examples == ["HGNC:5"]


def test_duplicate_examples_capped(tmp_path: Path) -> None:
    """Duplicate-id examples are capped while the count stays exact."""
    nodes: Path = _write_ndjson(tmp_path / "n.ndjson", _records(*({"id": f"X:{i}"} for i in range(30)), *({"id": f"X:{i}"} for i in range(30))))
    _, edges = _clean(tmp_path)
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, edges, example_limit=10))
    assert checks["duplicate-node-ids"].count == 30
    assert len(checks["duplicate-node-ids"].examples) == 10


def test_undeclared_nodes(tmp_path: Path) -> None:
    """Edge subject/object ids missing from the nodes file fail the undeclared assertion."""
    nodes: Path = _write_ndjson(tmp_path / "n.ndjson", _records({"id": "HGNC:5"}))
    edges: Path = _write_ndjson(tmp_path / "e.ndjson", _records({"subject": "HGNC:5", "object": "HGNC:6"}, {"subject": "HGNC:7", "object": "HGNC:5"}))
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, edges))
    assert checks["undeclared-nodes"].count == 2
    assert checks["undeclared-nodes"].examples == ["HGNC:6", "HGNC:7"]


def test_isolated_nodes(tmp_path: Path) -> None:
    """Declared nodes participating in no edge fail the isolated assertion."""
    nodes: Path = _write_ndjson(tmp_path / "n.ndjson", _records({"id": "HGNC:5"}, {"id": "HGNC:6"}, {"id": "HGNC:7"}))
    edges: Path = _write_ndjson(tmp_path / "e.ndjson", _records({"subject": "HGNC:5", "object": "HGNC:6"}))
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, edges))
    assert checks["isolated-nodes"].count == 1
    assert checks["isolated-nodes"].examples == ["HGNC:7"]


def test_malformed_lines(tmp_path: Path) -> None:
    """Empty lines, invalid JSON, and non-object lines all count as malformed."""
    nodes, _ = _clean(tmp_path)
    edges: Path = _write_ndjson(tmp_path / "e.ndjson", [json.dumps({"subject": "HGNC:5", "object": "HGNC:6"}), "", "{not json", '["a list"]'])
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, edges))
    assert checks["malformed-lines"].label == "edges"
    assert checks["malformed-lines"].count == 3


def test_whitespace_values(tmp_path: Path) -> None:
    """Leading/trailing whitespace in string values is counted per field."""
    nodes: Path = _write_ndjson(tmp_path / "n.ndjson", _records({"id": "HGNC:5", "name": " padded"}, {"id": "HGNC:6 ", "name": "x"}))
    edges: Path = _write_ndjson(tmp_path / "e.ndjson", _records({"subject": "HGNC:5", "object": "HGNC:6"}))
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, edges))
    violation: study.StudyViolation = checks["whitespace-values"]
    assert violation.label == "nodes"
    assert violation.count == 2
    assert sorted(violation.examples) == ["id (1)", "name (1)"]


def test_missing_file_is_a_violation(tmp_path: Path) -> None:
    """A missing path must never read as a clean bill of health."""
    nodes, _ = _clean(tmp_path)
    checks: dict[str, study.StudyViolation] = _checks(study.study_kgx(nodes, tmp_path / "nope.edges.ndjson"))
    assert checks["file-missing"].label == "edges"


def test_format_violations(tmp_path: Path) -> None:
    """The stderr summary renders one line per failed assertion with examples."""
    violations: list[study.StudyViolation] = [study.StudyViolation("duplicate-node-ids", "nodes", 3, ["HGNC:5", "HGNC:6"])]
    assert study.format_violations(violations) == "nodes: 3 duplicate node ids (e.g. HGNC:5, HGNC:6)"


def test_study_final_ndjson_exits_on_violations(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    """The --qc build stage fails the build (SystemExit 1) when assertions are violated."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study, "study_kgx", lambda *args: [study.StudyViolation("isolated-nodes", "nodes", 1, ["HGNC:7"])])
    with pytest.raises(SystemExit) as excinfo:
        cli.study_final_ndjson("g", "1")
    assert excinfo.value.code == 1
    assert "no edge" in capsys.readouterr().err


def test_study_final_ndjson_passes_clean(monkeypatch: Any, tmp_path: Path) -> None:
    """The --qc build stage returns normally when every assertion passes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study, "study_kgx", lambda *args: [])
    cli.study_final_ndjson("g", "1")
