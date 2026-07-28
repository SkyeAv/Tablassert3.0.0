"""Tests for US-005 ``build_and_audit``: ONE deterministic validate->build->QC->coverage mega-tool.

The ``build_and_audit`` core is PURE (base deps + the real Rust redb only) and runs in
the base environment with NO ``importorskip``: it drives the REAL ``validate_pipeline``
+ ``build_pipeline`` (headless ``_NullProgress``) inside an isolated ``workdir`` against
a tiny REAL redb (the offline recipe from ``tests/test_e2e_smoke.py``: ``brca1`` ->
HGNC:1100, ``mapk1`` -> HGNC:6871). The smolagents ``Tool`` wrapper test calls
``pytest.importorskip("smolagents")``. Every test is offline + fast and uses ``tmp_path``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import build_and_audit, make_build_and_audit_tool


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _synonym_row(curie: str, preferred_name: str, names: list[str], category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


def _build_real_redb(root: Path) -> Path:
    """Build a tiny REAL fullmap redb: ``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871."""
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [_class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [_synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "brca1"], "Gene"), _synonym_row("HGNC:6871", "MAPK1", ["MAPK1", "mapk1"], "Gene")],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


@pytest.fixture
def redb(tmp_path: Path) -> Path:
    """Offline real redb under ``tmp_path`` (absolute path; no chdir needed to build it)."""
    return _build_real_redb(tmp_path / "fullmap")


def _write_table(tmp_path: Path, text: str) -> Path:
    data: Path = tmp_path / "data.tsv"
    data.write_text(text)
    return data


def _section_config(data: Path) -> dict[str, Any]:
    """A bare merged Section config: column A subject, column B object, PMC provenance."""
    return {
        "source": {"kind": "text", "local": str(data), "url": "https://example.com/data.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


# --------------------------------------------------------------------------- #
# PURE core tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


def test_build_and_audit_happy_path(tmp_path: Path, redb: Path) -> None:
    """A valid 2-row config builds a KG end-to-end: ok=True, full coverage, artifacts on disk."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\nbrca1\tmapk1\n")
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is True
    assert result["coverage_pct"] == 1.0
    node_count = result["node_count"]
    assert isinstance(node_count, int)
    assert node_count > 0
    edge_count = result["edge_count"]
    assert isinstance(edge_count, int)
    assert edge_count > 0
    assert result["qc_pass_rate"] is None  # qc defaults False

    kgx_path = result["kgx_path"]
    assert isinstance(kgx_path, str)
    assert Path(kgx_path).is_file()
    edges_path = result["edges_path"]
    assert isinstance(edges_path, str)
    assert Path(edges_path).is_file()

    assert result["errors"] == []
    assert result["error_codes"] == []
    assert result["unresolved"] == []


def test_build_and_audit_bad_predicate_surfaces_coded_error(tmp_path: Path, redb: Path) -> None:
    """A bad predicate enum fails validation: ok=False, coded message verbatim, no artifact, no raise."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: dict[str, Any] = _section_config(data)
    cfg["statement"]["predicate"] = "NOT_A_PREDICATE"

    result = build_and_audit(_yaml(cfg), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert errors
    assert isinstance(errors[0], str)
    assert errors[0]
    # The coded error surfaces its code and/or the offending field; the docs URL is appended.
    codes = result["error_codes"]
    assert isinstance(codes, list)
    assert codes or "predicate" in errors[0]
    assert result["kgx_path"] is None
    assert result["node_count"] == 0


def test_build_and_audit_missing_source_fails(tmp_path: Path, redb: Path) -> None:
    """A config missing ``source`` fails validation cleanly: ok=False, no exception."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: dict[str, Any] = _section_config(data)
    del cfg["source"]

    result = build_and_audit(_yaml(cfg), fullmap=redb, workdir=tmp_path)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert errors
    assert result["kgx_path"] is None


def test_build_and_audit_not_yaml(tmp_path: Path, redb: Path) -> None:
    """Invalid YAML and non-mapping YAML both fail cleanly (ok=False), never raising."""
    # Invalid YAML -> YAMLError path.
    bad = build_and_audit("[", fullmap=redb, workdir=tmp_path)
    assert bad["ok"] is False
    bad_errors = bad["errors"]
    assert isinstance(bad_errors, list)
    assert bad_errors

    # Valid YAML but not a mapping -> "config is not a YAML mapping".
    not_mapping = build_and_audit("- 1\n- 2\n", fullmap=redb, workdir=tmp_path)
    assert not_mapping["ok"] is False
    nm_errors = not_mapping["errors"]
    assert isinstance(nm_errors, list)
    assert nm_errors
    assert not_mapping["kgx_path"] is None


def test_build_and_audit_empty_table_no_traceback(tmp_path: Path, redb: Path) -> None:
    """An empty (0 data row) table never tracebacks: whatever the pipeline does, it returns a dict.

    The guarantee is NO uncaught exception; the build may succeed with 0 counts or fail
    with a clean coded error depending on how polars reads an empty source.
    """
    data: Path = _write_table(tmp_path, "")  # exists but has no rows
    result = build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)

    assert isinstance(result, dict)
    assert isinstance(result["ok"], bool)
    assert isinstance(result["errors"], list)
    assert isinstance(result["node_count"], int)


def test_build_and_audit_cwd_preserved(tmp_path: Path, redb: Path) -> None:
    """The internal ``contextlib.chdir`` is scoped: the caller's cwd is unchanged after the call."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    before: str = os.getcwd()
    build_and_audit(_yaml(_section_config(data)), fullmap=redb, workdir=tmp_path)
    assert os.getcwd() == before


# --------------------------------------------------------------------------- #
# Tool wrapper test (requires the [agent] extra; skips cleanly when absent)
# --------------------------------------------------------------------------- #


def test_build_and_audit_tool(tmp_path: Path, redb: Path) -> None:
    """The lazily-built tool binds the fullmap via closure and returns a JSON audit report."""
    pytest.importorskip("smolagents")
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    tool = make_build_and_audit_tool(lambda: redb)
    assert tool.name == "build_and_audit"

    parsed: dict[str, Any] = json.loads(tool.forward(_yaml(_section_config(data))))
    assert "ok" in parsed
    assert "coverage_pct" in parsed
    assert parsed["ok"] is True


def test_build_and_audit_measures_relative_source_with_correct_cwd(tmp_path: Path, redb: Path) -> None:
    """Regression (review fix 2): coverage is measured INSIDE the build's chdir(workdir).

    A config with a RELATIVE source ``local`` builds fine (the pipeline chdir's into workdir) but used to
    be measured by map_coverage from the ORIGINAL cwd -> frame reproduction failed -> false/unmeasurable
    coverage. build_and_audit now measures inside the same chdir(workdir), so a relative path resolves and
    coverage is a real measurement (1.0 here), never an 'unmeasurable' note.
    """
    workdir: Path = tmp_path / "work"
    workdir.mkdir(parents=True)
    (workdir / "rel.tsv").write_text("brca1\tmapk1\nbrca1\tmapk1\n")
    cfg: dict[str, Any] = {
        "source": {"kind": "text", "local": "rel.tsv", "url": "https://example.com/rel.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }
    report: dict[str, Any] = build_and_audit(_yaml(cfg), fullmap=redb, workdir=workdir)
    assert report["ok"] is True
    assert report["coverage_pct"] == 1.0  # brca1/mapk1 resolve -> a REAL measurement, not vacuous
    assert not any("unmeasurable" in str(note) for note in report["errors"])
