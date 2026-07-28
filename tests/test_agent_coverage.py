"""Tests for US-006 ``map_coverage``: fullmap term-resolution coverage.

The ``map_coverage`` core is PURE (base deps + the real Rust redb only) and runs in
the base environment with NO ``importorskip``. It reuses the production normalization
(``Tcode._source_ops`` + ``node_prep``) and the real ``fullmap.distinct`` /
``lookup_rows`` against a tiny REAL redb (the offline recipe from
``tests/test_e2e_smoke.py``: ``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871). The
smolagents ``Tool`` wrapper test calls ``pytest.importorskip("smolagents")``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import make_map_coverage_tool, map_coverage


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
def redb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Offline real redb + an isolated cwd (``.tablassert/store`` mirrors the e2e recipe)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)
    return _build_real_redb(tmp_path / "fullmap")


def _write_table(tmp_path: Path, text: str) -> Path:
    data: Path = tmp_path / "data.tsv"
    data.write_text(text)
    return data


def _section_config(data: Path, *, object_method: str = "column", object_encoding: str = "B") -> dict[str, Any]:
    """A bare merged Section config: column A subject, configurable object, PMC provenance."""
    return {
        "source": {"kind": "text", "local": str(data), "url": "https://example.com/data.tsv", "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": object_method, "encoding": object_encoding},
        },
        "provenance": {"repo": "PMC", "publication": "PMC0000000"},
    }


# --------------------------------------------------------------------------- #
# PURE core tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


def test_coverage_all_resolvable(tmp_path: Path, redb: Path) -> None:
    """Two resolvable terms (brca1/mapk1) give overall 1.0 and no unresolved terms."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\nbrca1\tmapk1\n")
    result = map_coverage(_section_config(data), fullmap=redb, workdir=tmp_path)
    assert result["overall"] == 1.0
    per_column = result["per_column"]
    assert isinstance(per_column, dict)
    assert per_column["subject"]["coverage"] == 1.0
    assert per_column["object"]["coverage"] == 1.0
    assert result["unresolved"] == []


def test_coverage_mixed_unresolvable(tmp_path: Path, redb: Path) -> None:
    """An unresolvable object term (zzznotreal) drops object coverage to 0.0, overall to 0.5."""
    data: Path = _write_table(tmp_path, "brca1\tzzznotreal\n")
    result = map_coverage(_section_config(data), fullmap=redb, workdir=tmp_path)
    per_column = result["per_column"]
    assert isinstance(per_column, dict)
    assert per_column["subject"]["coverage"] == 1.0
    assert per_column["object"]["coverage"] == 0.0
    assert result["overall"] == 0.5
    top_unresolved = result["unresolved"]
    assert isinstance(top_unresolved, list)
    assert "zzznotreal" in top_unresolved
    object_unresolved = per_column["object"]["unresolved"]
    assert isinstance(object_unresolved, list)
    assert "zzznotreal" in object_unresolved


def test_coverage_value_node_vacuous(tmp_path: Path, redb: Path) -> None:
    """A ``method: value`` object is a pre-resolved literal: vacuous, not counted against overall."""
    data: Path = _write_table(tmp_path, "brca1\tanything\n")
    cfg: dict[str, Any] = _section_config(data, object_method="value", object_encoding="CHEBI:41774")
    result = map_coverage(cfg, fullmap=redb, workdir=tmp_path)
    per_column = result["per_column"]
    assert isinstance(per_column, dict)
    assert per_column["object"]["method"] == "value"
    assert per_column["object"]["coverage"] == 1.0
    assert per_column["subject"]["method"] == "column"
    # Overall is computed over the subject column only (brca1 resolves) -> 1.0.
    assert result["overall"] == 1.0
    assert result["unresolved"] == []


def test_coverage_per_column_breakdown(tmp_path: Path, redb: Path) -> None:
    """Each per-column entry has the documented shape with int totals/resolved."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    result = map_coverage(_section_config(data), fullmap=redb, workdir=tmp_path)
    per_column = result["per_column"]
    assert isinstance(per_column, dict)
    assert set(per_column) == {"subject", "object"}
    for col in ("subject", "object"):
        entry = per_column[col]
        assert isinstance(entry, dict)
        assert set(entry) == {"coverage", "total", "resolved", "unresolved", "method"}
        assert isinstance(entry["total"], int)
        assert isinstance(entry["resolved"], int)
        assert isinstance(entry["coverage"], float)
        assert isinstance(entry["unresolved"], list)
        assert entry["method"] == "column"
        assert entry["total"] == 1
        assert entry["resolved"] == 1


def test_coverage_accepts_yaml_string(tmp_path: Path, redb: Path) -> None:
    """A YAML-string config parses identically to a dict config."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: str = yaml.safe_dump(_section_config(data), sort_keys=False)
    result = map_coverage(cfg, fullmap=redb, workdir=tmp_path)
    assert result["overall"] == 1.0


def test_coverage_accepts_template_config(tmp_path: Path, redb: Path) -> None:
    """A ``{template: {...}}`` table config is fast-merged via the shared helper."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: dict[str, Any] = {"template": _section_config(data)}
    result = map_coverage(cfg, fullmap=redb, workdir=tmp_path)
    assert result["overall"] == 1.0
    assert result["unresolved"] == []


def test_coverage_odd_config_is_vacuous(tmp_path: Path, redb: Path) -> None:
    """A config with no resolvable structure returns a vacuous perfect score, never raises."""
    # Not a dict after parsing -> vacuous empty result.
    assert map_coverage("just a string", fullmap=redb, workdir=tmp_path) == {"overall": 1.0, "per_column": {}, "unresolved": []}
    # A structurally invalid section (missing source) -> vacuous empty result, no crash.
    assert map_coverage({"statement": {"predicate": "associated_with"}}, fullmap=redb, workdir=tmp_path) == {
        "overall": 1.0,
        "per_column": {},
        "unresolved": [],
    }


def test_coverage_bad_fullmap_raises(tmp_path: Path, redb: Path) -> None:
    """A non-existent fullmap path raises from the redb lookup; it is NOT silently swallowed."""
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    bad: Path = tmp_path / "nope" / "fullmap.redb"
    # The Rust redb open surfaces the missing file as an I/O RuntimeError (verified).
    with pytest.raises(RuntimeError, match="I/O error"):
        map_coverage(_section_config(data), fullmap=bad, workdir=tmp_path)


# --------------------------------------------------------------------------- #
# Tool wrapper test (requires the [agent] extra; skips cleanly when absent)
# --------------------------------------------------------------------------- #


def test_map_coverage_tool_builds_and_forwards(tmp_path: Path, redb: Path) -> None:
    """The lazily-built tool binds the fullmap via closure and returns JSON with 'overall'."""
    pytest.importorskip("smolagents")
    data: Path = _write_table(tmp_path, "brca1\tmapk1\n")
    cfg: str = yaml.safe_dump(_section_config(data), sort_keys=False)
    tool = make_map_coverage_tool(lambda: redb)
    assert tool.name == "map_coverage"
    parsed: dict[str, Any] = json.loads(tool.forward(cfg))
    assert "overall" in parsed
    assert parsed["overall"] == 1.0
