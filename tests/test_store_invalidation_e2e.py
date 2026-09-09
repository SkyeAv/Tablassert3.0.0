"""End-to-end coverage for content-aware section-store invalidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from tablassert import rs
from tablassert.cli import build_pipeline
from tablassert.ingests import from_yaml, to_sections, to_yaml
from tablassert.progress import PipelineProgress
from tablassert.utils import mkhash


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _build_real_redb(root: Path) -> Path:
    """Build the tiny real fullmap used by these invalidation smokes."""
    root.mkdir(parents=True, exist_ok=True)
    classes = _write_jsonl(root / "classes.ndjson", [{"id": "HGNC:1100", "equivalent_identifiers": [{"identifier": "NCBIGene:672"}]}])
    synonyms = _write_jsonl(
        root / "synonyms.ndjson",
        [
            {"curie": "HGNC:1100", "preferred_name": "BRCA1", "names": ["BRCA1", "brca1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]},
            {"curie": "HGNC:6871", "preferred_name": "MAPK1", "names": ["MAPK1", "mapk1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]},
        ],
    )
    output = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms])
    return output


def _write_build_inputs(tmp_path: Path, rig_factory: Any, source_text: str) -> tuple[Path, Path, Path]:
    source = tmp_path / "data.tsv"
    source.write_text(source_text)
    table = tmp_path / "table.yaml"
    to_yaml(
        table,
        {
            "template": {
                "source": {"kind": "text", "local": str(source), "url": ["https://example.com/data.tsv"], "delimiter": "\t"},
                "statement": {
                    "subject": {"method": "column", "encoding": "A"},
                    "predicate": "associated_with",
                    "object": {"method": "column", "encoding": "B"},
                },
                "provenance": {"repo": "PMC", "publication": "PMC0000000"},
            }
        },
    )
    fullmap = _build_real_redb(tmp_path / "fullmap")
    graph = tmp_path / "graph.yaml"
    to_yaml(
        graph,
        {
            "name": "INVALIDATION_KG",
            "version": "1.0.0",
            "tables": [str(table)],
            "fullmap": str(fullmap),
            "rig": rig_factory(tmp_path, infores_id="infores:invalidation-kg", source_info={"description": "store invalidation smoke graph"}),
        },
    )
    return source, graph, table


def _store_files() -> dict[str, Path]:
    return {path.name: path for path in Path(".tablassert/store").glob("*.parquet")}


def _build(graph: Path) -> None:
    build_pipeline(graph, PipelineProgress(total_stages=6))


def test_unchanged_build_reuses_store_and_preserves_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig_factory: Any) -> None:
    """An unchanged build must quick-exit the section cache without changing artifacts.

    WHY: content hashing should invalidate stale source data, but a repeat build with
    identical bytes must retain the existing parquet and byte-identical KGX output.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert/store").mkdir(parents=True)
    _, graph, _ = _write_build_inputs(tmp_path, rig_factory, "brca1\tmapk1\n")

    _build(graph)
    edges = tmp_path / "INVALIDATION_KG_1.0.0.edges.ndjson"
    nodes = tmp_path / "INVALIDATION_KG_1.0.0.nodes.ndjson"
    first_outputs = (nodes.read_bytes(), edges.read_bytes())
    first_mtimes = {name: path.stat().st_mtime_ns for name, path in _store_files().items()}
    assert first_mtimes

    _build(graph)
    assert (nodes.read_bytes(), edges.read_bytes()) == first_outputs
    assert {name: path.stat().st_mtime_ns for name, path in _store_files().items()} == first_mtimes


def test_source_edit_writes_new_store_and_new_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig_factory: Any) -> None:
    """Adding a source row rekeys the store and makes the row reach the final edges.

    WHY: a config-only cache key would quick-exit the old parquet and silently omit
    assertions added to an otherwise unchanged TSV configuration.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert/store").mkdir(parents=True)
    source, graph, _ = _write_build_inputs(tmp_path, rig_factory, "brca1\tmapk1\n")

    _build(graph)
    old_stores = _store_files()
    source.write_text("brca1\tmapk1\nmapk1\tbrca1\n")
    _build(graph)
    new_stores = _store_files()

    assert set(old_stores) < set(new_stores)
    edge_rows = [json.loads(line) for line in (tmp_path / "INVALIDATION_KG_1.0.0.edges.ndjson").read_text().splitlines() if line]
    assert {(row["subject"], row["object"]) for row in edge_rows} == {("HGNC:1100", "HGNC:6871"), ("HGNC:6871", "HGNC:1100")}


def test_legacy_store_orphan_is_not_a_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig_factory: Any) -> None:
    """A legacy config-only parquet remains untouched while a content-aware store is built.

    WHY: upgrading keying must orphan old stores safely; treating one as a hit could
    expose stale or incompatible rows while overwriting user data would destroy it.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert/store").mkdir(parents=True)
    _, graph, table = _write_build_inputs(tmp_path, rig_factory, "brca1\tmapk1\n")

    raw = from_yaml(table)
    section = to_sections(raw, table)[0]  # type: ignore[index]
    legacy = Path(".tablassert/store") / f"{mkhash(section)}.parquet"
    pl.DataFrame({"legacy_marker": ["must not be read"]}).write_parquet(legacy)
    marker_bytes = legacy.read_bytes()

    _build(graph)
    stores = _store_files()
    assert legacy.read_bytes() == marker_bytes
    assert len(stores) == 2
    assert any(path != legacy for path in stores.values())
    edges = (tmp_path / "INVALIDATION_KG_1.0.0.edges.ndjson").read_text()
    assert "HGNC:1100" in edges
    assert "HGNC:6871" in edges
