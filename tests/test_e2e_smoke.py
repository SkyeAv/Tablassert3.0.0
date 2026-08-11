"""Fast offline end-to-end smokes wiring the REAL Rust redb to the REAL Python pipeline.

These two smokes catch Rust<->Python contract drift in the quick suite instead of a
multi-hour ``build-kg`` run:

* ``test_build_pipeline_against_real_redb`` drives the full six-stage ``build_pipeline``
  against a tiny REAL ``rs.build_fullmap_db`` redb (no monkeypatched lookup) and asserts
  the KGX NDJSON output contains CURIEs that only the real redb could resolve.
* ``test_validate_command_happy_path`` exercises the ``validate`` cyclopts
  command wrapper (``cli.py``) plus ``validate_pipeline`` on a valid table config.

Both are additive, offline, and fast (<~5s each); all artifacts land in ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tablassert import rs
from tablassert.cli import build_pipeline, validate, validate_pipeline
from tablassert.ingests import to_yaml
from tablassert.progress import PipelineProgress


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


def test_build_pipeline_against_real_redb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SMOKE (i): the six-stage build resolves terms through a REAL fullmap redb end-to-end.

    The fullmap lookup is NOT monkeypatched: the only way ``HGNC:1100``/``HGNC:6871`` can
    appear in the KGX output is if the Python pipeline really called the Rust redb. Output
    is redirected by ``chdir`` (``compile_graph`` writes to cwd, ``STORE`` is cwd-relative)
    so the repo working tree stays clean.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".tablassert" / "store").mkdir(parents=True)

    fullmap: Path = _build_real_redb(tmp_path / "fullmap")

    # Two-row headerless text source: column A = subject term, column B = object term.
    data: Path = tmp_path / "data.tsv"
    data.write_text("brca1\tmapk1\nbrca1\tmapk1\n")

    table: Path = tmp_path / "table.yaml"
    table_config: dict[str, Any] = {
        "template": {
            "source": {"kind": "text", "local": str(data), "url": ["https://example.com/data.tsv"], "delimiter": "\t"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "predicate": "associated_with",
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        }
    }
    to_yaml(table, table_config)

    graph: Path = tmp_path / "graph.yaml"
    graph_config: dict[str, Any] = {
        "name": "SMOKE_KG",
        "version": "1.0.0",
        "description": "e2e smoke graph",
        "tables": [str(table)],
        "fullmap": str(fullmap),
    }
    to_yaml(graph, graph_config)

    build_pipeline(graph, PipelineProgress(total_stages=6))

    nodes: Path = tmp_path / "SMOKE_KG_1.0.0.nodes.ndjson"
    edges: Path = tmp_path / "SMOKE_KG_1.0.0.edges.ndjson"
    assert nodes.is_file()
    assert edges.is_file()

    node_text: str = nodes.read_text()
    edge_text: str = edges.read_text()
    assert node_text.strip()
    assert edge_text.strip()

    # These CURIEs exist only in the real redb; their presence proves the contract held.
    assert "HGNC:1100" in node_text
    assert "HGNC:6871" in node_text
    assert "HGNC:1100" in edge_text
    assert "HGNC:6871" in edge_text


def test_validate_command_happy_path(tmp_path: Path) -> None:
    """SMOKE (ii): ``validate`` (cli.py wrapper) + ``validate_pipeline`` accept a valid table.

    Mirrors ``tests/fixtures/minimal_section.yaml`` (value-encoded BRCA1/TP53, PMC
    provenance) wrapped in the ``template`` shape ``to_sections`` requires. Validation never
    reads the source data file, so the ``local`` path need not exist.
    """
    config: Path = tmp_path / "table.yaml"
    table_config: dict[str, Any] = {
        "template": {
            "source": {"kind": "text", "local": "./test.tsv", "url": ["https://example.com/test.tsv"], "delimiter": "\t"},
            "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        }
    }
    to_yaml(config, table_config)

    # The three-stage validate pipeline alone does not raise on a valid section.
    assert validate_pipeline(config, PipelineProgress(total_stages=3)) is None
    # The cyclopts command wrapper (cli.py validate -> run(3, validate_pipeline, ...)).
    assert validate(config, schema="table") is None
