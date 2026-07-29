from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tablassert.cli import _load_graph, build_pipeline, validate, validate_pipeline
from tablassert.errors import GraphValidationError, SectionValidationError
from tablassert.ingests import to_yaml
from tablassert.models import DEFAULT_RIG_CONTRIBUTIONS, DEFAULT_RIG_UI_EXPLANATION, Graph
from tablassert.progress import PipelineProgress


def test_validate_pipeline_rejects_section_missing_source(fixtures_path: Path) -> None:
    """Guard: `validate` fails fast on a table that cannot form a valid section.

    A table config with no usable source would otherwise surface only deep inside a
    multi-hour `build-kg` run; this pre-build check rejects it up front.
    """
    config: Path = fixtures_path / "invalid_section_missing_source.yaml"
    with pytest.raises(SectionValidationError) as exc_info:
        validate_pipeline(config, PipelineProgress(total_stages=3))
    assert exc_info.value.code == "section-validation-failed"


def test_build_pipeline_rejects_graph_referencing_invalid_section(tmp_path: Path, fixtures_path: Path) -> None:
    """Guard: `build-kg` fails fast when a referenced table has an invalid section.

    A graph that points at a malformed table would otherwise fail deep inside a
    multi-hour build; this rejects it during section validation, before any work starts.
    """
    bad_table: Path = fixtures_path / "invalid_section_missing_source.yaml"
    graph_file: Path = tmp_path / "graph.yaml"
    graph: dict[str, Any] = {"name": "TEST", "version": "1.0.0", "description": "test graph", "tables": [str(bad_table)], "fullmap": ".fullmap"}
    to_yaml(graph_file, graph)
    with pytest.raises(SectionValidationError) as exc_info:
        build_pipeline(graph_file, PipelineProgress(total_stages=6))
    assert exc_info.value.code == "section-validation-failed"


def test_build_pipeline_rejects_malformed_graph(tmp_path: Path) -> None:
    """Guard: `build-kg` fails fast on a graph config missing required keys.

    A graph without `tables`/`fullmap` cannot build; rejecting it at validation avoids
    a confusing failure deep inside a multi-hour build.
    """
    graph_file: Path = tmp_path / "graph.yaml"
    graph: dict[str, Any] = {"name": "TEST", "version": "1.0.0", "description": "test graph"}
    to_yaml(graph_file, graph)
    with pytest.raises(GraphValidationError) as exc_info:
        build_pipeline(graph_file, PipelineProgress(total_stages=6))
    assert exc_info.value.code == "graph-validation-failed"


def _valid_table_config() -> dict[str, Any]:
    """Minimal valid table config (value-encoded; validation never reads the source file)."""
    return {
        "template": {
            "source": {"kind": "text", "local": "./test.tsv", "url": "https://example.com/test.tsv", "delimiter": "\t"},
            "statement": {"subject": {"method": "value", "encoding": "BRCA1"}, "object": {"method": "value", "encoding": "TP53"}},
            "provenance": {"repo": "PMC", "publication": "PMC0000000"},
        }
    }


def test_load_graph_wraps_table_config_as_temp_kg(fixtures_path: Path) -> None:
    """Guard: `build-kg --table-config` wraps a table YAML in a throwaway TEMP_KG graph.

    A single table config must build/test without authoring a full graph config. The
    wrapper fixes name/version/description, points `tables` at the config path, uses the
    given fullmap, and leaves contributions/ui_explanation on the Graph model defaults.
    """
    table: Path = fixtures_path / "minimal_section.yaml"
    g: Graph = _load_graph(table, table_config=True, fullmap=Path("./fullmap"))
    assert g.name == "TEMP_KG"
    assert g.version == "0.0.0"
    assert g.description == "Temporary knowledge graph generated to test a table configuration"
    assert g.tables == [table]
    assert g.fullmap == Path("./fullmap")
    assert g.contributions == DEFAULT_RIG_CONTRIBUTIONS
    assert g.ui_explanation == DEFAULT_RIG_UI_EXPLANATION


def test_validate_command_dispatches_graph_and_table(tmp_path: Path) -> None:
    """Guard: generic `validate` accepts both a table config and a graph config.

    `validate` detects the config kind from a top-level `tables` key: a table config
    validates section syntax only; a graph config validates the Graph model AND each
    referenced table. Both return None when valid.
    """
    table: Path = tmp_path / "table.yaml"
    to_yaml(table, _valid_table_config())
    # Table branch: no top-level `tables` key.
    assert validate(table) is None
    # Graph branch: top-level `tables` key -> validates the graph and its tables.
    graph_file: Path = tmp_path / "graph.yaml"
    to_yaml(graph_file, {"name": "TEST", "version": "1.0.0", "description": "test graph", "tables": [str(table)], "fullmap": ".fullmap"})
    assert validate(graph_file) is None


def test_validate_command_graph_branch_rejects_invalid_table(tmp_path: Path, fixtures_path: Path) -> None:
    """Guard: generic `validate` on a graph fails fast when a referenced table is invalid."""
    bad_table: Path = fixtures_path / "invalid_section_missing_source.yaml"
    graph_file: Path = tmp_path / "graph.yaml"
    to_yaml(graph_file, {"name": "TEST", "version": "1.0.0", "description": "test graph", "tables": [str(bad_table)], "fullmap": ".fullmap"})
    with pytest.raises(SectionValidationError) as exc_info:
        validate(graph_file)
    assert exc_info.value.code == "section-validation-failed"
