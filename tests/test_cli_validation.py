from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tablassert.cli import build_pipeline, validate_pipeline
from tablassert.errors import GraphValidationError, SectionValidationError
from tablassert.ingests import to_yaml
from tablassert.progress import PipelineProgress


def test_validate_pipeline_rejects_section_missing_source(fixtures_path: Path) -> None:
    """Guard: `validate-table` fails fast on a table that cannot form a valid section.

    A table config with no usable source would otherwise surface only deep inside a
    multi-hour `build-graph` run; this pre-build check rejects it up front.
    """
    config: Path = fixtures_path / "invalid_section_missing_source.yaml"
    with pytest.raises(SectionValidationError) as exc_info:
        validate_pipeline(config, PipelineProgress(total_stages=3))
    assert exc_info.value.code == "section-validation-failed"


def test_build_pipeline_rejects_graph_referencing_invalid_section(tmp_path: Path, fixtures_path: Path) -> None:
    """Guard: `build-graph` fails fast when a referenced table has an invalid section.

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
    """Guard: `build-graph` fails fast on a graph config missing required keys.

    A graph without `tables`/`fullmap` cannot build; rejecting it at validation avoids
    a confusing failure deep inside a multi-hour build.
    """
    graph_file: Path = tmp_path / "graph.yaml"
    graph: dict[str, Any] = {"name": "TEST", "version": "1.0.0", "description": "test graph"}
    to_yaml(graph_file, graph)
    with pytest.raises(GraphValidationError) as exc_info:
        build_pipeline(graph_file, PipelineProgress(total_stages=6))
    assert exc_info.value.code == "graph-validation-failed"
