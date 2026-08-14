"""Tests for caller-owned target graph preparation and atomic agent appends."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert.errors import GraphValidationError
from tablassert.graph_target import append_successful_config, prepare_graph
from tablassert.models import Graph


def _rig() -> dict[str, Any]:
    return {
        "source_info": {
            "infores_id": "infores:target-test",
            "terms_of_use_info": {"license_name": "CC0"},
            "data_access_locations": ["Test source - https://example.org/data"],
            "source_status": "unknown",
        },
        "ingest_info": {"utility": "Target test.", "scope": "Target test."},
        "provenance_info": {"contributions": ["Test author"]},
        "artifact_base_url": "https://example.org/target-test",
        "artifact_base_path": "./output",
    }


def _write_graph(path: Path, tables: list[str] | None = None) -> Path:
    path.write_text(
        yaml.safe_dump(
            {"name": "TARGET", "version": "2.0.0", "tables": [] if tables is None else tables, "fullmap": "./fullmap", "rig": _rig()}, sort_keys=False
        )
    )
    return path


def _write_config(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("template: {}\n")
    return path


def test_prepare_graph_resolves_execution_paths_without_rewriting_yaml(tmp_path: Path) -> None:
    target = _write_graph(tmp_path / "graph.yaml")
    before = target.read_text()

    prepared = prepare_graph(target)

    assert prepared.path == target.resolve()
    assert prepared.graph.fullmap == (tmp_path / "fullmap").resolve()
    assert prepared.graph.rig.artifact_base_path == (tmp_path / "output").resolve()
    assert target.read_text() == before


def test_append_preserves_metadata_and_replaces_matching_pmc(tmp_path: Path) -> None:
    old = _write_config(tmp_path / "old" / "PMC1.yaml")
    other = _write_config(tmp_path / "other" / "PMC2.yaml")
    target = _write_graph(tmp_path / "graph.yaml", [str(old), str(other)])
    new = _write_config(tmp_path / "new" / "PMC1.yaml")

    append_successful_config(target, "PMC1", new)

    data: dict[str, Any] = yaml.safe_load(target.read_text())
    assert data["name"] == "TARGET"
    assert data["version"] == "2.0.0"
    assert data["fullmap"] == "./fullmap"
    assert data["rig"] == _rig()
    assert data["tables"] == [str(other), str(new.resolve())]
    Graph.model_validate(data)
    assert (tmp_path / "graph.yaml.lock").is_file()
    assert not (tmp_path / ".graph.yaml.tmp").exists()


def test_append_rejects_invalid_target_without_changing_it(tmp_path: Path) -> None:
    target = tmp_path / "graph.yaml"
    target.write_text("name: not a graph\n")
    before = target.read_bytes()
    config = _write_config(tmp_path / "PMC1.yaml")

    with pytest.raises(GraphValidationError):
        append_successful_config(target, "PMC1", config)

    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.corrupt-*"))


def _append_worker(args: tuple[str, str, str]) -> str:
    target, pmc_id, config = args
    append_successful_config(Path(target), pmc_id, Path(config))
    return pmc_id


def test_concurrent_appends_do_not_lose_entries(tmp_path: Path) -> None:
    target = _write_graph(tmp_path / "graph.yaml")
    tasks: list[tuple[str, str, str]] = []
    for i in range(8):
        config = _write_config(tmp_path / "configs" / f"PMC{i}.yaml")
        tasks.append((str(target), f"PMC{i}", str(config)))

    with multiprocessing.Pool(processes=8) as pool:
        assert sorted(pool.map(_append_worker, tasks)) == [f"PMC{i}" for i in range(8)]

    data: dict[str, Any] = yaml.safe_load(target.read_text())
    tables: list[str] = data["tables"]
    assert len(tables) == 8
    assert {Path(table).stem for table in tables} == {f"PMC{i}" for i in range(8)}
    assert all(Path(table).is_absolute() for table in tables)
    Graph.model_validate(data)
