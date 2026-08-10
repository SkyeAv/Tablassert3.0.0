"""Tests for the concurrency-safe SHARED graph registry (``<state-dir>/graph.yaml``).

Several ``tablassert agent`` processes pointed at the same ``--state-dir`` self-register their
successful builds into ONE aggregate graph config. Covered here: upsert/dedupe/replace-by-pmc
semantics, fresh creation, corrupt-file quarantine + self-healing rebuild, the first-wins fullmap
rule (with a warning on mismatch), TRUE multi-process concurrency over one registry path, and
``rebuild_graph`` reconstruction/pruning from ``state.json``. The registry module needs no extras;
``rebuild_graph`` reaches ``tablassert.agent.load_state`` lazily, which is lazy-import safe in the
base environment too.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest
import yaml

from tablassert.agent import ConfigRecord, SupervisorState, save_state
from tablassert.graph_registry import GRAPH_DESCRIPTION, GRAPH_NAME, GRAPH_VERSION, rebuild_graph, register_build
from tablassert.models import Graph


def _make_config(tmp_path: Path, pmc_id: str, body: str = "sections: []\n") -> Path:
    """Write a stand-in best-config file for ``pmc_id`` (content is irrelevant to the registry)."""
    config: Path = tmp_path / "configs" / f"{pmc_id}.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(body)
    return config


def _read_registry(state_dir: Path) -> dict[str, Any]:
    """Parse the registry YAML and assert it satisfies ``Graph.model_validate``."""
    path: Path = state_dir / "graph.yaml"
    assert path.is_file(), "the registry must exist"
    data: object = yaml.safe_load(path.read_text())
    assert isinstance(data, dict)
    Graph.model_validate(data)
    return data


def test_register_build_creates_fresh_registry(tmp_path: Path) -> None:
    """First registration creates the template registry with one ABSOLUTE entry + resolved fullmap."""
    state_dir: Path = tmp_path / "state"
    config: Path = _make_config(tmp_path, "PMC1")
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()

    path: Path = register_build(state_dir, "PMC1", config, fullmap)

    assert path == state_dir / "graph.yaml"
    data: dict[str, Any] = _read_registry(state_dir)
    assert data["name"] == GRAPH_NAME
    assert data["version"] == GRAPH_VERSION
    assert data["description"] == GRAPH_DESCRIPTION
    assert data["tables"] == [str(config.resolve())]
    assert Path(data["tables"][0]).is_absolute()
    assert data["fullmap"] == str(fullmap.resolve())
    assert not (state_dir / "graph.yaml.tmp").exists(), "the atomic write must not leave a tmp behind"


def test_register_build_replaces_same_pmc_and_preserves_order(tmp_path: Path) -> None:
    """Re-registering a pmc REPLACES its entry (matched by basename stem); others keep their order."""
    state_dir: Path = tmp_path / "state"
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()
    first: Path = _make_config(tmp_path, "PMC1", "v: 1\n")
    other: Path = _make_config(tmp_path, "PMC2")

    register_build(state_dir, "PMC1", first, fullmap)
    register_build(state_dir, "PMC2", other, fullmap)

    # Re-run of PMC1 with a NEW config path: the old entry is dropped, the new one appended last.
    rerun: Path = tmp_path / "configs-rerun" / "PMC1.yaml"
    rerun.parent.mkdir()
    rerun.write_text("v: 2\n")
    register_build(state_dir, "PMC1", rerun, fullmap)

    data: dict[str, Any] = _read_registry(state_dir)
    assert data["tables"] == [str(other.resolve()), str(rerun.resolve())]
    assert [Path(str(t)).stem for t in data["tables"]].count("PMC1") == 1, "exactly one entry per pmc id"
    assert str(first.resolve()) not in data["tables"], "the replaced entry must be gone"


def test_register_build_preserves_unrelated_entries(tmp_path: Path) -> None:
    """Entries whose basename stem differs from the pmc id survive the upsert untouched."""
    state_dir: Path = tmp_path / "state"
    state_dir.mkdir()
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()
    foreign: Path = _make_config(tmp_path, "hand-authored")
    seed: dict[str, Any] = {
        "name": GRAPH_NAME,
        "version": GRAPH_VERSION,
        "description": GRAPH_DESCRIPTION,
        "tables": [str(foreign.resolve())],
        "fullmap": str(fullmap.resolve()),
    }
    (state_dir / "graph.yaml").write_text(yaml.safe_dump(seed, sort_keys=False))

    config: Path = _make_config(tmp_path, "PMC9")
    register_build(state_dir, "PMC9", config, fullmap)

    data: dict[str, Any] = _read_registry(state_dir)
    assert data["tables"] == [str(foreign.resolve()), str(config.resolve())]


@pytest.mark.parametrize(
    "payload",
    [
        "name: [unclosed",  # YAML parse error
        "- just\n- a list\n",  # top level is not a mapping
        yaml.safe_dump({"name": "not-a-graph"}),  # valid mapping, fails Graph.model_validate
    ],
    ids=["yaml-error", "not-a-mapping", "schema-invalid"],
)
def test_corrupt_registry_quarantined_and_rebuilt(payload: str, tmp_path: Path) -> None:
    """A corrupt registry is renamed ``graph.yaml.corrupt-<UTC ts>`` and rebuilt fresh (self-healing)."""
    state_dir: Path = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "graph.yaml").write_text(payload)
    config: Path = _make_config(tmp_path, "PMC1")
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()

    path: Path = register_build(state_dir, "PMC1", config, fullmap)

    quarantined: list[Path] = sorted(state_dir.glob("graph.yaml.corrupt-*"))
    assert len(quarantined) == 1, "exactly one quarantine file"
    assert quarantined[0].read_text() == payload, "the corrupt bytes are preserved"
    data: dict[str, Any] = _read_registry(state_dir)
    assert path == state_dir / "graph.yaml"
    assert data["tables"] == [str(config.resolve())], "the rebuild starts fresh"


def test_corrupt_registry_quarantine_on_rebuild_graph(tmp_path: Path) -> None:
    """``rebuild_graph`` self-heals the same way before reconstructing from ``state.json``."""
    state_dir: Path = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "graph.yaml").write_text("{{{::: not yaml")
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()

    rebuild_graph(state_dir, fullmap)

    assert len(list(state_dir.glob("graph.yaml.corrupt-*"))) == 1
    data: dict[str, Any] = _read_registry(state_dir)
    assert data["tables"] == [], "no state.json -> empty tables"
    assert data["fullmap"] == str(fullmap.resolve())


def test_fullmap_first_wins_and_warns_on_mismatch(tmp_path: Path) -> None:
    """The first fullmap recorded stays; a differing later fullmap is rejected with a warning."""
    from tablassert.log import logger

    state_dir: Path = tmp_path / "state"
    fm1: Path = tmp_path / "fm1.redb"
    fm2: Path = tmp_path / "fm2.redb"
    fm1.touch()
    fm2.touch()
    register_build(state_dir, "PMC1", _make_config(tmp_path, "PMC1"), fm1)

    records: list[str] = []
    sink_id: int = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        register_build(state_dir, "PMC2", _make_config(tmp_path, "PMC2"), fm2)
    finally:
        logger.remove(sink_id)

    data: dict[str, Any] = _read_registry(state_dir)
    assert data["fullmap"] == str(fm1.resolve()), "fullmap is FIRST-WINS"
    assert any("first-wins" in record for record in records), f"expected a fullmap-mismatch warning, got {records}"

    # A repeat of the SAME fullmap never warns.
    records.clear()
    sink_id = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        register_build(state_dir, "PMC3", _make_config(tmp_path, "PMC3"), fm1)
    finally:
        logger.remove(sink_id)
    assert not any("first-wins" in record for record in records)


def _concurrent_register(args: tuple[str, str, str, str]) -> str:
    """Worker: register ONE pmc config from a child process (module-level => picklable)."""
    state_dir, pmc_id, config_path, fullmap = args
    from tablassert.graph_registry import register_build as worker_register

    worker_register(Path(state_dir), pmc_id, Path(config_path), Path(fullmap))
    return pmc_id


def test_concurrent_registrations_converge(tmp_path: Path) -> None:
    """TRUE concurrency: >=8 processes registering DISTINCT pmc ids at ONE path -> exactly N entries.

    Every process contends on the same ``graph.yaml.lock``; the flock + atomic write must serialize
    them so no registration is lost, duplicated, or torn (the result parses and validates).
    """
    state_dir: Path = tmp_path / "shared"
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()
    n: int = 10
    tasks: list[tuple[str, str, str, str]] = []
    for i in range(n):
        pmc_id: str = f"PMC{i}"
        config: Path = _make_config(tmp_path, pmc_id)
        tasks.append((str(state_dir), pmc_id, str(config), str(fullmap)))

    with multiprocessing.Pool(processes=n) as pool:
        done: list[str] = pool.map(_concurrent_register, tasks)

    assert sorted(done) == sorted(f"PMC{i}" for i in range(n))
    data: dict[str, Any] = _read_registry(state_dir)  # parses + Graph.model_validate passes
    tables: list[Any] = data["tables"]
    assert len(tables) == n, "no lost updates"
    assert len(set(tables)) == n, "no duplicates"
    assert {Path(str(t)).stem for t in tables} == {f"PMC{i}" for i in range(n)}
    assert all(Path(str(t)).is_absolute() for t in tables)
    assert data["fullmap"] == str(fullmap.resolve())


def _seed_state(state_dir: Path, records: dict[str, ConfigRecord]) -> None:
    """Persist a supervisor checkpoint with the given records."""
    save_state(state_dir, SupervisorState(pmc_ids=sorted(records), records=records))


def test_rebuild_graph_prunes_and_excludes(tmp_path: Path) -> None:
    """``rebuild_graph`` keeps registered statuses with existing configs, sorted; prunes everything else."""
    state_dir: Path = tmp_path / "state"
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()
    pmc1: Path = _make_config(tmp_path, "PMC1")  # MAPPED, exists
    pmc2: Path = _make_config(tmp_path, "PMC2")  # BUILT_UNMEASURED, exists
    _make_config(tmp_path, "PMC3")  # MAPPED, but deleted before rebuild
    (tmp_path / "configs" / "PMC3.yaml").unlink()
    _make_config(tmp_path, "PMC4")  # SKIPPED, exists on disk but must NOT register
    stale: Path = _make_config(tmp_path, "stale-leftover")

    # Pre-seed the registry with a stale entry that no record supports (must be pruned).
    state_dir.mkdir()
    seed: dict[str, Any] = {
        "name": GRAPH_NAME,
        "version": GRAPH_VERSION,
        "description": GRAPH_DESCRIPTION,
        "tables": [str(stale.resolve())],
        "fullmap": str(fullmap.resolve()),
    }
    (state_dir / "graph.yaml").write_text(yaml.safe_dump(seed, sort_keys=False))

    _seed_state(
        state_dir,
        {
            "PMC1": ConfigRecord(pmc_id="PMC1", status="MAPPED", best_config_path=str(pmc1)),
            "PMC2": ConfigRecord(pmc_id="PMC2", status="BUILT_UNMEASURED", best_config_path=str(pmc2)),
            "PMC3": ConfigRecord(pmc_id="PMC3", status="MAPPED", best_config_path=str(tmp_path / "configs" / "PMC3.yaml")),
            "PMC4": ConfigRecord(pmc_id="PMC4", status="SKIPPED", best_config_path=str(tmp_path / "configs" / "PMC4.yaml")),
        },
    )

    path: Path = rebuild_graph(state_dir, fullmap)

    assert path == state_dir / "graph.yaml"
    data: dict[str, Any] = _read_registry(state_dir)
    assert data["tables"] == [str(pmc1.resolve()), str(pmc2.resolve())], "sorted by pmc id; stale/missing/SKIPPED pruned"


def test_rebuild_graph_tolerates_corrupt_state_json(tmp_path: Path) -> None:
    """A damaged ``state.json`` warns + rebuilds an empty registry instead of raising (recovery path)."""
    state_dir: Path = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "state.json").write_text("{not json at all")
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()

    path: Path = rebuild_graph(state_dir, fullmap)  # must NOT raise

    data: dict[str, Any] = _read_registry(state_dir)
    assert path == state_dir / "graph.yaml"
    assert data["tables"] == []
    assert data["fullmap"] == str(fullmap.resolve())


def test_rebuild_graph_without_state_creates_empty_registry(tmp_path: Path) -> None:
    """No ``state.json`` at all -> an empty-but-valid registry carrying the fullmap."""
    state_dir: Path = tmp_path / "state"
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()

    path: Path = rebuild_graph(state_dir, fullmap)

    assert path.is_file()
    data: dict[str, Any] = _read_registry(state_dir)
    assert data["tables"] == []
    assert data["fullmap"] == str(fullmap.resolve())


def test_rebuild_graph_is_first_wins_for_fullmap(tmp_path: Path) -> None:
    """An existing registry fullmap survives a ``rebuild_graph`` that passes a different one."""
    state_dir: Path = tmp_path / "state"
    fm1: Path = tmp_path / "fm1.redb"
    fm2: Path = tmp_path / "fm2.redb"
    fm1.touch()
    fm2.touch()
    config: Path = _make_config(tmp_path, "PMC1")
    register_build(state_dir, "PMC1", config, fm1)
    _seed_state(state_dir, {"PMC1": ConfigRecord(pmc_id="PMC1", status="MAPPED", best_config_path=str(config))})

    rebuild_graph(state_dir, fm2)

    data: dict[str, Any] = _read_registry(state_dir)
    assert data["fullmap"] == str(fm1.resolve()), "rebuild honors the first-wins fullmap"
