"""Tests for the stable on-disk workspace layout used by the agent supervisor.

This module pins the CANONICAL ``<root>/{state.json, configs/, downloads/<pmc>/,
builds/<pmc>/}`` layout so every producer/consumer (fetch, derive, build, reuse)
agrees on where artifacts live. US-501 lands the FOUNDATION only: the PURE
pathlib resolver in ``tablassert.agent`` (``artifact_root`` + the ``*_dir`` /
``*_path`` helpers). The download / config / build / cross-run-REUSE behavior
tests land in LATER stories — this file currently holds ONLY the ``-k layout``
resolver tests.

Every test here is PURE + offline: the helpers are stdlib pathlib with NO
smolagents / network / fullmap needed, so they run in the BASE environment
(``tablassert.agent`` is import-light; the heavy ``[agent]`` stack is lazy). We
therefore import the helpers directly and do NOT ``importorskip("smolagents")``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tablassert.agent import (
    artifact_root,
    best_config_path,
    builds_dir,
    configs_dir,
    derived_config_path,
    downloads_dir,
    pmc_build_dir,
    pmc_download_dir,
)


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the module hermetically offline by disabling HuggingFace telemetry.

    Why: matches the supervisor suite's convention so ANY future test added here
    that touches ``agent.run`` never blocks on a network telemetry call. The pure
    layout tests never touch the network, so this is defensive + costs nothing.
    """
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# US-501: PURE workspace-layout resolver
# --------------------------------------------------------------------------- #


def test_layout_artifact_root_override_and_default(tmp_path: Path) -> None:
    """``artifact_root`` selects ``workdir`` when given, else falls back to ``state_dir``.

    Why: the whole point of the resolver is that bulky downloads/builds can live
    off the (possibly small / shared) state dir via ``workdir``, while a ``None``
    workdir co-locates artifacts with state. Pinning both branches guards the
    override contract every downstream caller relies on.
    """
    state_dir: Path = tmp_path / "state"
    workdir: Path = tmp_path / "w"
    assert artifact_root(state_dir, None) == state_dir
    assert artifact_root(state_dir) == state_dir  # default arg is None
    assert artifact_root(state_dir, workdir) == workdir


def test_layout_download_dirs(tmp_path: Path) -> None:
    """Download paths resolve to ``<root>/downloads`` and ``<root>/downloads/<pmc>``.

    Why: fetch writes one directory per article under a shared ``downloads/``
    parent; the exact join is the contract the fetch + reuse stories build on.
    """
    root: Path = tmp_path
    assert downloads_dir(root) == root / "downloads"
    assert pmc_download_dir(root, "PMC1") == root / "downloads" / "PMC1"


def test_layout_config_paths(tmp_path: Path) -> None:
    """Config paths live under ``<root>/configs/`` with ``.yaml`` / ``.derived.yaml`` names.

    Why: the agent-derived config and the accepted BEST config must be distinct,
    deterministically-named files in a shared ``configs/`` dir so the improve loop
    and reuse can find them without coordination.
    """
    root: Path = tmp_path
    assert configs_dir(root) == root / "configs"
    assert best_config_path(root, "PMC1") == root / "configs" / "PMC1.yaml"
    assert derived_config_path(root, "PMC1") == root / "configs" / "PMC1.derived.yaml"
    # Both configs share the configs/ parent and differ only by suffix.
    assert best_config_path(root, "PMC1").parent == configs_dir(root)
    assert derived_config_path(root, "PMC1").parent == configs_dir(root)


def test_layout_build_dirs(tmp_path: Path) -> None:
    """Build paths resolve to ``<root>/builds`` and ``<root>/builds/<pmc>``.

    Why: build_and_audit writes one output directory per article under a shared
    ``builds/`` parent; the exact join is the contract the build story builds on.
    """
    root: Path = tmp_path
    assert builds_dir(root) == root / "builds"
    assert pmc_build_dir(root, "PMC1") == root / "builds" / "PMC1"


def test_layout_helpers_do_no_io(tmp_path: Path) -> None:
    """The resolver is PURE: calling every helper creates NOTHING on disk.

    Why: callers (not the resolver) own ``mkdir`` at write time. If a helper
    silently created directories it would couple path resolution to I/O, break
    dry-run/planning paths, and make the layout untestable without a filesystem.
    """
    root: Path = tmp_path / "fresh"  # does not exist yet
    resolved: list[Path] = [
        artifact_root(root, None),
        downloads_dir(root),
        pmc_download_dir(root, "PMC1"),
        configs_dir(root),
        best_config_path(root, "PMC1"),
        derived_config_path(root, "PMC1"),
        builds_dir(root),
        pmc_build_dir(root, "PMC1"),
    ]
    assert resolved, "sanity: every helper returned a path"
    assert not root.exists(), "the resolver must not create the root"
    assert not (root / "configs").exists(), "the resolver must not mkdir configs/"
    assert not (root / "downloads").exists(), "the resolver must not mkdir downloads/"
    assert not (root / "builds").exists(), "the resolver must not mkdir builds/"


# --------------------------------------------------------------------------- #
# US-502: downloads land in a STABLE ``<art_root>/downloads/<pmc>/`` dir
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


@pytest.fixture
def fullmap_db(tmp_path: Path) -> Path:
    """A tiny REAL fullmap redb: ``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871."""
    from tablassert import rs

    root: Path = tmp_path / "fullmap"
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [{"id": "HGNC:1100", "equivalent_identifiers": [{"identifier": "NCBIGene:672"}]}])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [
            {"curie": "HGNC:1100", "preferred_name": "BRCA1", "names": ["BRCA1", "brca1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]},
            {"curie": "HGNC:6871", "preferred_name": "MAPK1", "names": ["MAPK1", "mapk1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]},
        ],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
    return output


def test_supervisor_downloads_to_stable_dir(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``workdir=None``, ``run_supervisor`` fetches into ``state_dir/downloads/<pmc>`` (REQ-LAYOUT-3/7).

    Why: downloads must be STABLE + pipeline-reusable, NOT a throwaway ``tempfile.mkdtemp``
    dir that vanishes with the process. The unified CLI layout (``workdir=None``) co-locates
    artifacts under ``state_dir``, so the fetch outdir must be exactly
    ``pmc_download_dir(state_dir, pmc)`` and must PERSIST after the run so downstream
    stages (derive/build/reuse) can find the payload without re-fetching.
    """
    pytest.importorskip("smolagents")
    import yaml

    from tablassert.agent import make_fake_model, run_supervisor

    state_dir: Path = tmp_path / "state"
    expected_outdir: Path = pmc_download_dir(state_dir, "PMC1")  # == state_dir/downloads/PMC1

    recorded: list[Path] = []

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        recorded.append(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        table: Path = outdir / "good.tsv"
        table.write_text("brca1\tmapk1\nbrca1\tmapk1\n")
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)

    good_yaml: str = yaml.safe_dump(
        {
            "source": {"kind": "text", "local": str(expected_outdir / "good.tsv"), "url": "https://e.com/d.tsv", "delimiter": "\t"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "predicate": "associated_with",
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {"repo": "PMC", "publication": "PMC1"},
        },
        sort_keys=False,
    )

    run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=state_dir,
        # workdir omitted => None => the unified CLI layout (art_root == state_dir)
    )

    assert recorded == [expected_outdir], "fetch outdir must be the stable pmc_download_dir, not a temp dir"
    assert expected_outdir == state_dir / "downloads" / "PMC1"
    assert expected_outdir.is_dir(), "the stable downloads dir must persist after the run"
    assert (expected_outdir / "good.tsv").is_file(), "the fetched payload must persist under the stable dir"


# --------------------------------------------------------------------------- #
# US-503: ALL configs land in ONE dedicated ``<state_dir>/configs/`` folder
# --------------------------------------------------------------------------- #


def test_supervisor_writes_configs_to_configs_folder(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``workdir=None``, ``run_supervisor`` writes BOTH configs into ``state_dir/configs/`` (REQ-LAYOUT-4/6).

    Why: every config artifact (the agent-derived ``<pmc>.derived.yaml`` AND the accepted BEST
    ``<pmc>.yaml``) must live in ONE dedicated ``configs/`` folder so reuse/cleanup can find them
    without scanning the state dir, while ``state.json`` stays at the state-dir ROOT (NEVER inside
    ``configs/``). This pins the US-503 contract and guards against regressing to the old FLAT
    ``state_dir/<pmc>.yaml`` location.
    """
    pytest.importorskip("smolagents")
    import yaml

    from tablassert.agent import ConfigRecord, make_fake_model, run_supervisor

    state_dir: Path = tmp_path / "state"
    download_dir: Path = pmc_download_dir(state_dir, "PMC1")  # == state_dir/downloads/PMC1

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        outdir.mkdir(parents=True, exist_ok=True)
        table: Path = outdir / "good.tsv"
        table.write_text("brca1\tmapk1\nbrca1\tmapk1\n")
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)

    good_yaml: str = yaml.safe_dump(
        {
            "source": {"kind": "text", "local": str(download_dir / "good.tsv"), "url": "https://e.com/d.tsv", "delimiter": "\t"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "predicate": "associated_with",
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {"repo": "PMC", "publication": "PMC1"},
        },
        sort_keys=False,
    )

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=state_dir,
        # workdir omitted => None => configs follow state_dir (cfg_root == state_dir)
    )

    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    rec: ConfigRecord = records["PMC1"]
    assert rec.status == "MAPPED", f"fake config covers both genes; expected MAPPED, got {rec.status}: {rec.notes}"

    configs: Path = state_dir / "configs"
    best: Path = configs / "PMC1.yaml"
    derived: Path = configs / "PMC1.derived.yaml"
    # BOTH configs live in the ONE dedicated configs/ folder, and the record points there.
    assert best.is_file(), "the BEST config must be written to configs/<pmc>.yaml"
    assert derived.is_file(), "the derived config must be written to configs/<pmc>.derived.yaml"
    assert rec.best_config_path == str(best), "best_config_path must point into configs/"
    assert rec.config_path == str(best), "config_path must point at the BEST config in configs/"
    # state.json stays at the state-dir ROOT, never inside configs/.
    assert (state_dir / "state.json").is_file(), "state.json must persist at the state-dir root"
    assert not (configs / "state.json").exists(), "state.json must NEVER live inside configs/"
    # The old FLAT location is gone.
    assert not (state_dir / "PMC1.yaml").exists(), "no stray flat BEST config at the old state_dir/<pmc>.yaml"
    assert not (state_dir / "PMC1.derived.yaml").exists(), "no stray flat derived config at the old location"


# --------------------------------------------------------------------------- #
# US-504: STABLE builds dir + pipeline-reuse contract
# --------------------------------------------------------------------------- #


def test_supervisor_builds_to_stable_builds_dir(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``workdir=None``, a MAPPED run persists the KGX artifacts under ``state_dir/builds/<pmc>/`` (REQ-LAYOUT-5/8).

    Why: build outputs must be STABLE + discoverable, NOT a throwaway ``tempfile.mkdtemp`` dir that
    vanishes with the process. The unified CLI layout (``workdir=None``) co-locates builds under
    ``state_dir``, so ``build_and_audit``'s workdir must be exactly ``pmc_build_dir(state_dir, pmc)``
    and the KGX ``<name>_<version>.{nodes,edges}.ndjson`` (name/version default to ``agent``/``0.0.1``)
    must PERSIST there after the run so downstream stages can find the graph without rebuilding.
    """
    pytest.importorskip("smolagents")
    import yaml

    from tablassert.agent import ConfigRecord, make_fake_model, run_supervisor

    state_dir: Path = tmp_path / "state"
    download_dir: Path = pmc_download_dir(state_dir, "PMC1")  # == state_dir/downloads/PMC1

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        outdir.mkdir(parents=True, exist_ok=True)
        table: Path = outdir / "good.tsv"
        table.write_text("brca1\tmapk1\nbrca1\tmapk1\n")
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)

    good_yaml: str = yaml.safe_dump(
        {
            "source": {"kind": "text", "local": str(download_dir / "good.tsv"), "url": "https://e.com/d.tsv", "delimiter": "\t"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "predicate": "associated_with",
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {"repo": "PMC", "publication": "PMC1"},
        },
        sort_keys=False,
    )

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=state_dir,
        # workdir omitted => None => builds follow state_dir (art_root == state_dir)
    )

    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    rec: ConfigRecord = records["PMC1"]
    assert rec.status == "MAPPED", f"fake config covers both genes; expected MAPPED, got {rec.status}: {rec.notes}"

    build_dir: Path = pmc_build_dir(state_dir, "PMC1")  # == state_dir/builds/PMC1
    assert build_dir == state_dir / "builds" / "PMC1"
    nodes: Path = build_dir / "agent_0.0.1.nodes.ndjson"
    edges: Path = build_dir / "agent_0.0.1.edges.ndjson"
    assert nodes.is_file(), "the KGX nodes artifact must persist under builds/<pmc>/"
    assert edges.is_file(), "the KGX edges artifact must persist under builds/<pmc>/"
    assert nodes.stat().st_size > 0, "the nodes artifact must be non-empty"
    assert edges.stat().st_size > 0, "the edges artifact must be non-empty"
    # The old FLAT build location (art_root/<pmc>) is gone.
    assert not (state_dir / "PMC1" / "agent_0.0.1.nodes.ndjson").exists(), "no stray flat build at the old state_dir/<pmc>/"


def test_supervisor_best_config_pipeline_reuse(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The BEST config references the STABLE download and rebuilds from a FRESH cwd (REQ-LAYOUT-5/8).

    Why: the pipeline-reuse contract. ``tablassert build-kg <cfg> --table-config --fullmap <fm>`` must
    be able to reuse the supervisor's accepted config WITHOUT re-fetching: its ``source.local`` must be
    the REAL, persisted download under ``state_dir/downloads/<pmc>/`` (not a temp path), and because that
    path is ABSOLUTE the config must build from ANY cwd. This proves the download is real + referenced
    and that the best config is self-sufficient for downstream reuse.
    """
    pytest.importorskip("smolagents")
    import yaml

    from tablassert.agent import ConfigRecord, build_and_audit, make_fake_model, run_supervisor

    state_dir: Path = tmp_path / "state"
    download_dir: Path = pmc_download_dir(state_dir, "PMC1")  # == state_dir/downloads/PMC1
    stable_table: Path = download_dir / "good.tsv"  # the deterministic download path

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        # Mimic the REAL fetch: write the payload INTO the outdir we are given, so the download
        # lands at the deterministic state_dir/downloads/PMC1/good.tsv (not a throwaway temp dir).
        outdir.mkdir(parents=True, exist_ok=True)
        table: Path = outdir / "good.tsv"
        table.write_text("brca1\tmapk1\nbrca1\tmapk1\n")
        return [table]

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)

    good_yaml: str = yaml.safe_dump(
        {
            "source": {"kind": "text", "local": str(stable_table), "url": "https://e.com/d.tsv", "delimiter": "\t"},
            "statement": {
                "subject": {"method": "column", "encoding": "A"},
                "predicate": "associated_with",
                "object": {"method": "column", "encoding": "B"},
            },
            "provenance": {"repo": "PMC", "publication": "PMC1"},
        },
        sort_keys=False,
    )

    result = run_supervisor(
        ["PMC1"],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=state_dir,
        # workdir omitted => None => the unified CLI layout (art_root == state_dir)
    )

    records: dict[str, ConfigRecord] = result["records"]  # pyright: ignore[reportAssignmentType]
    rec: ConfigRecord = records["PMC1"]
    assert rec.status == "MAPPED", f"fake config covers both genes; expected MAPPED, got {rec.status}: {rec.notes}"

    # (1) The BEST config parses and its source.local is the REAL, persisted download under downloads/.
    best: Path = best_config_path(state_dir, "PMC1")  # == state_dir/configs/PMC1.yaml
    assert best.is_file(), "the BEST config must be written to configs/<pmc>.yaml"
    best_cfg: dict[str, Any] = yaml.safe_load(best.read_text())
    section: dict[str, Any] = best_cfg.get("template", best_cfg)  # supervisor writes the bare merged section
    local_path: Path = Path(section["source"]["local"])
    assert local_path.is_file(), "the best config's source.local must exist as a real file (download is real)"
    assert downloads_dir(state_dir) in local_path.parents, "source.local must live under the stable downloads/ dir"
    assert local_path == stable_table, "source.local must be the deterministic download path (download is referenced)"

    # (2) That best config BUILDS from a FRESH cwd via the absolute source.local (pipeline reuse).
    fresh: Path = tmp_path / "fresh-reuse-cwd"
    report: dict[str, object] = build_and_audit(best.read_text(), fullmap=fullmap_db, workdir=fresh)
    assert report["ok"] is True, f"the best config must rebuild from a fresh cwd: {report.get('errors')}"
