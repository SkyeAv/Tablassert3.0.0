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

from pathlib import Path

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
