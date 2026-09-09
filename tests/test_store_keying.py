"""Stage 3 content-aware store-key tests.

These tests exercise the same key derivation seam used by ``build_graph_pipeline``
without running the unrelated fullmap and graph compilation stages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tablassert import cli
from tablassert.errors import SourceFileError
from tablassert.utils import section_store_key


def _section(source: Path | None, marker: str = "same") -> dict[str, Any]:
    """Build the smallest section-shaped mapping needed by Stage 3 keying."""
    return {
        "config": Path("table.yaml"),
        "marker": marker,
        "source": {"local": str(source)} if source is not None else {"url": ["https://example.org/data.tsv"]},
    }


def _key(section: dict[str, Any], tmp_path: Path, memo: dict[tuple[Path, int, int], str]) -> str:
    """Derive a key through the exact private helper used by the build loop."""
    return cli._section_store_key_for_build(section, tmp_path / "graph.yaml", memo)[0]


def test_content_change_gives_new_store_filename(tmp_path: Path) -> None:
    """Changing source bytes changes the Stage 3 parquet filename.

    WHY: a config-only key would let ``Tcode.collect`` quick-exit with a parquet made
    from stale source rows after the upstream file was edited.
    """
    source: Path = tmp_path / "data.tsv"
    source.write_bytes(b"v1")
    memo: dict[tuple[Path, int, int], str] = {}
    first: str = _key(_section(source), tmp_path, memo)
    source.write_bytes(b"v2")
    second: str = _key(_section(source), tmp_path, memo)
    assert first != second
    assert len(second) == 16
    assert (tmp_path / ".tablassert" / "store" / f"{second}.parquet").name == f"{second}.parquet"


def test_utime_only_touch_keeps_store_filename(tmp_path: Path) -> None:
    """Touching a source without changing bytes preserves its filename.

    WHY: stat metadata is only the per-build memo invalidation signature; it must not
    become part of the content-addressed store identity.
    """
    source: Path = tmp_path / "data.tsv"
    source.write_bytes(b"same bytes")
    memo: dict[tuple[Path, int, int], str] = {}
    first: str = _key(_section(source), tmp_path, memo)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert _key(_section(source), tmp_path, memo) == first


def test_stat_change_rehashes_shared_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed stat signature causes the per-build memo to read the file again.

    WHY: the memo is an intra-build optimization, not a correctness cache; a source
    replaced during a build must not keep using a digest associated with stale metadata.
    """
    source: Path = tmp_path / "data.tsv"
    source.write_bytes(b"same bytes")
    import tablassert.utils

    real_hash = tablassert.utils.file_content_hash
    calls: list[Path] = []

    def counted_hash(path: Path, **kwargs: Any) -> str:
        calls.append(path)
        return real_hash(path, **kwargs)

    monkeypatch.setattr(tablassert.utils, "file_content_hash", counted_hash)
    memo: dict[tuple[Path, int, int], str] = {}
    first: str = _key(_section(source), tmp_path, memo)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second: str = _key(_section(source), tmp_path, memo)
    assert first == second
    assert calls == [source.resolve(), source.resolve()]


def test_config_only_change_rekeys(tmp_path: Path) -> None:
    """Changing section configuration rekeys even when the source bytes are shared.

    WHY: content awareness augments, rather than replaces, the section configuration
    identity; two transformations over one file must never share a parquet.
    """
    source: Path = tmp_path / "data.tsv"
    source.write_bytes(b"same bytes")
    memo: dict[tuple[Path, int, int], str] = {}
    first: str = _key(_section(source, marker="one"), tmp_path, memo)
    second: str = _key(_section(source, marker="two"), tmp_path, memo)
    assert first != second


def test_shared_file_hashes_once_per_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct sections sharing a file invoke the content hash once per build.

    WHY: a graph can expand many sections over one downloaded source; Stage 3 should
    avoid repeatedly streaming the same large file while retaining per-build scope.
    """
    source: Path = tmp_path / "data.tsv"
    source.write_bytes(b"shared bytes")
    import tablassert.utils

    real_hash = tablassert.utils.file_content_hash
    calls: list[Path] = []

    def counted_hash(path: Path, **kwargs: Any) -> str:
        calls.append(path)
        return real_hash(path, **kwargs)

    monkeypatch.setattr(tablassert.utils, "file_content_hash", counted_hash)
    memo: dict[tuple[Path, int, int], str] = {}
    _key(_section(source, marker="one"), tmp_path, memo)
    _key(_section(source, marker="two"), tmp_path, memo)
    assert calls == [source.resolve()]


@pytest.mark.parametrize(("kind", "detail"), [("missing", "no such file"), ("directory", "not a regular file"), ("unreadable", "permission denied")])
def test_source_errors_name_config_section_and_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, detail: str) -> None:
    """Missing, directory, and hashing-read failures are coded and contextualized.

    WHY: Stage 3 is the first build stage that reads ``source.local`` for identity;
    users need the configuration, section label, and resolved offending path at once.
    """
    source: Path = tmp_path / f"{kind}.tsv"
    if kind == "directory":
        source.mkdir()
    elif kind == "unreadable":
        source.write_bytes(b"bytes")

        def fail_hash(path: Path, **kwargs: Any) -> str:
            raise SourceFileError(path.resolve(), detail, **kwargs)

        monkeypatch.setattr("tablassert.utils.file_content_hash", fail_hash)
    section: dict[str, Any] = _section(source)
    with pytest.raises(SourceFileError) as exc_info:
        _key(section, tmp_path, {})
    error = exc_info.value
    assert error.code == "source-file-unreadable"
    message = str(error)
    assert "graph.yaml" in message
    assert "table · " in message
    assert str(source.resolve()) in message


def test_no_local_uses_config_only_key(tmp_path: Path) -> None:
    """A URL-only section keeps the config-only key and does not read a data file.

    WHY: validation accepts URL-only/value sections without a local source, preserving
    the prior Tcode validation behavior and cache identity.
    """
    section = _section(None)
    assert _key(section, tmp_path, {}) == section_store_key(section)
