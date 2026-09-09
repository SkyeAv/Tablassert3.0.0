from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from tablassert import rs
from tablassert.errors import SourceFileError
from tablassert.utils import file_content_hash, mkhash, section_store_key


def test_mkhash_deterministic() -> None:
    """mkhash is deterministic."""
    a: str = mkhash("hello")
    b: str = mkhash("hello")
    assert a == b


def test_mkhash_different_inputs() -> None:
    """mkhash produces different hashes for different inputs."""
    a: str = mkhash("hello")
    b: str = mkhash("world")
    assert a != b


def test_mkhash_returns_hex_string() -> None:
    """mkhash returns the full 16-hex xxh64 digest.

    WHY: the digest is the content-addressed identity for section parquet stores and
    the section label in validation errors. A 32-bit digest (8 hex) invites birthday
    collisions (~50% at ~77k sections) that would silently reuse another section's
    cached subgraph; the 64-bit digest (16 hex) makes that collision space negligible.
    """
    h: str = mkhash("test")
    assert isinstance(h, str)
    assert len(h) == 16  # xxh64 hexdigest is 16 chars


def test_mkhash_handles_int() -> None:
    """mkhash handles various types."""
    h: str = mkhash(42)
    assert isinstance(h, str)
    assert len(h) == 16  # xxh64 hexdigest is 16 chars


def test_mkhash_is_full_xxh64_of_stringified_input() -> None:
    """mkhash hashes the SAME stringified input and returns the FULL xxh64 digest.

    WHY: guards the migration from xxh32 to xxh64 — the digest must equal
    ``rs.xxh64(str(x))`` exactly (no truncation, no algorithm drift), since every
    section store filename and validation label derives from this value.
    """
    assert mkhash("hello") == rs.xxh64("hello") == "26c7827d889f6da3"
    assert mkhash(42) == rs.xxh64("42")


def test_file_content_hash_utf8_parity_with_rs_xxh64(tmp_path: Path) -> None:
    """A file containing 'hello' hashes to the pinned digest 26c7827d889f6da3.

    WHY: every section store key derives from this digest. If the streaming file hash
    ever drifted from ``rs.xxh64`` over the same UTF-8 bytes, each cached section
    store would be silently invalidated — or worse, collide with another section's.
    """
    source: Path = tmp_path / "hello.txt"
    source.write_bytes(b"hello")  # ASCII == its UTF-8 encoding; parity with rs.xxh64("hello")
    digest: str = file_content_hash(source)
    assert digest == rs.xxh64("hello") == "26c7827d889f6da3"
    assert digest == rs.xxh64_file(str(source.resolve()))


def test_file_content_hash_resolves_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Relative paths resolve against the cwd before hashing.

    WHY: table configs name ``source.local`` relative to wherever the build runs; the
    digest must be of the file the user meant, and errors must name the resolved path.
    """
    (tmp_path / "rel.csv").write_bytes(b"hello")
    monkeypatch.chdir(tmp_path)
    assert file_content_hash(Path("rel.csv")) == "26c7827d889f6da3"


def test_file_content_hash_reflects_current_bytes(tmp_path: Path) -> None:
    """Editing the file changes the digest on the very next call.

    WHY: the helper is deliberately pure (no caching) — a memoized digest would let a
    stale cache entry survive an edit to the source file.
    """
    source: Path = tmp_path / "data.csv"
    source.write_bytes(b"v1")
    first: str = file_content_hash(source)
    source.write_bytes(b"v2")
    assert file_content_hash(source) != first


def test_file_content_hash_missing_path_raises(tmp_path: Path) -> None:
    """A missing source file fails loudly, never with a sentinel digest.

    WHY: hashing a nonexistent path must name the offending path so a user can fix
    their ``source.local`` instead of debugging a wrong-cache-hit downstream.
    """
    with pytest.raises(SourceFileError) as excinfo:
        file_content_hash(tmp_path / "nope.csv")
    assert excinfo.value.code == "source-file-unreadable"
    assert "nope.csv" in str(excinfo.value)


def test_file_content_hash_directory_raises(tmp_path: Path) -> None:
    """A directory passed as a source file is rejected before any read.

    WHY: directories can be 'opened' on some platforms but hashing one is nonsense;
    the failure must say the path is not a regular file, not surface a raw OS error.
    """
    with pytest.raises(SourceFileError) as excinfo:
        file_content_hash(tmp_path)
    assert excinfo.value.code == "source-file-unreadable"
    assert "not a regular file" in str(excinfo.value)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses file permission bits")
def test_file_content_hash_unreadable_file_raises_and_chains(tmp_path: Path) -> None:
    """An unreadable file wraps the OS error, chained as the cause.

    WHY: permission failures must surface as SourceFileError with the underlying
    OSError preserved on ``__cause__`` so the traceback still shows the real errno.
    """
    source: Path = tmp_path / "locked.csv"
    source.write_bytes(b"secret")
    source.chmod(0)
    try:
        with pytest.raises(SourceFileError) as excinfo:
            file_content_hash(source)
    finally:
        source.chmod(0o644)
    assert excinfo.value.code == "source-file-unreadable"
    assert isinstance(excinfo.value.__cause__, OSError)


def test_section_store_key_without_local_is_plain_mkhash() -> None:
    """Without a local file the store key is exactly ``mkhash(section)``.

    WHY: back-compat — sections sourced purely by URL keep their existing cache
    identity, so adopting content-aware keys does not invalidate every URL-sourced
    cache on upgrade.
    """
    section: dict[str, Any] = {"config": "x", "source": {"url": ["https://example.org/a.csv"]}}
    key: str = section_store_key(section)
    assert key == mkhash(section)
    assert len(key) == 16
    int(key, 16)


def test_section_store_key_with_local_mixes_content_digest(tmp_path: Path) -> None:
    """With a local file the key is mkhash('<mkhash(section)>:<file digest>').

    WHY: the local file's bytes are part of the section's identity; the key shape is
    pinned so a drift in the mixing formula is caught as a cache-busting change, and
    the output stays one 16-hex digest like every other store key.
    """
    source: Path = tmp_path / "s.csv"
    source.write_bytes(b"hello")
    section: dict[str, Any] = {"config": "x", "source": {"local": str(source)}}
    expected: str = mkhash(f"{mkhash(section)}:{file_content_hash(source)}")
    key: str = section_store_key(section, source)
    assert key == expected
    assert len(key) == 16
    int(key, 16)


def test_section_store_key_tracks_file_edits(tmp_path: Path) -> None:
    """Editing the local source changes the store key; the config-only key differs too.

    WHY: this is the entire point of content-aware keys — a stale parquet must never
    survive an edit to the file it was built from.
    """
    source: Path = tmp_path / "s.csv"
    source.write_bytes(b"v1")
    section: dict[str, Any] = {"config": "x", "source": {"local": str(source)}}
    first: str = section_store_key(section, source)
    source.write_bytes(b"v2")
    assert section_store_key(section, source) != first
    assert section_store_key(section) != first
