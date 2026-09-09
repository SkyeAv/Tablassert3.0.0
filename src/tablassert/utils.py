from __future__ import annotations

from pathlib import Path
from typing import Any

from tablassert import rs
from tablassert.errors import SourceFileError

BASE: Path = Path("./.tablassert")
STORE: Path = BASE / "store"
STORE.mkdir(parents=True, exist_ok=True)


def mkhash(x: Any) -> str:
    # xxh64 (not xxh32): the digest is the content-addressed identity for section
    # parquet stores and the section label in validation errors. 32 bits invites
    # birthday collisions (~50% at ~77k sections) that would silently reuse another
    # section's cached subgraph; the full 16-hex 64-bit digest avoids that.
    return rs.xxh64(str(x))


def file_content_hash(path: Path, *, config: Path | None = None, section_label: str | None = None) -> str:
    """XXH64 hex digest of a source file's raw bytes, or a loud :class:`SourceFileError`.

    Pure: no caching, so the digest always reflects the file's bytes at call time and an
    edited source always yields a new section store key. ``config`` / ``section_label``
    are optional build context forwarded to the error message when available.
    """
    resolved: Path = path.resolve()
    if not resolved.exists():
        raise SourceFileError(resolved, "no such file", config=config, section_label=section_label)
    if not resolved.is_file():
        raise SourceFileError(resolved, "not a regular file", config=config, section_label=section_label)
    try:
        return rs.xxh64_file(str(resolved))
    except OSError as e:
        raise SourceFileError(resolved, str(e), config=config, section_label=section_label) from e


def section_store_key(section: Any, local: Path | None = None, *, content_digest: str | None = None) -> str:
    """Content-addressed identity for a section's cached parquet store.

    With a ``local`` source file or a precomputed ``content_digest`` the key mixes the
    section config hash with the file's content digest, so editing the source file
    invalidates the cache; without one the key is the plain section hash. Always a
    single 16-hex xxh64 digest. ``content_digest`` lets a caller memoize file reads
    without changing the key formula.
    """
    if content_digest is None:
        if local is None:
            return mkhash(section)
        content_digest = file_content_hash(local)
    return mkhash(f"{mkhash(section)}:{content_digest}")
