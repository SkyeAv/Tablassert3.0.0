from __future__ import annotations

from tablassert import rs
from tablassert.utils import mkhash


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
