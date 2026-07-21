from __future__ import annotations

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
    """mkhash returns hex string."""
    h: str = mkhash("test")
    assert isinstance(h, str)
    assert len(h) == 8  # xxh32 hexdigest is 8 chars


def test_mkhash_handles_int() -> None:
    """mkhash handles various types."""
    h: str = mkhash(42)
    assert isinstance(h, str)
    assert len(h) == 8
