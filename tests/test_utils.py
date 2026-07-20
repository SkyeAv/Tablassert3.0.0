from __future__ import annotations

from tablassert.utils import mkhash


# ? mkhash Is Deterministic
def test_mkhash_deterministic() -> None:
    a: str = mkhash("hello")
    b: str = mkhash("hello")
    assert a == b


# ? mkhash Produces Different Hashes For Different Inputs
def test_mkhash_different_inputs() -> None:
    a: str = mkhash("hello")
    b: str = mkhash("world")
    assert a != b


# ? mkhash Returns Hex String
def test_mkhash_returns_hex_string() -> None:
    h: str = mkhash("test")
    assert isinstance(h, str)
    assert len(h) == 8  # ? xxh32 hexdigest is 8 chars


# ? mkhash Handles Various Types
def test_mkhash_handles_int() -> None:
    h: str = mkhash(42)
    assert isinstance(h, str)
    assert len(h) == 8
