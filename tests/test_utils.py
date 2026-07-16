from __future__ import annotations

from uuid import UUID

from tablassert.utils import basespace, mkhash, namespace_uuid


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


# ? basespace Returns UUID
def test_basespace_returns_uuid() -> None:
    result: UUID = basespace("test_domain")
    assert isinstance(result, UUID)


# ? basespace Is Cached
def test_basespace_is_cached() -> None:
    a: UUID = basespace("cache_test")
    b: UUID = basespace("cache_test")
    assert a == b


# ? basespace Different Domains Produce Different UUIDs
def test_basespace_different_domains() -> None:
    a: UUID = basespace("domain_a")
    b: UUID = basespace("domain_b")
    assert a != b


# ? namespace_uuid Returns String
def test_namespace_uuid_returns_string() -> None:
    result: str = namespace_uuid("domain", "val1", "val2")  # pyright: ignore
    assert isinstance(result, str)


# ? namespace_uuid Is Deterministic
def test_namespace_uuid_deterministic() -> None:
    a: str = namespace_uuid("domain", "val1", "val2")  # pyright: ignore
    b: str = namespace_uuid("domain", "val1", "val2")  # pyright: ignore
    assert a == b


# ? namespace_uuid Different Values Produce Different UUIDs
def test_namespace_uuid_different_values() -> None:
    a: str = namespace_uuid("domain", "val1")  # pyright: ignore
    b: str = namespace_uuid("domain", "val2")  # pyright: ignore
    assert a != b


# ? namespace_uuid With No Extra Values
def test_namespace_uuid_no_values() -> None:
    result: str = namespace_uuid("domain")
    assert isinstance(result, str)
    assert len(result) == 36  # ? Standard UUID string length


# ? graph_uuid Returns A Stable UUID Shaped String For A JSON Row
def test_graph_uuid_returns_stable_uuid() -> None:
    from tablassert import tablassert_rs

    row: str = '{"subject":"A","object":"B","predicate":"r"}'
    a: str = tablassert_rs.graph_uuid(None, row)
    b: str = tablassert_rs.graph_uuid(None, row)
    assert a == b
    assert isinstance(a, str)
    assert len(a) == 36  # ? Standard UUID string length
