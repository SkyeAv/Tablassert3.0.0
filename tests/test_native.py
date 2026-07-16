from __future__ import annotations


def test_native_version() -> None:
    from tablassert import _native

    assert isinstance(_native.version(), str)
