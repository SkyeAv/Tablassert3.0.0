"""tablassert: declarative knowledge-graph extraction from tabular data.

The compiled Rust extension (``tablassert.rs``) provides the performance-critical
primitives — ``fullmap``, ``dedup_ndjson``, ``xxh32``, ``namespace_uuid`` — and is
required by every entry point.  It is built by ``maturin`` (PyO3) and shipped inside
the wheel as ``tablassert/rs.<abi>.so``.

We import it eagerly here so that a wheel installed *without* the extension fails
loudly and with a self-explanatory message at the very first ``import tablassert``
— rather than as a cryptic ``cannot import name 'rs'`` deep in the import chain.
The usual cause of a missing extension is a stale ``uv`` wheel cache left over from
the pure-Python 7.x era (``tablassert-*-py3-none-any.whl``); see the message below.
"""

from __future__ import annotations

from importlib import import_module

try:
    import_module("tablassert.rs")
except ImportError as exc:  # pragma: no cover - only on broken/extensionless installs
    raise ImportError(
        "tablassert's compiled Rust extension (tablassert.rs) is missing — the "
        "installed wheel was built without it. This is almost always a stale `uv` "
        "wheel cache (from the pure-Python 7.x era) being served instead of a fresh "
        "maturin build. Fix with a Rust toolchain (cargo) available:\n"
        "    uv cache clean tablassert\n"
        "    uv tool install --force --reinstall <tablassert source>\n"
        "then verify with:  python -c \"from tablassert import rs; print(rs.xxh32('x'))\""
    ) from exc
