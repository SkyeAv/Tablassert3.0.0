from __future__ import annotations

from pathlib import Path
from typing import Any

from tablassert import rs

BASE: Path = Path("./.tablassert")
STORE: Path = BASE / "store"
STORE.mkdir(parents=True, exist_ok=True)


def mkhash(x: Any) -> str:
    # xxh64 (not xxh32): the digest is the content-addressed identity for section
    # parquet stores and the section label in validation errors. 32 bits invites
    # birthday collisions (~50% at ~77k sections) that would silently reuse another
    # section's cached subgraph; the full 16-hex 64-bit digest avoids that.
    return rs.xxh64(str(x))
