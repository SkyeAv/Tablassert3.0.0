from __future__ import annotations

from pathlib import Path
from typing import Any

from tablassert import rs

BASE: Path = Path("./.tablassert")
STORE: Path = BASE / "store"
STORE.mkdir(parents=True, exist_ok=True)


def mkhash(x: Any) -> str:
    return rs.xxh32(str(x))
