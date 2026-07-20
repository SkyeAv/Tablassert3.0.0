from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lazy_loader as Lazy

if TYPE_CHECKING:
    import xxhash
else:
    xxhash = Lazy.load("xxhash")

STORE: Path = Path("./.storassert")
STORE.mkdir(parents=True, exist_ok=True)


def mkhash(x: Any) -> str:
    b: bytes = str(x).encode("utf-8")
    return xxhash.xxh32(b).hexdigest()
