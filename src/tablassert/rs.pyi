from __future__ import annotations

from pathlib import Path
from typing import Optional

def dedup_ndjson(input: Path, output: Path, is_edges: bool, domain: Optional[str] = None) -> None: ...
def namespace_uuid(domain: str, values: list[str]) -> str: ...
