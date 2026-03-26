from __future__ import annotations

from pathlib import Path

from loguru import logger

LOGASSERT: Path = Path("./.logassert")
LOGASSERT.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    (LOGASSERT / "logassert.log"),
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    rotation="100 MB",
    encoding="utf-8",
    mode="w",
)
