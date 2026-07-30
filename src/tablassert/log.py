from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from tablassert.utils import BASE

if TYPE_CHECKING:
    from loguru import Logger

LOGASSERT: Path = BASE / "log"
LOGASSERT.mkdir(parents=True, exist_ok=True)

LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[category]} | {message}"

logger.configure(extra={"category": "PIPELINE"})
logger.remove()
# mode="a" + enqueue=True: under multiprocessing Pool() (spawn) each worker re-imports
# this module and reopens the log. mode="w" would truncate the parent's log mid-run, so
# append mode (O_APPEND) is what makes concurrent cross-process writes safe; enqueue=True
# additionally serializes writes through a per-process queue so threads within one process
# don't interleave partial lines.
logger.add((LOGASSERT / "tablassert.log"), level="INFO", format=LOG_FORMAT, rotation="100 MB", encoding="utf-8", mode="a", enqueue=True)


def cat(name: str) -> Logger:
    return logger.bind(category=name)
