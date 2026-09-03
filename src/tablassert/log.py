"""Pipeline logging: loguru when the ``log`` extra is installed, stdlib-backed shim otherwise.

Every entry point imports this module eagerly, so a missing loguru must degrade, never
raise. The fallback below implements exactly the loguru subset tablassert uses
(``configure``/``bind``/``add``/``remove`` plus ``info``/``warning``/``error``/``debug``
with brace-style kwargs) and writes the same ``LOGASSERT / "tablassert.log"`` file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, TextIO

from tablassert.utils import BASE

if TYPE_CHECKING:
    from loguru import Logger

LOGASSERT: Path = BASE / "log"
LOGASSERT.mkdir(parents=True, exist_ok=True)

LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[category]} | {message}"

_LOG_FILE: Final[Path] = LOGASSERT / "tablassert.log"

_LEVELS: Final[dict[str, int]] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}


class _StdlibLogger:
    """Loguru-shaped fallback used only when the ``log`` extra is absent.

    Rotation and enqueue are loguru-only features and are dropped here; append mode is
    kept so concurrent multiprocessing workers still share the one log file safely.
    """

    _next_id: ClassVar[int] = 0
    _sinks: ClassVar[dict[int, tuple[int, Callable[[str], None]]]] = {}

    def __init__(self: _StdlibLogger, category: str) -> None:
        self._category: str = category

    def configure(self: _StdlibLogger, *, extra: dict[str, Any] | None = None, **_: Any) -> None:
        if extra and isinstance(extra.get("category"), str):
            self._category = extra["category"]

    def bind(self: _StdlibLogger, *, category: str, **_: Any) -> _StdlibLogger:
        return _StdlibLogger(category)

    def add(self: _StdlibLogger, sink: Any, *, level: str = "INFO", **_: Any) -> int:
        min_level: int = _LEVELS.get(str(level).upper(), 20)
        writer: Callable[[str], None]
        if callable(sink):

            def write_to_callable(line: str) -> None:
                sink(line)

            writer = write_to_callable
        else:
            handle: TextIO = Path(str(sink)).open("a", encoding="utf-8")  # noqa: SIM115 - the sink lives until remove() or process exit

            def write_to_file(line: str) -> None:
                handle.write(line + "\n")
                handle.flush()

            writer = write_to_file

        sink_id: int = _StdlibLogger._next_id
        _StdlibLogger._next_id += 1
        _StdlibLogger._sinks[sink_id] = (min_level, writer)
        return sink_id

    def remove(self: _StdlibLogger, sink_id: int | None = None) -> None:
        if sink_id is None:
            _StdlibLogger._sinks.clear()
        else:
            _StdlibLogger._sinks.pop(sink_id, None)

    def _emit(self: _StdlibLogger, level: str, message: str, fields: dict[str, Any]) -> None:
        if fields:
            # logging must never crash the pipeline
            with suppress(KeyError, IndexError, ValueError):
                message = message.format(**fields)
        line: str = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {level} | {self._category} | {message}"
        levelno: int = _LEVELS[level]
        for min_level, writer in list(_StdlibLogger._sinks.values()):
            if levelno >= min_level:
                writer(line)

    def info(self: _StdlibLogger, message: str, /, **fields: Any) -> None:
        self._emit("INFO", message, fields)

    def warning(self: _StdlibLogger, message: str, /, **fields: Any) -> None:
        self._emit("WARNING", message, fields)

    def error(self: _StdlibLogger, message: str, /, **fields: Any) -> None:
        self._emit("ERROR", message, fields)

    def debug(self: _StdlibLogger, message: str, /, **fields: Any) -> None:
        self._emit("DEBUG", message, fields)


try:
    from loguru import logger
except ImportError:
    logger = _StdlibLogger("PIPELINE")
    logger.add(_LOG_FILE, level="INFO")
    logging.getLogger("tablassert").warning(
        'loguru is not installed; using the stdlib logging fallback. For full logging (rotation, enqueue): pip install "tablassert[log]"'
    )
else:
    logger.configure(extra={"category": "PIPELINE"})
    logger.remove()
    # mode="a" + enqueue=True: under multiprocessing Pool() (spawn) each worker re-imports
    # this module and reopens the log. mode="w" would truncate the parent's log mid-run, so
    # append mode (O_APPEND) is what makes concurrent cross-process writes safe; enqueue=True
    # additionally serializes writes through a per-process queue so threads within one process
    # don't interleave partial lines.
    logger.add(_LOG_FILE, level="INFO", format=LOG_FORMAT, rotation="100 MB", encoding="utf-8", mode="a", enqueue=True)


def cat(name: str) -> Logger | _StdlibLogger:
    return logger.bind(category=name)
