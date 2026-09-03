"""Tests for the optional ``[log]`` extra: the packaging contract and the stdlib fallback.

loguru is an optional extra, so ``tablassert.log`` must import and log without it. The
fallback is exercised by reloading the module with ``sys.modules["loguru"]`` poisoned;
the fixture reloads once more on teardown so later tests in this worker see real loguru.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

import tablassert.log as tablassert_log

PYPROJECT: Path = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_log_extra_declared_and_loguru_not_core() -> None:
    with PYPROJECT.open("rb") as f:
        project = tomllib.load(f)["project"]
    extra = project["optional-dependencies"]["log"]
    assert any(dep.startswith("loguru") for dep in extra)
    assert not any(dep.startswith("loguru") for dep in project["dependencies"])


@pytest.fixture
def blocked_loguru(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setitem(sys.modules, "loguru", None)
    yield importlib.reload(tablassert_log)
    monkeypatch.undo()
    importlib.reload(tablassert_log)


def test_fallback_logs_without_loguru(blocked_loguru: ModuleType) -> None:
    logger = blocked_loguru.cat("TEST")
    logger.info("f-string style message with no fields")
    logger.info("brace style {name}={value}", name="x", value=1)
    logger.warning("warn {x}", x=2)
    logger.error("err")
    logger.debug("dbg")


def test_fallback_file_sink_add_remove(blocked_loguru: ModuleType, tmp_path: Path) -> None:
    sink_file: Path = tmp_path / "sink.log"
    sink_id: int = blocked_loguru.cat("TEST").add(sink_file, level="INFO")
    assert isinstance(sink_id, int)
    blocked_loguru.cat("TEST").info("written {what}", what="line")
    blocked_loguru.cat("TEST").remove(sink_id)
    contents: str = sink_file.read_text(encoding="utf-8")
    assert "written line" in contents
    assert "| INFO | TEST |" in contents


def test_fallback_callable_sink(blocked_loguru: ModuleType) -> None:
    lines: list[str] = []
    logger = blocked_loguru.cat("TEST")
    sink_id: int = logger.add(lines.append, level="INFO")
    logger.info("to callable")
    logger.remove(sink_id)
    logger.info("after remove")
    assert any("to callable" in line for line in lines)
    assert not any("after remove" in line for line in lines)


def test_fallback_survives_unformattable_fields(blocked_loguru: ModuleType) -> None:
    blocked_loguru.cat("TEST").info("mentions {missing} placeholder", other=1)
