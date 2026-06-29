from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest


class FakeHeadResponse:
    status_code: int = 200

    def raise_for_status(self) -> None:
        return None


def fakehead(url: str, *args: Any, **kwargs: Any) -> FakeHeadResponse:
    return FakeHeadResponse()


@pytest.fixture(autouse=True)
def mockhttpxhead(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "head", fakehead)


@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def datassert_dir() -> Path:
    # ? Shared Datassert Shard Directory (Skipped When Unavailable)
    env: Optional[str] = os.environ.get("DATASSERT")
    if not env:
        pytest.skip("DATASSERT env var not set; skipping datassert-dependent test")
    directory: Path = Path(env)
    if not (directory / "data" / "0.duckdb").is_file():
        pytest.skip(f"datassert shard data/0.duckdb not found under {directory}")
    return directory
