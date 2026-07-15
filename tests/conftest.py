from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class FakeHeadResponse:
    status_code: int = 200

    def raise_for_status(self) -> None:
        return None


def fakehead(url: str, *args: Any, **kwargs: Any) -> FakeHeadResponse:
    return FakeHeadResponse()

@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"
