from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"
