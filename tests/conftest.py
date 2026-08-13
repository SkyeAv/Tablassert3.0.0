from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def fixtures_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def rig_factory() -> Any:
    """Build minimal VALID `rig:` dicts for graph configs in tests.

    Every field required by the RIG contract is present; callers override or
    extend via keyword arguments (``source_info={...}`` replaces the whole
    section, ``infores_id="..."` patches one value). ``artifact_base_path``
    defaults to the calling test's ``tmp_path`` when that fixture is in scope
    via the ``path`` override -- builds write artifacts there and the RIG
    audit cross-checks the directory.
    """

    def _make(path: Path | str = ".", infores_id: str = "infores:test-kg", **overrides: Any) -> dict[str, Any]:
        rig: dict[str, Any] = {
            "source_info": {
                "infores_id": infores_id,
                "terms_of_use_info": {"license_name": "CC0 1.0 Universal"},
                "data_access_locations": ["Test source - https://example.org/data"],
                "source_status": "unknown",
            },
            "ingest_info": {"utility": "Test utility.", "scope": "Test scope."},
            "provenance_info": {"contributions": ["Test author - code author"]},
            "artifact_base_url": f"https://example.org/{str(infores_id).removeprefix('infores:')}",
            "artifact_base_path": str(path),
        }
        for key, value in overrides.items():
            if key in ("source_info", "ingest_info", "provenance_info") and isinstance(value, dict) and key in rig:
                rig[key].update(value)
            else:
                rig[key] = value
        return rig

    return _make
