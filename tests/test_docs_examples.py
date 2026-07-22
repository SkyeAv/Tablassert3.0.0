from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pytest

from tablassert.ingests import from_yaml, to_sections
from tablassert.lib import Tcode
from tablassert.models import Graph

# Repository root (parent of tests/).
ROOT: Path = Path(__file__).resolve().parent.parent
DOCS: Path = ROOT / "docs"
EXAMPLES: Path = DOCS / "examples"

# Config keys and runtime backends removed in 8.0.0. Their presence anywhere in
# the documentation signals doc drift and fails CI. Patterns are matched
# case-insensitively; ``contributors`` is anchored to a YAML key at line start so
# the unrelated RIG ``contributions`` field does not trip it.
BANNED_PATTERNS: list[tuple[str, str]] = [
    (r"syntax:\s*(TC|GC)\d", "the retired `syntax` config key (TC*/GC*)"),
    (r"status:\s*(alpha|beta|primetime)", "the retired `status` config key"),
    (r"^\s*contributors:", "the retired `provenance.contributors` config key"),
    (r"qc-cuda", "the removed `qc-cuda` extra"),
    (r"onnxruntime", "the removed ONNX Runtime QC backend"),
    (r"\.tablassert/onnx", "the removed `.tablassert/onnx` working directory"),
]


def _example_yaml_files() -> list[Path]:
    """Return every shipped example configuration under ``docs/examples``.

    Returns:
        Sorted list of example YAML paths.
    """
    return sorted(EXAMPLES.glob("*.yaml"))


def _doc_text_files() -> list[Path]:
    """Return every Markdown page and example YAML under ``docs/``.

    Returns:
        Sorted list of documentation text paths.
    """
    return sorted([*DOCS.rglob("*.md"), *EXAMPLES.glob("*.yaml")])


def test_example_yaml_files_exist() -> None:
    """The docs ship at least one example configuration."""
    assert _example_yaml_files(), f"no example YAML files found under {EXAMPLES}"


@pytest.mark.parametrize("path", _example_yaml_files(), ids=lambda p: p.name)
def test_example_yaml_validates(path: Path) -> None:
    """Each shipped example configuration validates against the live schema.

    Args:
        path: Example YAML file to validate.
    """
    raw: Any = from_yaml(path)
    assert isinstance(raw, dict), f"{path.name} did not parse to a mapping"

    if "name" in raw and "tables" in raw:
        # Graph configuration.
        Graph.model_validate(raw)
        return

    # Table configuration: expand template/sections and validate each section the
    # same way `tablassert validate-table` does (via Tcode).
    sections: list[dict[str, Any]] = to_sections(raw, path)  # pyright: ignore
    assert sections, f"{path.name} produced no sections"
    store: Path = ROOT / ".tablassert" / "store" / "docs-example.parquet"
    for section in sections:
        Tcode.model_validate({**section, "store": store})


@pytest.mark.parametrize("path", _doc_text_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_docs_have_no_removed_8_0_0_tokens(path: Path) -> None:
    """Documentation does not reintroduce keys or backends removed in 8.0.0.

    Args:
        path: Documentation text file to scan.
    """
    text: str = path.read_text(encoding="utf-8")
    for pattern, description in BANNED_PATTERNS:
        match: Optional[re.Match[str]] = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            raise AssertionError(f"{path.relative_to(ROOT)} references {description}: {match.group(0)!r}")
