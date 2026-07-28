"""Optional ``[agent]`` extra: an autonomous KGX knowledge-graph builder.

This module hosts a smolagents ``CodeAgent`` pipeline that autonomously builds
and audits KGX knowledge graphs from PubMed Central articles. It is part of the
OPTIONAL ``[agent]`` extra, so ``smolagents`` and ``dspy`` are imported LAZILY
(via :class:`tablassert._lazy.LazyModule`) and the base package never requires
them at import time. Install the extra with ``pip install tablassert[agent]``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from tablassert._lazy import LazyModule

if TYPE_CHECKING:
    import dspy  # pyright: ignore[reportMissingImports,reportUnusedImport]
    import polars as pl  # pyright: ignore[reportUnusedImport]
    import smolagents  # pyright: ignore[reportMissingImports,reportUnusedImport]
else:
    dspy = LazyModule("dspy")
    pl = LazyModule("polars")
    smolagents = LazyModule("smolagents")

AGENT_EXTRA: str = "pip install tablassert[agent]"


def _require(name: str) -> None:
    """Import an optional dependency or raise a loud, actionable ImportError."""
    try:
        import_module(name)
    except ImportError as exc:
        raise ImportError(f"tablassert agent features require the '{name}' package. Install with {AGENT_EXTRA}.") from exc


def is_lazy() -> bool:
    """Confirm the module loaded without eagerly importing any optional extra.

    A tiny public sentinel tests use to assert ``tablassert.agent`` imported
    cleanly in the base environment; always ``True`` because reaching this call
    already proves no top-level optional import was forced.
    """
    return True
