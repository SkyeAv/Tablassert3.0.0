"""Minimal lazy-import proxy.

Replaces the third-party ``lazy-loader`` dependency with the small subset of
behaviour Tablassert actually relies on: defer a module's import cost (and, for
optional extras such as ``sentence_transformers``, avoid importing it at all on
code paths that never touch it) while still allowing attribute access such as
``pl.col`` / ``pl.DataFrame``.

Type annotations are unaffected because every module keeps
``from __future__ import annotations``, so annotations stay strings and never
trigger the proxy; the ``if TYPE_CHECKING`` imports give type checkers the real
module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


class LazyModule:
    """A module that is imported lazily on first attribute access.

    Args:
        name: Fully-qualified module name (e.g. ``"polars"``).
    """

    __slots__ = ("_module", "_name")

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def _load(self) -> Any:
        module = self._module
        if module is None:
            try:
                module = import_module(self._name)
            except ImportError as exc:
                # Deferred: keeps this module import-light (everything imports it) and the
                # registry lookup off the success path. A proxied optional dependency that
                # is absent must name its extra here -- this is the LAST point that knows
                # the module name, and the caller only sees "No module named 'x'".
                from tablassert.extras import actionable_import_error

                actionable = actionable_import_error(self._name)
                if actionable is None:
                    raise
                raise actionable from exc
            self._module = module
        return module

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._load(), attr)

    def __repr__(self) -> str:
        state = "loaded" if self._module is not None else "pending"
        return f"<LazyModule {self._name!r} ({state})>"
