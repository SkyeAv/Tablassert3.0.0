from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

import tablassert.agent as agent_mod


def test_agent_module_imports_without_extra() -> None:
    """``tablassert.agent`` imports cleanly in the base environment.

    Why: the ``[agent]`` extra is optional, so merely importing the module must
    never require ``smolagents``/``dspy``. Pinning the ``AGENT_EXTRA`` install
    hint guards the actionable error message users rely on when the extra is
    missing, and confirms the module surface is present without the extra.
    """
    assert agent_mod.AGENT_EXTRA == "pip install tablassert[agent]"
    assert "agent" in agent_mod.AGENT_EXTRA


def test_require_raises_actionable_when_extra_absent() -> None:
    """``_require`` raises an actionable ImportError naming the extra.

    Why: when the optional dependency is absent, users must get a loud error
    that tells them exactly which package and which extra to install — not a
    bare ``ModuleNotFoundError``. When the extra IS installed the failure path
    cannot be exercised, so the test skips that branch gracefully and still
    passes in both environments.
    """
    if importlib.util.find_spec("smolagents") is not None:
        # Extra installed: the actionable-error path is unreachable; confirm the
        # import simply succeeds instead of asserting on a raise that won't happen.
        agent_mod._require("smolagents")
        return

    with pytest.raises(ImportError, match=r"tablassert\[agent\].*smolagents|smolagents.*tablassert\[agent\]") as excinfo:
        agent_mod._require("smolagents")
    message: str = str(excinfo.value)
    assert "tablassert[agent]" in message
    assert "smolagents" in message


def test_lazy_proxy_does_not_eagerly_import() -> None:
    """Importing ``tablassert.agent`` never forces ``smolagents`` to load.

    Why: the whole point of the lazy proxy is that importing the module costs
    nothing and never pulls in the heavy optional stack. A fresh re-import must
    not register ``smolagents`` in ``sys.modules`` when the extra is absent, and
    the proxy must report a ``pending`` (not yet loaded) state — proving no
    top-level attribute access triggered an eager import.
    """
    # Re-import freshly to prove the import itself is side-effect free.
    importlib.reload(agent_mod)
    assert agent_mod.is_lazy()

    if importlib.util.find_spec("smolagents") is None:
        # Absent: a successful import cannot have registered it.
        assert "smolagents" not in sys.modules
    # In both environments the proxy stays pending until first attribute access.
    assert "pending" in repr(agent_mod.smolagents)
