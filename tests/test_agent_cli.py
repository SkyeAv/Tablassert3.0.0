"""US-010: the ``tablassert agent`` cyclopts subcommand (wiring only, no live model).

Every test here runs in the BASE environment (no ``importorskip``): the command lazy-imports
``tablassert.agent`` (which never eagerly imports smolagents), the no-secret path fails BEFORE any
model is built, and the forwarding path monkeypatches ``run_supervisor``/``build_model`` so no real
agent or network call ever fires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tablassert.agent import ENV_API_BASE, ENV_API_KEY, ENV_MODEL_ID
from tablassert.cli import APP, agent


def test_agent_command_registered() -> None:
    """The ``agent`` subcommand is registered on the shared CLI ``APP``.

    Why: ``tablassert agent`` must be a discoverable peer of ``build-kg``/``validate``. Asserting
    against cyclopts' resolved-command mapping proves the ``@APP.command(name="agent")`` decorator
    actually attached it (a direct registration check, not a fragile ``--help`` scrape).
    """
    assert "agent" in APP.resolved_commands()


def test_agent_no_secret_fails_loud(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """With no model config anywhere, ``agent`` exits 2 and names the missing env var on stderr.

    Why: a secret is never defaulted or hardcoded. With the three ``TABLASSERT_AGENT_*`` vars unset
    and no flags given, the command must fail LOUD (``SystemExit(2)``) naming the model-id env var and
    the no-hardcode policy — and it must do so BEFORE building a model, so this path never touches
    smolagents (proving the secret check precedes any extra-triggering call).
    """
    monkeypatch.delenv(ENV_MODEL_ID, raising=False)
    monkeypatch.delenv(ENV_API_BASE, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(SystemExit) as exc_info:
        agent(["PMC1"], fullmap=Path("/tmp/fm"))
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert ENV_MODEL_ID in captured.err
    assert "secret" in captured.err.lower() or "hardcode" in captured.err.lower()


def test_agent_env_fallback_and_forwarding(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Model config resolves from env and every knob is forwarded to the supervisor verbatim.

    Why: the CLI is thin glue over ``run_supervisor``. With the three env vars set (and no flags),
    ``resolve_model_config`` must fill the model config from the environment, and the thresholds /
    executor / fetch flag / fullmap must reach the supervisor unchanged. ``run_supervisor`` and
    ``build_model`` are monkeypatched (module attributes the command looks up at call time) so no real
    agent runs; invoking the forwarded ``build_model_factory`` then proves the factory resolved the env
    config and handed it to ``build_model``.
    """
    monkeypatch.setenv(ENV_MODEL_ID, "env-model")
    monkeypatch.setenv(ENV_API_BASE, "env-base")
    monkeypatch.setenv(ENV_API_KEY, "env-key")

    captured: dict[str, object] = {}

    def fake_run_supervisor(pmc_ids: list[str], **kwargs: object) -> dict[str, object]:
        captured["pmc_ids"] = pmc_ids
        captured.update(kwargs)
        return {"records": {}, "metrics": {"mapped": 0, "skipped": 0}}

    build_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_build_model(*args: object, **kwargs: object) -> object:
        build_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("tablassert.agent.run_supervisor", fake_run_supervisor)
    monkeypatch.setattr("tablassert.agent.build_model", fake_build_model)

    agent(["PMC1", "PMC2"], fullmap=Path("/tmp/fm"), map_threshold=0.7, max_improve_iters=5, executor="docker", no_fetch=True)

    assert captured["pmc_ids"] == ["PMC1", "PMC2"]
    assert captured["fullmap"] == Path("/tmp/fm")
    assert captured["map_threshold"] == 0.7
    assert captured["max_improve_iters"] == 5
    assert captured["executor"] == "docker"
    assert captured["fetch"] is False  # no_fetch=True -> fetch=False

    # The forwarded factory resolves config from the environment and builds via the patched build_model.
    factory = captured["build_model_factory"]
    assert callable(factory)
    factory()
    assert build_calls, "build_model_factory() must invoke build_model"
    args, kwargs = build_calls[0]
    assert args == ("env-model", "env-base", "env-key")
    assert kwargs == {"backend": "openai"}
    # The summary line printed without error.
    assert "tablassert agent:" in capsys.readouterr().out


def test_agent_cli_flag_parsing() -> None:
    """A full argv parses into the command's bound args WITHOUT executing the body.

    Why: the documented UX is positional PMC ids plus flags. cyclopts' ``parse_args`` binds tokens to
    the signature without running the function, so this proves ``agent PMC9 --fullmap ... --executor
    docker`` parses (positional list + required ``--fullmap`` + typed flags) with no model/network run.
    """
    fn, bound, _ = APP.parse_args(["agent", "PMC9", "--fullmap", "/tmp/fm", "--map-threshold", "0.5", "--executor", "docker"], exit_on_error=False)
    assert fn is agent
    assert bound.args == (["PMC9"],)
    assert bound.kwargs["fullmap"] == Path("/tmp/fm")
    assert bound.kwargs["map_threshold"] == 0.5
    assert bound.kwargs["executor"] == "docker"
