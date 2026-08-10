"""US-010: the ``tablassert agent`` cyclopts subcommand (wiring only, no live model).

Every test here runs in the BASE environment (no ``importorskip``): the command lazy-imports
``tablassert.agent`` (which never eagerly imports smolagents), the no-secret path fails BEFORE any
model is built, and the forwarding path monkeypatches ``run_supervisor``/``build_model`` so no real
agent or network call ever fires.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml
from cyclopts.exceptions import MissingArgumentError  # pyright: ignore[reportMissingImports]

from tablassert.agent import ENV_API_BASE, ENV_API_KEY, ENV_MODEL_ID, load_optimized_instructions, save_optimized_instructions
from tablassert.cli import APP, agent, rebuild_agent_graph


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
    fullmap must reach the supervisor unchanged. ``run_supervisor`` and
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

    agent(["PMC1", "PMC2"], fullmap=Path("/tmp/fm"), map_threshold=0.7, max_improve_iters=5)

    assert captured["pmc_ids"] == ["PMC1", "PMC2"]
    assert captured["fullmap"] == Path("/tmp/fm")
    assert captured["map_threshold"] == 0.7
    assert captured["max_improve_iters"] == 5

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
    the signature without running the function, so this proves ``agent PMC9 --fullmap ... --map-threshold``
    parses (positional list + required ``--fullmap`` + typed flags) with no model/network run.
    """
    fn, bound, _ = APP.parse_args(["agent", "PMC9", "--fullmap", "/tmp/fm", "--map-threshold", "0.5"], exit_on_error=False)
    assert fn is agent
    assert bound.args == (["PMC9"],)
    assert bound.kwargs["fullmap"] == Path("/tmp/fm")
    assert bound.kwargs["map_threshold"] == 0.5


def test_agent_optimize_flag_parses() -> None:
    """``-o``/``--optimize`` parses to optimize=True without executing the body."""
    fn, bound, _ = APP.parse_args(["agent", "PMC9", "--fullmap", "/tmp/fm", "-o"], exit_on_error=False)
    assert fn is agent
    assert bound.kwargs["optimize"] is True


def test_agent_optimize_persists_instructions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """W6: ``--optimize`` runs GEPA (stubbed) and persists optimized instructions; the supervisor is NOT run."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    monkeypatch.setattr("tablassert.agent.make_dspy_lm", lambda *a, **k: object())

    def fake_run_gepa(**kwargs: object) -> dict[str, object]:
        assert kwargs.get("seed_instructions")  # the seed prompt is passed
        return {"optimized_instructions": "OPTIMIZED PROMPT", "optimized_descriptions": {"propose": "DESC"}, "stats": {}, "frontier": []}

    monkeypatch.setattr("tablassert.agent.run_gepa", fake_run_gepa)

    def fail_supervisor(*a: object, **k: object) -> object:
        raise AssertionError("run_supervisor must NOT run when --optimize is set")

    monkeypatch.setattr("tablassert.agent.run_supervisor", fail_supervisor)

    out: Path = tmp_path / "opt.yaml"
    agent(["PMC1"], fullmap=Path("/tmp/fm"), optimize=True, instructions_out=out)

    assert out.is_file()
    assert load_optimized_instructions(out) == "OPTIMIZED PROMPT"
    assert "optimized instructions" in capsys.readouterr().out


def test_agent_instructions_file_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """W6: ``--instructions-file`` loads optimized instructions and forwards them to the supervisor."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    captured: dict[str, object] = {}

    def fake_run_supervisor(pmc_ids: list[str], **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"records": {}, "metrics": {}}

    monkeypatch.setattr("tablassert.agent.run_supervisor", fake_run_supervisor)

    instr_file: Path = tmp_path / "instr.yaml"
    save_optimized_instructions(instr_file, "CUSTOM PROMPT")

    agent(["PMC1"], fullmap=Path("/tmp/fm"), instructions_file=instr_file)
    assert captured["instructions"] == "CUSTOM PROMPT"


def test_agent_no_instructions_file_passes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without ``--instructions-file`` the supervisor receives instructions=None (default INSTRUCTIONS)."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    captured: dict[str, object] = {}

    def fake_run_supervisor(pmc_ids: list[str], **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"records": {}, "metrics": {}}

    monkeypatch.setattr("tablassert.agent.run_supervisor", fake_run_supervisor)

    agent(["PMC1"], fullmap=Path("/tmp/fm"))
    assert captured["instructions"] is None


@pytest.mark.parametrize("bad_threshold", [-1.0, 2.0, float("nan"), float("inf")])
def test_agent_judge_threshold_out_of_range_exits_2(
    bad_threshold: float, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CodeRabbit: --judge-threshold outside [0, 1] (or non-finite) fails loud (exit 2) before any model runs."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    def fail_supervisor(*a: object, **k: object) -> object:
        raise AssertionError("run_supervisor must NOT run with an invalid --judge-threshold")

    monkeypatch.setattr("tablassert.agent.run_supervisor", fail_supervisor)

    with pytest.raises(SystemExit) as exc_info:
        agent(["PMC1"], fullmap=Path("/tmp/fm"), judge_threshold=bad_threshold)
    assert exc_info.value.code == 2
    assert "judge-threshold" in capsys.readouterr().err


def test_agent_judge_threshold_valid_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid --judge-threshold (here 0.7) passes validation and reaches the supervisor unchanged."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    captured: dict[str, object] = {}

    def fake_run_supervisor(pmc_ids: list[str], **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"records": {}, "metrics": {}}

    monkeypatch.setattr("tablassert.agent.run_supervisor", fake_run_supervisor)

    agent(["PMC1"], fullmap=Path("/tmp/fm"), judge_threshold=0.7)
    assert captured["judge_threshold"] == 0.7


@pytest.mark.parametrize("bad_threads", [0, -1, -8])
def test_agent_gepa_threads_non_positive_exits_2(bad_threads: int, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CodeRabbit: a non-positive --gepa-threads fails loud (exit 2) before any model is built."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    def fail_supervisor(*a: object, **k: object) -> object:
        raise AssertionError("run_supervisor must NOT run with an invalid --gepa-threads")

    def fail_model_init(*a: object, **k: object) -> object:
        raise AssertionError("make_dspy_lm must NOT run with an invalid --gepa-threads")

    monkeypatch.setattr("tablassert.agent.run_supervisor", fail_supervisor)
    # also prove NO model construction happens before validation (not just no supervisor run)
    monkeypatch.setattr("tablassert.agent.make_dspy_lm", fail_model_init)

    with pytest.raises(SystemExit) as exc_info:
        agent(["PMC1"], fullmap=Path("/tmp/fm"), gepa_threads=bad_threads)
    assert exc_info.value.code == 2
    assert "gepa-threads" in capsys.readouterr().err


@pytest.mark.parametrize("bad_spec", ["PMC1=", "=DIR", "PMC1=   "])
def test_agent_local_rejects_empty_mapping_components(bad_spec: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """CodeRabbit: --local PMCid=DIR with a blank PMC id or blank DIR fails loud (exit 2), not Path('.')."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    def fail_supervisor(*a: object, **k: object) -> object:
        raise AssertionError("run_supervisor must NOT run with an invalid --local mapping")

    monkeypatch.setattr("tablassert.agent.run_supervisor", fail_supervisor)

    with pytest.raises(SystemExit) as exc_info:
        agent(["PMC1"], fullmap=Path("/tmp/fm"), local=[bad_spec])
    assert exc_info.value.code == 2
    assert "--local" in capsys.readouterr().err


def test_agent_local_valid_mapping_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A valid --local PMCid=DIR mapping (existing dir) parses and reaches the supervisor as a dict."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    captured: dict[str, object] = {}

    def fake_run_supervisor(pmc_ids: list[str], **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"records": {}, "metrics": {}}

    monkeypatch.setattr("tablassert.agent.run_supervisor", fake_run_supervisor)

    agent(["PMC1"], fullmap=Path("/tmp/fm"), local=[f"PMC1={tmp_path}"])
    assert captured["local"] == {"PMC1": tmp_path}


def test_agent_optimize_forwards_backend_to_dspy_lm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CodeRabbit: --optimize forwards --backend to the GEPA reflection LM (not always openai)."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    lm_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_make_dspy_lm(*args: object, **kwargs: object) -> object:
        lm_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr("tablassert.agent.make_dspy_lm", fake_make_dspy_lm)
    monkeypatch.setattr(
        "tablassert.agent.run_gepa", lambda **k: {"optimized_instructions": "X", "optimized_descriptions": {}, "stats": {}, "frontier": []}
    )

    out: Path = tmp_path / "opt.yaml"
    agent(["PMC1"], fullmap=Path("/tmp/fm"), optimize=True, backend="litellm", instructions_out=out)

    # Without --task-model the CLI builds EXACTLY ONE LM (the reflection LM) — assert the count so a
    # regression that reorders/adds LM constructions cannot hide behind lm_calls[0].
    assert len(lm_calls) == 1
    args, kwargs = lm_calls[0]  # the reflection LM
    assert args == ("m", "b", "k")
    assert kwargs == {"backend": "litellm"}
    assert out.is_file()  # a successful compile still persists


def test_agent_optimize_gepa_error_exits_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CodeRabbit: a failed GEPA compile (stats['error']) is NOT reported as optimized; exit 1, nothing saved."""
    monkeypatch.setenv(ENV_MODEL_ID, "m")
    monkeypatch.setenv(ENV_API_BASE, "b")
    monkeypatch.setenv(ENV_API_KEY, "k")

    monkeypatch.setattr("tablassert.agent.make_dspy_lm", lambda *a, **k: object())
    monkeypatch.setattr(
        "tablassert.agent.run_gepa",
        lambda **k: {"optimized_instructions": "SEED", "optimized_descriptions": {}, "stats": {"error": "boom"}, "frontier": []},
    )

    out: Path = tmp_path / "opt.yaml"
    with pytest.raises(SystemExit) as exc_info:
        agent(["PMC1"], fullmap=Path("/tmp/fm"), optimize=True, instructions_out=out)
    assert exc_info.value.code == 1
    assert not out.is_file()  # the unoptimized seed is NOT persisted
    assert "GEPA optimization failed" in capsys.readouterr().err


def test_make_dspy_lm_honors_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """CodeRabbit: make_dspy_lm maps backend -> litellm model string (openai/ prefix vs pass-through)."""
    import tablassert.agent as agent_mod

    captured: list[dict[str, object]] = []

    class _FakeLM:
        def __init__(
            self,
            model: str,
            api_base: object = None,
            api_key: object = None,
            temperature: object = None,
            max_tokens: object = None,
            timeout: object = None,
        ) -> None:
            captured.append(
                {"model": model, "api_base": api_base, "api_key": api_key, "temperature": temperature, "max_tokens": max_tokens, "timeout": timeout}
            )

    monkeypatch.setitem(sys.modules, "dspy", types.SimpleNamespace(LM=_FakeLM))

    agent_mod.make_dspy_lm("gpt-x", "base", "key")  # default backend=openai
    agent_mod.make_dspy_lm("anthropic/claude", "base", "key", backend="litellm")

    assert captured[0]["model"] == "openai/gpt-x"
    assert captured[1]["model"] == "anthropic/claude"
    # reasoning-model-safe defaults are passed through (a truncated config_yaml would stall GEPA)
    assert captured[0]["temperature"] == 1.0
    assert captured[0]["max_tokens"] == 16000
    # default timeout bounds each request so a stalled connection cannot hang the optimizer
    assert captured[0]["timeout"] == 600


# --------------------------------------------------------------------------- #
# rebuild-agent-graph: the shared-registry reconstruction command (wiring only)
# --------------------------------------------------------------------------- #


def test_rebuild_agent_graph_command_registered() -> None:
    """The ``rebuild-agent-graph`` subcommand is registered as a flat peer of ``build-kg``."""
    assert "rebuild-agent-graph" in APP.resolved_commands()


def test_rebuild_agent_graph_flags_parse(tmp_path: Path) -> None:
    """``--state-dir``/``-sd`` + required ``--fullmap``/``-f`` bind; the state-dir default is pinned."""

    def parse(argv: list[str]) -> dict[str, object]:
        fn, bound, _ = APP.parse_args(argv, exit_on_error=False)
        assert fn is rebuild_agent_graph
        bound.apply_defaults()  # bound.arguments only carries explicitly-parsed tokens
        return dict(bound.arguments)

    arguments = parse(["rebuild-agent-graph", "--state-dir", str(tmp_path), "--fullmap", "/tmp/fm.redb"])
    assert arguments["state_dir"] == tmp_path
    assert arguments["fullmap"] == Path("/tmp/fm.redb")

    alias_arguments = parse(["rebuild-agent-graph", "-sd", str(tmp_path), "-f", "/tmp/fm.redb"])
    assert alias_arguments["state_dir"] == tmp_path
    assert alias_arguments["fullmap"] == Path("/tmp/fm.redb")

    default_arguments = parse(["rebuild-agent-graph", "-f", "/tmp/fm.redb"])
    assert default_arguments["state_dir"] == Path(".tablassert") / "agent"

    with pytest.raises(MissingArgumentError):
        APP.parse_args(["rebuild-agent-graph", "--state-dir", str(tmp_path)], exit_on_error=False)


def test_rebuild_agent_graph_rebuilds_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command rebuilds the registry from ``state.json`` and prints the path + entry count."""
    from tablassert.agent import ConfigRecord, SupervisorState, save_state

    config: Path = tmp_path / "configs" / "PMC1.yaml"
    config.parent.mkdir()
    config.write_text("sections: []\n")
    save_state(
        tmp_path, SupervisorState(pmc_ids=["PMC1"], records={"PMC1": ConfigRecord(pmc_id="PMC1", status="MAPPED", best_config_path=str(config))})
    )
    fullmap: Path = tmp_path / "fullmap.redb"
    fullmap.touch()

    rebuild_agent_graph(state_dir=tmp_path, fullmap=fullmap)

    out: str = capsys.readouterr().out
    assert str(tmp_path / "graph.yaml") in out
    assert "1 table config" in out
    data: object = yaml.safe_load((tmp_path / "graph.yaml").read_text())
    assert isinstance(data, dict)
    assert data["tables"] == [str(config.resolve())]
