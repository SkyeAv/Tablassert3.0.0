"""Tests for US-008 model builders + INSTRUCTIONS + step_callback + build_agent + FakeModel.

The no-secret / env / INSTRUCTIONS / step-callback tests are PURE and run in the base
environment (no ``[agent]`` extra): ``build_model`` performs its secret check BEFORE the
lazy ``smolagents`` import, and ``make_step_callback`` is pure Python over duck-typed
objects. The smolagents-dependent tests (model construction, agent wiring, offline run)
call ``pytest.importorskip("smolagents")`` so they skip cleanly when the extra is absent.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tablassert.agent import (
    ENV_API_BASE,
    ENV_API_KEY,
    ENV_MODEL_ID,
    INSTRUCTIONS,
    build_agent,
    build_model,
    make_fake_model,
    make_step_callback,
    resolve_model_config,
    validate_section,
    validate_table_config,
)


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``agent.run`` hermetically offline by disabling HuggingFace telemetry.

    Why: a real ``CodeAgent.run`` fires huggingface_hub telemetry that BLOCKS on a network
    call (observed: the run hangs indefinitely at zero CPU when telemetry is enabled). The
    standard opt-out vars make every ``agent.run`` in this module deterministic and
    network-free. Scoped to agent tests (NOT global conftest) so the QC suite's own
    HuggingFace model loading is unaffected.
    """
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# PURE tests (base env; no importorskip)
# --------------------------------------------------------------------------- #


def test_build_model_fails_loud_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing model_id/api_base/api_key raises RuntimeError naming its env var (no secret).

    The check precedes the lazy smolagents import, so this passes even without the extra.
    """
    monkeypatch.delenv(ENV_MODEL_ID, raising=False)
    monkeypatch.delenv(ENV_API_BASE, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)

    with pytest.raises(RuntimeError, match=ENV_MODEL_ID) as excinfo:
        build_model(None, None, None)
    message: str = str(excinfo.value).lower()
    assert "hardcode" in message or "secret" in message

    with pytest.raises(RuntimeError, match=ENV_API_BASE):
        build_model("m", None, None)
    with pytest.raises(RuntimeError, match=ENV_API_KEY):
        build_model("m", "http://x/v1", None)


def test_resolve_model_config_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """None args are filled from the TABLASSERT_AGENT_* env vars; explicit args win."""
    monkeypatch.setenv(ENV_MODEL_ID, "env-model")
    monkeypatch.setenv(ENV_API_BASE, "env-base")
    monkeypatch.setenv(ENV_API_KEY, "env-key")

    assert resolve_model_config(None, None, None) == ("env-model", "env-base", "env-key")
    assert resolve_model_config("a", "b", "c") == ("a", "b", "c")
    assert resolve_model_config("a", None, None) == ("a", "env-base", "env-key")


def test_instructions_content() -> None:
    """INSTRUCTIONS carries the data-fence markers, both exemplars, ReAct/plan, and final_answer."""
    assert "<<<PMC_DATA_BEGIN>>>" in INSTRUCTIONS
    assert "<<<PMC_DATA_END>>>" in INSTRUCTIONS
    assert "tutorial-table" in INSTRUCTIONS or "tutorial" in INSTRUCTIONS
    assert "ALAMV6" in INSTRUCTIONS
    assert "ReAct" in INSTRUCTIONS or "plan" in INSTRUCTIONS
    assert "final_answer" in INSTRUCTIONS


def test_step_callback_tallies() -> None:
    """make_step_callback tallies steps/tokens/tool-call quality over duck-typed steps (pure)."""
    metrics: dict[str, object] = {}
    cb = make_step_callback(metrics)
    agent = SimpleNamespace(memory=None)

    step = SimpleNamespace(
        error=None,
        tool_calls=[SimpleNamespace(name="build_and_audit", arguments={"config_yaml": "x"})],
        observations="ok",
        token_usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
        is_final_answer=False,
    )
    cb(step, agent)
    assert metrics["steps"] == 1
    assert metrics["total_tool_calls"] == 1
    assert metrics["input_tokens"] == 10
    assert metrics["output_tokens"] == 5
    assert metrics["total_tokens"] == 15

    # Same tool-call signature again -> redundant.
    cb(step, agent)
    assert metrics["redundant_tool_calls"] == 1

    # Observations that look like an error/traceback -> wrong.
    bad = SimpleNamespace(error=None, tool_calls=[], observations="Traceback: Error", token_usage=None, is_final_answer=False)
    cb(bad, agent)
    assert metrics["wrong_tool_calls"] >= 1  # type: ignore[operator]

    # step.error set -> failed.
    failed = SimpleNamespace(error="boom", tool_calls=[], observations=None, token_usage=None, is_final_answer=False)
    cb(failed, agent)
    assert metrics["failed_tool_calls"] == 1


# --------------------------------------------------------------------------- #
# smolagents-dependent tests (require the [agent] extra; skip cleanly when absent)
# --------------------------------------------------------------------------- #


def test_build_model_constructs_offline() -> None:
    """build_model constructs OpenAI/LiteLLM models offline (no network at construction)."""
    pytest.importorskip("smolagents")
    model = build_model("test-model", "http://localhost:9/v1", "sk-test")
    assert "OpenAI" in type(model).__name__
    lite = build_model("test-model", "http://localhost:9/v1", "sk-test", backend="litellm")
    assert "LiteLLM" in type(lite).__name__


def test_build_agent_wires_checks_and_callback() -> None:
    """build_agent wires validate_table_config into final_answer_checks and a default step callback."""
    pytest.importorskip("smolagents")
    agent = build_agent(model=make_fake_model(), tools=[])
    assert agent is not None
    checks: Any = getattr(agent, "final_answer_checks", [])
    assert validate_table_config in checks
    assert getattr(agent, "step_callbacks", None) is not None


def test_fake_model_drives_agent_run_offline() -> None:
    """A FakeModel-driven agent.run reaches a final answer offline (no network, terminates)."""
    pytest.importorskip("smolagents")
    from smolagents import LogLevel  # pyright: ignore[reportMissingImports]

    agent: Any = build_agent(model=make_fake_model(), tools=[], max_steps=3, verbosity_level=LogLevel.ERROR)
    result = agent.run("Derive a Tablassert config.")
    # smolagents 1.26.0 returns an AgentText (a str subclass) holding the final answer.
    assert result is not None
    assert str(result).strip()
    assert validate_section(str(result)) is True


def test_instructions_carry_a_generated_predicate_cheatsheet() -> None:
    """The prompt teaches predicate<->class legality, and does so from the INSTALLED model.

    The ~30 KB Section schema the derive_config tool injects lists all predicates and all categories
    as flat enums with nothing tying the two together, which is how the agent came to recommend
    `gene_associated_with_condition` for gene~disease -- a predicate GeneToDiseaseAssociation forbids.
    """
    from tablassert.agent import predicate_cheatsheet

    # Fully rendered: no template placeholder survives into the live prompt.
    assert "{{PREDICATE_CHEATSHEET}}" not in INSTRUCTIONS
    assert predicate_cheatsheet() in INSTRUCTIONS

    # The flagship pair, generated from biolink-model rather than hand-written.
    assert "Gene ~ Disease -> GeneToDiseaseAssociation: affects, associated_with, contributes_to" in INSTRUCTIONS
    assert "demoted_edge_pct" in INSTRUCTIONS

    # And the relocation/disabled-field rules the pipeline enforces.
    assert "supporting_study_size" in INSTRUCTIONS  # named as a slot that does NOT reach the edge
    assert "adjusted_p_value" in INSTRUCTIONS  # the recommended alternative
    assert "species_context_qualifier` is" in INSTRUCTIONS
    assert "species_context_qualifier` is\n  auto-derived" not in INSTRUCTIONS


def test_instructions_do_not_recommend_a_class_forbidden_predicate() -> None:
    """Every predicate the prompt shows in an exemplar must be legal for that exemplar's pair."""
    import re

    from tablassert.lib import predicate_options

    # Exemplar (a) is gene~disease; whatever predicate it demonstrates must keep the class.
    exemplar: str = INSTRUCTIONS[INSTRUCTIONS.index("# (a) tutorial-table") : INSTRUCTIONS.index("# (b) ALAMV6")]
    match = re.search(r"predicate:\s*(\w+)", exemplar)
    assert match is not None
    legal = predicate_options("Gene", "Disease")
    assert legal is not None
    assert f"biolink:{match.group(1)}" in legal
