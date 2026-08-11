"""Coverage tests for the PURE/offline eval-harness heuristics in ``tablassert.agent``.

Targets the specific previously-uncovered branches in ``src/tablassert/agent.py``:
``_debias_verbosity`` (non-positive baseline), the ``except`` fallbacks of
``_judge_predicate_category`` / ``_judge_provenance`` / ``judge_config``, the ``bad == 1``
branch of ``_judge_cleanliness``, the non-callable branches of ``_call_judge``, the Reflexion
fullmap-regression + abort paths, ``_default_gepa_program.forward``, and the ``run_gepa``
dataset->Example path. The heuristics are pure (no dspy); the two dspy-dependent tests use
``pytest.importorskip("dspy")`` and an injectable ``gepa_cls`` stub so they stay offline.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tablassert.agent import (
    JUDGE_DIMENSIONS,
    _call_judge,
    _debias_verbosity,
    _default_gepa_program,
    _judge_cleanliness,
    _judge_predicate_category,
    _judge_provenance,
    judge_config,
    reflexion_improve,
    run_gepa,
)

# A genuinely valid minimal Section config (schema-valid wherever a valid YAML string is needed).
VALID_CFG: str = yaml.safe_dump(
    {
        "source": {"kind": "text", "local": "./t.tsv", "url": ["https://e.com/t.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "value", "encoding": "BRCA1"},
            "predicate": "associated_with",
            "object": {"method": "value", "encoding": "TP53"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC0000000"},
    }
)


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep any dspy run hermetically offline (HF telemetry blocks on network)."""
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# Verbosity debiasing: non-positive baseline guard (line 2329)
# --------------------------------------------------------------------------- #


def test_debias_verbosity_nonpositive_baseline_clamps() -> None:
    """Line 2329: ``baseline_len <= 0`` short-circuits to a clamped score (no ratio division).

    A zero or negative baseline would make ``config_len / baseline_len`` divide-by-zero / negative;
    the guard returns ``max(0, min(1, score))`` instead. Verify both the pass-through and the clamp.
    """
    assert _debias_verbosity(0.7, 100, 0) == 0.7  # baseline 0 -> identity (already in [0,1])
    assert _debias_verbosity(1.5, 100, -1) == 1.0  # negative baseline -> clamped down to 1.0
    assert _debias_verbosity(-0.3, 100, 0) == 0.0  # clamped up to 0.0


# --------------------------------------------------------------------------- #
# Offline judge heuristics: exception fallbacks + cleanliness branch
# --------------------------------------------------------------------------- #


def test_judge_predicate_category_branches_and_exception() -> None:
    """Lines 2345-2346: a config that parses to a non-mapping makes ``section.get`` raise -> ``return 1``.

    ``yaml.safe_load`` of a bare scalar yields a ``str``; ``_merge_first_section`` returns it unchanged
    (no ``template`` substring), then ``section.get("statement")`` raises ``AttributeError`` -> except.
    Also exercises the three normal returns (no predicate -> 0, prioritize -> 3, plain -> 2) for context.
    """
    assert _judge_predicate_category("just a string") == 1  # except branch (2345-2346)

    no_predicate = yaml.safe_dump({"statement": {"subject": {"method": "value", "encoding": "A"}, "object": {"method": "value", "encoding": "B"}}})
    assert _judge_predicate_category(no_predicate) == 0  # no predicate

    prioritize = yaml.safe_dump(
        {
            "statement": {
                "predicate": "associated_with",
                "subject": {"method": "column", "encoding": "A", "prioritize": ["x"]},
                "object": {"method": "value", "encoding": "B"},
            }
        }
    )
    assert _judge_predicate_category(prioritize) == 3  # subject has a prioritize dict

    plain = yaml.safe_dump(
        {
            "statement": {
                "predicate": "associated_with",
                "subject": {"method": "value", "encoding": "A"},
                "object": {"method": "value", "encoding": "B"},
            }
        }
    )
    assert _judge_predicate_category(plain) == 2  # predicate present, no prioritize


def test_judge_provenance_branches_and_exception() -> None:
    """Lines 2356-2357: a non-mapping config makes ``section.get`` raise -> ``return 0`` (except).

    Same trigger as the predicate heuristic: a bare scalar parses to ``str`` and ``section.get``
    raises. Also verifies complete provenance -> 3 and missing provenance -> 0 for context.
    """
    assert _judge_provenance("just a string") == 0  # except branch (2356-2357)
    assert _judge_provenance(yaml.safe_dump({"provenance": {"repo": "PMC", "publication": "PMC1"}})) == 3  # complete
    assert _judge_provenance(yaml.safe_dump({"provenance": {}})) == 0  # missing repo + publication


def test_judge_cleanliness_single_bad_call() -> None:
    """Line 2366: exactly one failed/wrong/redundant call (``bad == 1``) scores 2.

    Also pins the surrounding ladder: 0 bad -> 3, 2-3 bad -> 1, >3 bad -> 0.
    """
    assert _judge_cleanliness({"failed_tool_calls": 1}) == 2  # bad == 1 (line 2366)
    assert _judge_cleanliness({}) == 3  # bad == 0
    assert _judge_cleanliness({"wrong_tool_calls": 2}) == 1  # bad == 2 (<=3)
    assert _judge_cleanliness({"failed_tool_calls": 2, "wrong_tool_calls": 2, "redundant_tool_calls": 1}) == 0  # bad == 5 (>3)


# --------------------------------------------------------------------------- #
# _call_judge: non-callable model branches (lines 2374-2377)
# --------------------------------------------------------------------------- #


def test_call_judge_generate_method_and_plain_object() -> None:
    """Lines 2374-2377: a non-callable model is invoked via ``.generate`` or stringified as a fallback.

    An object with a callable ``.generate`` hits 2374-2376 (``generate(prompt)``); a plain object with
    no ``generate`` attribute hits 2374, 2375 (False) and the ``str(judge_model)`` fallback at 2377.
    """

    class WithGenerate:
        def generate(self, prompt: str) -> str:
            return f"GEN:{prompt}"

    assert _call_judge(WithGenerate(), "p") == "GEN:p"  # 2374-2376

    plain = object()
    assert _call_judge(plain, "p") == str(plain)  # 2374, 2375 False, 2377


# --------------------------------------------------------------------------- #
# judge_config: model-raise fallback (lines 2441-2442)
# --------------------------------------------------------------------------- #


def test_judge_config_falls_back_when_model_raises() -> None:
    """Lines 2441-2442: a judge_model that raises is caught and falls back to the offline heuristic.

    A callable model whose body raises propagates out of ``_call_judge`` into ``judge_config``'s
    ``try``; the ``except`` returns ``judge_config(...)`` with no model (the deterministic heuristic).
    """

    def bad_judge(prompt: str) -> str:  # pyright: ignore[reportUnusedParameter]
        raise RuntimeError("judge backend unavailable")

    result = judge_config(VALID_CFG, {"coverage_pct": 1.0, "qc_pass_rate": 1.0}, {"steps": 1}, judge_model=bad_judge)
    assert set(result["scores"]) == set(JUDGE_DIMENSIONS)
    assert result["rationale"].startswith("Offline heuristic judge")  # the fallback path ran
    assert 0.0 <= result["normalized"] <= 1.0


# --------------------------------------------------------------------------- #
# Reflexion: fullmap regression (2476) + abort (2480-2482)
# --------------------------------------------------------------------------- #


def test_reflexion_fullmap_regression_does_not_promote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Line 2476: with a fullmap, an edit whose re-audited coverage does NOT beat the best is not promoted.

    ``propose_config_edit`` is stubbed to return a schema-valid edit and ``build_and_audit`` to return a
    LOW coverage report, so ``cov2 > best_score`` is False and the ``else`` keeps ``current = edited``
    without promoting ``best``. Isolates Reflexion's control flow (no real build/network).
    """
    monkeypatch.setattr("tablassert.agent.propose_config_edit", lambda cfg, cov: (VALID_CFG, "stub edit"))
    monkeypatch.setattr("tablassert.agent.build_and_audit", lambda *a, **k: {"coverage_pct": 0.0})

    best, reflections = reflexion_improve(VALID_CFG, {"coverage_pct": 1.0}, {}, fullmap=Path("dummy.redb"))
    assert best == VALID_CFG  # regression not promoted: best stays the original
    assert reflections  # a reflection was recorded
    assert all("aborted" not in r for r in reflections)  # no exception path taken


def test_reflexion_aborts_gracefully_when_proposer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lines 2480-2482: an exception inside the reflection loop is caught, logged, and returns the best so far.

    Stubbing ``propose_config_edit`` to raise forces the loop body to throw; the ``except`` appends a
    ``reflection aborted: ...`` note and returns ``(best, reflections)`` rather than propagating.
    """

    def raiser(cfg: str, cov: dict[str, Any]) -> tuple[str, str]:  # pyright: ignore[reportUnusedParameter]
        raise RuntimeError("proposer exploded")

    monkeypatch.setattr("tablassert.agent.propose_config_edit", raiser)

    best, reflections = reflexion_improve(VALID_CFG, {"coverage_pct": 0.5}, {})
    assert best == VALID_CFG  # returned the original best unchanged
    assert any("reflection aborted" in r for r in reflections)  # abort note recorded (2481)


# --------------------------------------------------------------------------- #
# dspy GEPA: default program forward (2539) + dataset->Example path (2582)
# --------------------------------------------------------------------------- #


def test_default_gepa_program_forward_delegates() -> None:
    """Line 2539: ``_ConfigProposer.forward`` delegates to ``self.propose(table_summary, coverage_feedback)``.

    Builds the real default program (construction is offline-safe), stubs its ``propose`` predictor so no
    LM runs, and calls ``forward`` directly to execute the delegation return statement.
    """
    pytest.importorskip("dspy")
    prog = _default_gepa_program("SEED INSTRUCTIONS")
    prog.propose = lambda **kwargs: SimpleNamespace(config_yaml=f"STUB:{kwargs['table_summary']}")  # stub predictor (no LM)
    out = prog.forward(table_summary="tbl", coverage_feedback="cov")
    assert out.config_yaml == "STUB:tbl"  # forward delegated to self.propose (line 2539)


def test_run_gepa_builds_examples_from_dataset() -> None:
    """Line 2582: with ``trainset=None`` and a non-empty ``dataset``, run_gepa builds one dspy.Example per row.

    Mirrors the existing stub pattern: an injectable ``StubGEPA`` records the ``trainset`` it receives so
    we can assert the dataset row was converted into a single Example (the 2581-2586 loop body, incl. 2582).
    """
    pytest.importorskip("dspy")
    seen: dict[str, Any] = {}

    class StubGEPA:
        def __init__(self, metric: Any = None, **kwargs: Any) -> None:
            self.gepa_stats: dict[str, Any] = {}

        def compile(self, program: Any, *, trainset: Any = None, **kwargs: Any) -> Any:  # pyright: ignore[reportUnusedParameter]
            seen["trainset"] = trainset
            predictor = SimpleNamespace(signature=SimpleNamespace(instructions="OPT_DS"))
            return SimpleNamespace(named_predictors=lambda: [("propose", predictor)])

    result = run_gepa(
        seed_instructions="SEED", gepa_cls=StubGEPA, reflection_lm=SimpleNamespace(), dataset=[{"table_summary": "ts", "coverage_feedback": "cf"}]
    )
    assert seen["trainset"] is not None
    assert len(seen["trainset"]) == 1  # one Example built from the dataset row
    assert result["optimized_instructions"] == "OPT_DS"
