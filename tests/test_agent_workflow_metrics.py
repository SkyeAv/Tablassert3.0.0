"""Tests for US-001: OFFLINE workflow-efficiency measurement harness + pre-redesign baseline.

The harness pins HOW the current pipeline spends its budget — steps, tokens, and tool calls —
through the seam the supervisor already exposes: ``run_supervisor([pmc], fullmap=<tiny REAL
redb>, build_model_factory=lambda: make_fake_model(responses=[...]), state_dir=tmp, workdir=tmp)
-> result["metrics"], result["records"]``. The inner ``CodeAgent`` runs OFFLINE against canned
transcripts while ``make_step_callback`` tallies every step into ``state.metrics``.

The named ``BASELINE_*`` constants were captured by RUNNING this harness against the current
(pre-redesign) code; they pin the efficiency profile so US-006 delta assertions are honest —
any future change that moves step/token/tool-call accounting fails here loudly first, forcing
a deliberate re-baseline.

Everything is hermetic: ``fetch_pmc_article`` is monkeypatched (no network), ``FakeModel``
supplies canned responses (no live model), and an autouse fixture disables HuggingFace
telemetry (enabled telemetry blocks ``agent.run`` on a network call). NO wall-clock
assertions: efficiency is measured in steps/tokens/tool calls, which are deterministic offline.

US-006 (closeout) extends this module with the DELTA side of the story, still fully offline:
the full-mode agent's exact four-tool surface, the guarantee that the supervisor-only helpers
(``map_coverage`` / ``propose_config_edit``) never reach that surface or its transcript, a
canonical scripted derive→build→answer run whose budget never exceeds the US-001 baseline
constants above, and a golden-fixture accuracy-invariance assertion proving the efficiency
redesign (and the US-005 config compaction it persists) changed HOW the budget is spent
without changing WHAT the pipeline scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import (
    build_agent,
    build_and_audit,
    compact_config,
    config_size_metric,
    load_kgx,
    make_fake_model,
    make_step_callback,
    make_tools,
    node_edge_f1,
    quality_score,
    run_supervisor,
    validate_table_config,
)

pytest.importorskip("smolagents")


@pytest.fixture(autouse=True)
def _offline_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``agent.run`` hermetically offline by disabling HuggingFace telemetry.

    Why: a real ``CodeAgent.run`` fires huggingface_hub telemetry that BLOCKS on a network call
    (observed: the run hangs indefinitely at zero CPU when telemetry is enabled). Every test here
    drives a real agent, so the standard opt-out vars are mandatory. Scoped to this file (NOT
    global conftest) so the QC suite's own HuggingFace model loading is unaffected.
    """
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")


# --------------------------------------------------------------------------- #
# Baseline capture (US-001): efficiency counts of the scripted transcript on the
# CURRENT (pre-redesign) pipeline, captured by RUNNING this harness OFFLINE at
#   commit 3603031a9af5a91640d42819f94d9f95f01816aa (2026-09-03, branch
#   more-efficent-workflows, smolagents 1.26.0).
# US-006 delta assertions compare against these; update ONLY via a deliberate
# re-capture (run this module, update the constants AND this comment).
# --------------------------------------------------------------------------- #

# Scripted transcript: one ``read_table`` fallback step + one ``pmc_article_context`` fallback
# step + the final-answer step. smolagents' CodeAgent registers ONE ``python_interpreter``
# ToolCall per executed code block, so every fallback-tool step lands in ``total_tool_calls``.
BASELINE_SCRIPTED_TOTAL_STEPS: int = 3
BASELINE_SCRIPTED_TOTAL_TOOL_CALLS: int = 3
BASELINE_SCRIPTED_TOTAL_TOKENS: int = 45
BASELINE_SCRIPTED_FAILED_TOOL_CALLS: int = 0
BASELINE_SCRIPTED_WRONG_TOOL_CALLS: int = 0
BASELINE_SCRIPTED_REDUNDANT_TOOL_CALLS: int = 0

# Plain transcript (final answer only) — the efficiency floor the scripted run is measured over.
BASELINE_PLAIN_TOTAL_STEPS: int = 1
BASELINE_PLAIN_TOTAL_TOOL_CALLS: int = 1
BASELINE_PLAIN_TOTAL_TOKENS: int = 15

# FakeModel attaches TokenUsage(input=10, output=5) to every generate -> 15 total per step.
TOKENS_PER_FAKE_STEP: int = 15


# --------------------------------------------------------------------------- #
# Offline fixtures: tiny REAL redb + small text table + fetch seam (mirror
# tests/test_agent_supervisor.py so both suites measure the identical pipeline).
# --------------------------------------------------------------------------- #


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _synonym_row(curie: str, preferred_name: str, names: list[str], category: str) -> dict[str, Any]:
    return {"curie": curie, "preferred_name": preferred_name, "names": names, "types": [category], "taxa": ["NCBITaxon:9606"]}


def _class_row(curie: str, equivalents: list[str]) -> dict[str, Any]:
    return {"id": curie, "equivalent_identifiers": [{"identifier": x} for x in equivalents]}


@pytest.fixture
def fullmap_db(tmp_path: Path) -> Path:
    """A tiny REAL fullmap redb: ``brca1`` -> HGNC:1100, ``mapk1`` -> HGNC:6871."""
    root: Path = tmp_path / "fullmap"
    root.mkdir(parents=True, exist_ok=True)
    classes: Path = _write_jsonl(root / "classes.ndjson", [_class_row("HGNC:1100", ["NCBIGene:672"])])
    synonyms: Path = _write_jsonl(
        root / "synonyms.ndjson",
        [_synonym_row("HGNC:1100", "BRCA1", ["BRCA1", "brca1"], "Gene"), _synonym_row("HGNC:6871", "MAPK1", ["MAPK1", "mapk1"], "Gene")],
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms])
    return output


def _write_table(tmp_path: Path, name: str, text: str) -> Path:
    table: Path = tmp_path / name
    table.write_text(text)
    return table


def _column_cfg(table: Path) -> dict[str, Any]:
    """A valid merged Section config: subject=column A, object=column B, PMC provenance."""
    return {
        "source": {"kind": "text", "local": str(table), "url": ["https://e.com/d.tsv"], "delimiter": "\t"},
        "statement": {
            "subject": {"method": "column", "encoding": "A"},
            "predicate": "associated_with",
            "object": {"method": "column", "encoding": "B"},
        },
        "provenance": {"repo": "PMC", "publication": "PMC1"},
    }


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, files: list[Path]) -> None:
    """Monkeypatch ``fetch_pmc_article`` to return ``files`` (the supervisor's ONLY network seam)."""

    def fake_fetch(pmc_id: str, outdir: Path, *, timeout: int = 120) -> list[Path]:  # pyright: ignore[reportUnusedParameter]
        return list(files)

    monkeypatch.setattr("tablassert.agent.fetch_pmc_article", fake_fetch)


def _run_offline_supervisor(
    tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch, *, responses: list[str] | None, fetched: list[Path], pmc_id: str = "PMC1"
) -> dict[str, Any]:
    """Drive ``run_supervisor`` OFFLINE with a canned FakeModel transcript over the tiny redb.

    Returns the supervisor result dict; the harness contract under test is
    ``result["metrics"]`` + ``result["records"]``.
    """
    _patch_fetch(monkeypatch, fetched)
    table: Path = next(path for path in fetched if path.suffix == ".tsv")
    good_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)
    result: dict[str, object] = run_supervisor(
        [pmc_id],
        fullmap=fullmap_db,
        build_model_factory=lambda: make_fake_model(responses=responses, final_yaml=good_yaml),
        map_threshold=0.8,
        state_dir=tmp_path / "state",
        workdir=tmp_path / "w",
        min_rows=0,
    )
    return result  # pyright: ignore[reportReturnType]


def _action_step(
    *,
    tool_calls: list[tuple[str, object]] | None = None,
    tokens: tuple[int, int] | None = (10, 5),
    error: Exception | None = None,
    observations: str | None = None,
) -> SimpleNamespace:
    """A synthetic smolagents-shaped ActionStep (the callback only reads attributes via getattr)."""
    usage: SimpleNamespace | None = None
    if tokens is not None:
        usage = SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1], total_tokens=tokens[0] + tokens[1])
    calls: list[SimpleNamespace] = [SimpleNamespace(name=name, arguments=arguments) for name, arguments in (tool_calls or [])]
    return SimpleNamespace(token_usage=usage, error=error, tool_calls=calls, observations=observations)


# --------------------------------------------------------------------------- #
# Per-step tally semantics (the measurement the supervisor aggregates)
# --------------------------------------------------------------------------- #


def test_step_callback_tallies_steps_tokens_and_tool_calls() -> None:
    """``make_step_callback`` accumulates steps, token usage, and tool-call volume per step.

    Why: the per-step dict is the raw feed ``run_supervisor`` aggregates into ``state.metrics``
    (``steps`` surfaces there as ``total_steps``); pinning its exact accumulation here makes any
    future accounting drift visible BEFORE it silently corrupts a baseline comparison.
    """
    metrics: dict[str, object] = {}
    cb = make_step_callback(metrics)
    agent = SimpleNamespace(memory=None)

    cb(_action_step(tool_calls=[("read_table", {"source": "t.tsv"})]), agent)
    cb(_action_step(tool_calls=[("pmc_article_context", {"source": "a.txt"}), ("read_table", {"source": "u.tsv"})]), agent)

    assert metrics["steps"] == 2
    assert metrics["input_tokens"] == 20
    assert metrics["output_tokens"] == 10
    assert metrics["total_tokens"] == 30
    assert metrics["total_tool_calls"] == 3
    assert metrics.get("failed_tool_calls", 0) == 0
    assert metrics.get("redundant_tool_calls", 0) == 0


def test_step_callback_counts_failed_wrong_and_redundant_tool_calls() -> None:
    """Per-tool-call quality accounting: failed (step error), wrong (error-flavored observation), redundant (repeated signature).

    Why: US-006 efficiency deltas are not just volume — a redesign that trades steps for
    retries must show up in these counters; the exact trigger conditions are pinned here.
    """
    metrics: dict[str, object] = {}
    cb = make_step_callback(metrics)
    agent = SimpleNamespace(memory=None)
    signature: tuple[str, object] = ("read_table", {"source": "t.tsv"})

    cb(_action_step(tool_calls=[signature]), agent)
    cb(_action_step(tool_calls=[signature]), agent)  # identical (name, arguments) -> redundant
    cb(_action_step(error=RuntimeError("boom")), agent)  # step error -> failed AND wrong
    cb(_action_step(observations="Traceback (most recent call last): ..."), agent)  # error-flavored text -> wrong

    assert metrics["total_tool_calls"] == 2
    assert metrics["redundant_tool_calls"] == 1
    assert metrics["failed_tool_calls"] == 1
    assert metrics["wrong_tool_calls"] == 2
    assert metrics["steps"] == 4


def test_step_callback_survives_unexpected_shapes() -> None:
    """The callback never raises on odd step/metric shapes and never counts non-int values.

    Why: it runs inside every agent step of a long batch; a defensive miss there would abort a
    whole supervisor run, so the guard contract (getattr everywhere, non-int -> 0) is pinned.
    """
    metrics: dict[str, object] = {"steps": "corrupt"}  # a pre-existing non-int value is treated as 0
    cb = make_step_callback(metrics)

    cb(object(), SimpleNamespace(memory=None))  # a step with no known attributes must still count
    assert metrics["steps"] == 1
    assert metrics.get("total_tool_calls", 0) == 0

    # Context trimming: only steps older than the last 2 shrink; recent observations are untouched.
    old = SimpleNamespace(observations="x" * 5000)
    recent_a = SimpleNamespace(observations="y" * 5000)
    recent_b = SimpleNamespace(observations="z" * 5000)
    agent = SimpleNamespace(memory=SimpleNamespace(steps=[old, recent_a, recent_b]))
    cb(_action_step(), agent)
    assert old.observations == "[trimmed observation: 5000 chars]"
    assert recent_a.observations == "y" * 5000
    assert recent_b.observations == "z" * 5000


# --------------------------------------------------------------------------- #
# End-to-end harness: scripted transcripts -> supervisor metrics (the baselines)
# --------------------------------------------------------------------------- #


def test_scripted_transcript_baseline_metrics(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins the pre-redesign baseline so US-006 delta assertions are honest.

    The canned transcript spends two steps on the fallback inspection tools (``read_table`` +
    ``pmc_article_context``) before the final answer; every step registers in the aggregated
    metrics (steps surface as ``total_steps``; smolagents' CodeAgent records ONE
    ``python_interpreter`` ToolCall per executed code block, so the fallback-tool steps are
    counted in ``total_tool_calls`` with zero failed/wrong/redundant noise).
    """
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    article: Path = tmp_path / "article.txt"
    article.write_text("A small article body for the fallback context tool.")
    responses: list[str] = [
        f"<code>\npreview = read_table(source={str(table.resolve())!r})\nprint('preview-fence-ok:', 'PMC_DATA_BEGIN' in preview)\n</code>",
        f"<code>\nctx = pmc_article_context(source={str(article.resolve())!r})\nprint('ctx-fence-ok:', 'PMC_DATA_BEGIN' in ctx)\n</code>",
    ]

    result = _run_offline_supervisor(tmp_path, fullmap_db, monkeypatch, responses=responses, fetched=[table, article])

    assert result["records"]["PMC1"].status == "MAPPED"  # the transcript completed the real pipeline
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    # Harness contract: the efficiency counters exist and carry the baseline counts.
    assert metrics["total_steps"] == BASELINE_SCRIPTED_TOTAL_STEPS
    assert metrics["total_tool_calls"] == BASELINE_SCRIPTED_TOTAL_TOOL_CALLS
    assert metrics["total_tokens"] == BASELINE_SCRIPTED_TOTAL_TOKENS
    assert metrics["failed_tool_calls"] == BASELINE_SCRIPTED_FAILED_TOOL_CALLS
    assert metrics["wrong_tool_calls"] == BASELINE_SCRIPTED_WRONG_TOOL_CALLS
    assert metrics["redundant_tool_calls"] == BASELINE_SCRIPTED_REDUNDANT_TOOL_CALLS
    # Structural accounting invariants of the offline harness (FakeModel emits 15 tokens/step).
    assert metrics["total_tool_calls"] == metrics["total_steps"]  # one code-block ToolCall per step
    total_steps: object = metrics["total_steps"]
    assert isinstance(total_steps, int)
    assert metrics["total_tokens"] == total_steps * TOKENS_PER_FAKE_STEP
    assert (tmp_path / "state" / "state.json").is_file()


def test_plain_transcript_baseline_metrics(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The efficiency floor: a final-answer-only transcript costs exactly one step/tool-call.

    Why: US-006 measures redesign savings as deltas over a transcript; this floor anchors the
    scale (any fixed per-article overhead beyond the single final step would fail loudly here).
    """
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")

    result = _run_offline_supervisor(tmp_path, fullmap_db, monkeypatch, responses=None, fetched=[table])

    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportAttributeAccessIssue]
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    assert metrics["total_steps"] == BASELINE_PLAIN_TOTAL_STEPS
    assert metrics["total_tool_calls"] == BASELINE_PLAIN_TOTAL_TOOL_CALLS
    assert metrics["total_tokens"] == BASELINE_PLAIN_TOTAL_TOKENS
    assert metrics["total_tokens"] == TOKENS_PER_FAKE_STEP


def test_failed_tool_call_is_registered(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A coded fallback-tool failure is measured, never swallowed: failed AND wrong increment.

    Why: the harness must see failure paths — a redesign that silently retries broken tool calls
    has to show up as failed/wrong deltas; the run still terminates MAPPED (one bad step does not
    abort the article), matching the supervisor's fail-loudly-but-continue contract.
    """
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    responses: list[str] = ["<code>\npreview = read_table(source='/nonexistent/missing.tsv')\nprint('unreachable')\n</code>"]

    result = _run_offline_supervisor(tmp_path, fullmap_db, monkeypatch, responses=responses, fetched=[table])

    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportAttributeAccessIssue]
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    assert metrics["total_steps"] == 2  # the failed step + the final-answer step
    assert metrics["total_tool_calls"] == 2
    assert metrics["failed_tool_calls"] == 1
    assert metrics["wrong_tool_calls"] == 1  # a step error also trips the wrong-call heuristic
    assert metrics["redundant_tool_calls"] == 0


def test_redundant_tool_call_is_registered(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated identical step is measured as redundant — the retry-churn signal.

    Why: redundancy accounting is what lets US-006 detect a redesign that re-runs the same
    inspection instead of reusing it; pinning the exact trigger end-to-end keeps that honest.
    """
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    block: str = "<code>\nprint('warmup')\n</code>"

    result = _run_offline_supervisor(tmp_path, fullmap_db, monkeypatch, responses=[block, block], fetched=[table])

    assert result["records"]["PMC1"].status == "MAPPED"  # pyright: ignore[reportAttributeAccessIssue]
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    assert metrics["total_steps"] == 3  # two warmup steps + the final-answer step
    assert metrics["total_tool_calls"] == 3
    assert metrics["redundant_tool_calls"] == 1  # the second identical code block
    assert metrics["failed_tool_calls"] == 0


# --------------------------------------------------------------------------- #
# US-006 closeout (deltas vs the US-001 baseline): the redesigned full-mode
# agent surface is EXACTLY four tools, the supervisor-only helpers never reach
# the agent or its transcript, and the canonical scripted derive→build→answer
# run stays within the pre-redesign budget with zero redundant calls. All
# OFFLINE; every assertion is a delta against the BASELINE_* constants above —
# the constants themselves are frozen (no weakening, no wall-clock claims).
# --------------------------------------------------------------------------- #

#: The exact LLM-facing tool surface the full-mode supervisor registers (US-002).
FULL_MODE_TOOL_NAMES: tuple[str, ...] = ("read_table", "pmc_article_context", "derive_config", "build_and_audit")
#: Pure helpers ONLY the deterministic supervisor may call; never the full-mode agent.
SUPERVISOR_ONLY_HELPERS: tuple[str, ...] = ("map_coverage", "propose_config_edit")


def test_full_mode_make_tools_exposes_exactly_the_four_tool_names(fullmap_db: Path) -> None:
    """US-002/US-006 delta: full mode registers EXACTLY the four derive→build→answer tools.

    Why: the pre-redesign surface handed the LLM ``map_coverage`` and ``propose_config_edit``
    and let it churn coverage loops itself; the redesign moves coverage improvement into the
    deterministic supervisor. This pins the new contract set-theoretically: exactly the four
    names, in the documented order, and neither supervisor helper among them. The two batch
    derive modes are pinned in the same call so a future surface change anywhere in
    ``make_tools`` surfaces here first.
    """
    full: list[object] = make_tools(fullmap=fullmap_db)
    full_names: list[str] = [tool.name for tool in full]  # pyright: ignore[reportAttributeAccessIssue]
    assert len(full) == 4
    assert set(full_names) == set(FULL_MODE_TOOL_NAMES)
    assert full_names == list(FULL_MODE_TOOL_NAMES)  # the documented order
    for helper in SUPERVISOR_ONLY_HELPERS:
        assert helper not in full_names, f"supervisor-only helper {helper} leaked into the full-mode LLM surface"

    # derive_only: the three inspection/authoring tools, no fullmap tools at all.
    derive_only_names: list[str] = [tool.name for tool in make_tools(fullmap=fullmap_db, derive_mode="derive_only")]  # pyright: ignore[reportAttributeAccessIssue]
    assert derive_only_names == ["read_table", "pmc_article_context", "derive_config"]

    # derive_coverage: map_coverage REPLACES build_and_audit (coverage feedback without a KGX
    # build); propose_config_edit is still never a tool.
    coverage_names: list[str] = [tool.name for tool in make_tools(fullmap=fullmap_db, derive_mode="derive_coverage")]  # pyright: ignore[reportAttributeAccessIssue]
    assert coverage_names == ["read_table", "pmc_article_context", "derive_config", "map_coverage"]
    assert "propose_config_edit" not in coverage_names


def test_full_agent_transcript_never_sees_the_supervisor_helpers(fullmap_db: Path) -> None:
    """US-006 delta: no supervisor helper name exists in the full agent's tool set OR transcript.

    Why: a name the agent can see (tool table or instructions) is a name it can call, and the
    whole point of the US-002 slimming is that coverage improvement happens in deterministic
    Python AFTER the agent answers. This builds the agent EXACTLY as ``run_supervisor`` does
    (``build_agent`` + ``make_tools`` + the default INSTRUCTIONS), runs a canned FakeModel
    transcript to completion, and scans the full rendered memory — system prompt, task, every
    model output, observation, and code action — for the helper names. Both must be absent
    while all four registered tools are present (the transcript really does carry the tool
    table, so the absence is meaningful, not vacuous).
    """
    from tablassert.agent import INSTRUCTIONS

    for helper in SUPERVISOR_ONLY_HELPERS:
        assert helper not in INSTRUCTIONS, f"supervisor-only helper {helper} leaked into the system prompt"

    agent: object = build_agent(model=make_fake_model(), tools=make_tools(fullmap=fullmap_db), max_steps=5)
    tool_names: set[str] = set(agent.tools.keys())  # pyright: ignore[reportAttributeAccessIssue]
    assert set(FULL_MODE_TOOL_NAMES) <= tool_names
    for helper in SUPERVISOR_ONLY_HELPERS:
        assert helper not in tool_names, f"supervisor-only helper {helper} registered as an agent tool"

    agent.run("Return a valid minimal table config YAML.")  # pyright: ignore[reportAttributeAccessIssue]

    # The full transcript: system prompt + every memory step rendered to text.
    parts: list[str] = [agent.memory.system_prompt.system_prompt]  # pyright: ignore[reportAttributeAccessIssue]
    for step in agent.memory.steps:  # pyright: ignore[reportAttributeAccessIssue]
        for attribute in ("task", "model_output", "observations", "code_action"):
            value: object = getattr(step, attribute, None)
            if isinstance(value, str):
                parts.append(value)
    transcript: str = "\n".join(parts)
    assert transcript  # guard: an empty scan would pass vacuously
    for helper in SUPERVISOR_ONLY_HELPERS:
        assert helper not in transcript, f"supervisor-only helper {helper} surfaced in the agent transcript"
    for name in FULL_MODE_TOOL_NAMES:
        assert name in transcript, f"registered tool {name} missing from the transcript's tool table"


def test_us006_canonical_workflow_stays_within_baseline_budget(tmp_path: Path, fullmap_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """US-006 delta: the canonical scripted derive→build→answer run fits the pre-redesign budget.

    Why: this is THE efficiency claim of the redesign, measured deterministically instead of by
    wall clock. The pre-redesign baseline transcript spent 3 steps / 3 tool calls on pure
    inspection (``read_table`` + ``pmc_article_context`` fallbacks) before answering. The
    scripted transcript here instead spends its steps on the REAL workflow — one
    ``derive_config`` authoring step and one ``build_and_audit`` validate+build+score step —
    because the supervisor pre-renders all inspection context into the task. It must complete
    MAPPED with ``total_steps`` and ``total_tool_calls`` at or under the frozen US-001 baseline
    constants and ZERO redundant tool calls (no retry churn); the baseline constants themselves
    are untouched, so any future regression that re-inflates the canonical workflow fails here.
    """
    table: Path = _write_table(tmp_path, "good.tsv", "brca1\tmapk1\nbrca1\tmapk1\n")
    good_yaml: str = yaml.safe_dump(_column_cfg(table), sort_keys=False)  # identical to the helper's final_yaml
    responses: list[str] = [
        f"<code>\nderived = derive_config(config_yaml={good_yaml!r})\nprint('derived-ok:', 'source:' in derived)\n</code>",
        f"<code>\naudit = build_and_audit(config_yaml={good_yaml!r})\nprint('audit-ok:', '\"ok\": true' in audit)\n</code>",
    ]

    result = _run_offline_supervisor(tmp_path, fullmap_db, monkeypatch, responses=responses, fetched=[table])

    assert result["records"]["PMC1"].status == "MAPPED"  # the canonical workflow completes the real pipeline
    metrics: dict[str, object] = result["metrics"]  # pyright: ignore[reportAssignmentType]
    total_steps: object = metrics["total_steps"]
    total_tool_calls: object = metrics["total_tool_calls"]
    assert isinstance(total_steps, int)
    assert isinstance(total_tool_calls, int)
    assert total_steps <= BASELINE_SCRIPTED_TOTAL_STEPS, "canonical workflow exceeds the pre-redesign step budget"
    assert total_tool_calls <= BASELINE_SCRIPTED_TOTAL_TOOL_CALLS, "canonical workflow exceeds the pre-redesign tool-call budget"
    assert metrics["redundant_tool_calls"] == 0  # no repeated call churn
    assert metrics["failed_tool_calls"] == 0
    assert metrics["wrong_tool_calls"] == 0

    # US-005 wiring is live in the supervisor path: state.json records ``config_chars`` as the
    # exact length of the compacted best config actually persisted (not of any intermediate).
    state: dict[str, Any] = json.loads((tmp_path / "state" / "state.json").read_text())
    record: dict[str, Any] = state["records"]["PMC1"]
    best_path: Path = tmp_path / "state" / "configs" / "PMC1.yaml"
    assert best_path.is_file()
    assert record["config_chars"] == len(best_path.read_text())


# --------------------------------------------------------------------------- #
# US-006 closeout: OFFLINE accuracy invariance on the golden fixture.
#
# The efficiency redesign (US-002..US-005) changed HOW the agent spends its
# budget and WHAT gets persisted — it must not change WHAT the pipeline scores.
# This rebuilds the committed golden PMC11708054 reference config against a tiny
# REAL redb (the same recipe as tests/test_agent_eval.py) and pins the resulting
# quality_score to a named recorded baseline; then it proves the US-005 compaction
# of that config is semantics-preserving: identical scored quality and an
# identical emitted KGX (as sets — row order is not stable across builds).
# --------------------------------------------------------------------------- #

GOLDEN_FIXTURE_DIR: Path = Path(__file__).parent / "agent_fixtures" / "PMC11708054"

#: Recorded US-006 accuracy baseline: ``quality_score`` of the golden reference config built
#: offline against the tiny redb below. Deterministic decomposition with the FROZEN quality
#: weights (never changed by the redesign): schema validity 0.1*1.0 + coverage 0.4*1.0 +
#: Biolink validity 0.25*(8/15) + QC 0.1*0.0 (unmeasurable -> 0) + mean self-F1 0.15*1.0
#: = 47/60. Any drift means a behavior change, not a re-scoring: the weights and the golden
#: fixture are immutable.
ACCURACY_BASELINE_QUALITY_SCORE: float = 47.0 / 60.0


def _golden_reference_yaml() -> str:
    """The golden reference config with its ``source.local`` pointed at the absolute fixture CSV.

    Same rewrite ``tests/test_agent_eval.py::test_reference_kgx_builds_and_self_f1`` performs:
    the committed config carries a relative path, and the build chdir's into its workdir.
    """
    cfg: dict[str, Any] = yaml.safe_load((GOLDEN_FIXTURE_DIR / "reference_config.yaml").read_text())
    cfg["template"]["source"]["local"] = str((GOLDEN_FIXTURE_DIR / "source_table.csv").resolve())
    return yaml.safe_dump(cfg, sort_keys=False)


def _build_golden_redb(root: Path) -> Path:
    """Tiny REAL redb registering the golden fixture organisms + CHEBI:41774 (ChemicalEntity).

    The identical recipe ``tests/test_agent_eval.py::_build_reference_redb`` uses (kept local so
    this module stays self-contained, mirroring how its ``fullmap_db`` fixture is self-contained).
    """
    root.mkdir(parents=True, exist_ok=True)
    organisms: list[str] = [
        "Lactobacillus rhamnosus",
        "Bacteroides fragilis",
        "Clostridium sp",
        "Escherichia coli",
        "Faecalibacterium prausnitzii",
        "Bifidobacterium longum",
        "Akkermansia muciniphila",
    ]
    synonyms: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    for i, name in enumerate(organisms, start=100):
        curie: str = f"NCBITaxon:{i}"
        synonyms.append({"curie": curie, "preferred_name": name, "names": [name.lower(), name], "types": ["OrganismTaxon"], "taxa": ["NCBITaxon:1"]})
        classes.append({"id": curie, "equivalent_identifiers": [{"identifier": curie}]})
    synonyms.append(
        {
            "curie": "CHEBI:41774",
            "preferred_name": "13C-tamoxifen",
            "names": ["chebi:41774", "13c-tamoxifen"],
            "types": ["ChemicalEntity"],
            "taxa": ["NCBITaxon:0"],
        }
    )
    classes.append({"id": "CHEBI:41774", "equivalent_identifiers": [{"identifier": "CHEBI:41774"}]})
    classes_path: Path = _write_jsonl(root / "classes.ndjson", classes)
    synonyms_path: Path = _write_jsonl(root / "synonyms.ndjson", synonyms)
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes_path], [synonyms_path])
    return output


def _kgx_keyset(rows: list[dict[str, Any]]) -> set[str]:
    """Order-insensitive identity of a KGX row list (row ORDER is not stable across builds)."""
    return {json.dumps(row, sort_keys=True) for row in rows}


def test_us006_accuracy_invariant_on_golden_fixture(tmp_path: Path) -> None:
    """US-006: the efficiency redesign preserves scored accuracy, pinned to a recorded baseline.

    Why: step/tool-call deltas prove the budget shrank; this test proves the RESULT did not.
    The golden reference config is built offline against a tiny real redb, and its
    ``quality_score`` (coverage/Biolink/QC/F1/validity with the frozen weights) must equal the
    named ``ACCURACY_BASELINE_QUALITY_SCORE``. Then the US-005 persisted-config compaction is
    exercised on the same config: it must shrink the YAML, stay schema-valid and idempotent,
    preserve the section count, and — rebuilt — produce the SAME scored quality and the SAME
    emitted KGX (node/edge sets and cross-build F1 of 1.0). Any semantic drift in the
    compaction, or any scoring drift in the redesigned pipeline, fails here loudly.
    """
    config_yaml: str = _golden_reference_yaml()
    db: Path = _build_golden_redb(tmp_path / "fullmap")

    report: dict[str, object] = build_and_audit(config_yaml, fullmap=db, workdir=tmp_path / "build_original")
    assert report["ok"] is True, f"golden reference build failed: {report.get('errors')}"
    nodes: list[dict[str, Any]] = load_kgx(Path(str(report["kgx_path"])))
    edges: list[dict[str, Any]] = load_kgx(Path(str(report["edges_path"])))
    assert nodes, "golden reference build emitted no nodes"
    assert edges, "golden reference build emitted no edges"
    self_f1: dict[str, float] = node_edge_f1(nodes, edges, nodes, edges)
    assert quality_score(config_yaml, report, self_f1) == pytest.approx(ACCURACY_BASELINE_QUALITY_SCORE)

    # US-005 compaction: provably shrinks the config, never its meaning.
    compacted: str = compact_config(config_yaml)
    assert validate_table_config(compacted)
    assert len(compacted) < len(config_yaml), "compaction removed no provably-no-op entries from the golden config"
    assert compact_config(compacted) == compacted  # idempotent
    assert config_size_metric(compacted)["sections"] == config_size_metric(config_yaml)["sections"]

    compact_report: dict[str, object] = build_and_audit(compacted, fullmap=db, workdir=tmp_path / "build_compacted")
    assert compact_report["ok"] is True, f"compacted golden build failed: {compact_report.get('errors')}"
    for key in ("coverage_pct", "biolink_valid_pct", "qc_pass_rate", "node_count", "edge_count"):
        assert compact_report[key] == report[key], f"compaction moved build metric {key!r}"

    compact_nodes: list[dict[str, Any]] = load_kgx(Path(str(compact_report["kgx_path"])))
    compact_edges: list[dict[str, Any]] = load_kgx(Path(str(compact_report["edges_path"])))
    assert _kgx_keyset(nodes) == _kgx_keyset(compact_nodes), "compaction changed the emitted KGX nodes"
    assert _kgx_keyset(edges) == _kgx_keyset(compact_edges), "compaction changed the emitted KGX edges"
    cross_f1: dict[str, float] = node_edge_f1(nodes, edges, compact_nodes, compact_edges)
    assert cross_f1["node_f1"] == 1.0
    assert cross_f1["edge_f1"] == 1.0
    assert quality_score(compacted, compact_report, cross_f1) == pytest.approx(ACCURACY_BASELINE_QUALITY_SCORE)
