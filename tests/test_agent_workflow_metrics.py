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
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tablassert import rs
from tablassert.agent import make_fake_model, make_step_callback, run_supervisor

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
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)
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
