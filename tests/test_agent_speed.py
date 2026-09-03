"""Tests for the speed pass: pre-rendered task context + build memoization + planning off.

WHY this module exists: fleet logs showed ~2,100 ``pmc_article_context`` + ~2,500 ``read_table``
emissions and 68% of articles exhausting the 20-step budget largely on deterministic inspection
calls whose inputs the supervisor had ALREADY downloaded. These tests pin the three fixes:

1. ``render_task_context`` ships the article summary + head previews of EVERY candidate
   table/worksheet inside the task text (fail-visible notes instead of exceptions).
2. The ``build_and_audit`` tool memoizes identical configs via ``functools.lru_cache`` so a
   repeated call never pays a second full validate+build+coverage pass.
3. Periodic re-planning is disabled by default in ``build_agent`` (each planning turn is a whole
   extra LLM round trip) and INSTRUCTIONS target a short fixed workflow.

Every test is offline; the tool-cache test needs the ``[agent]`` extra.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from tablassert.agent import DATA_FENCE_BEGIN, INSTRUCTIONS, build_agent, build_and_audit, make_build_and_audit_tool, render_task_context


def _write_table(tmp_path: Path, text: str) -> Path:
    data: Path = tmp_path / "data.tsv"
    data.write_text(text)
    return data


# --------------------------------------------------------------------------- #
# render_task_context (pure; base env)
# --------------------------------------------------------------------------- #


def test_render_task_context_previews_tables_and_article(tmp_path: Path) -> None:
    """The block contains the article summary AND a fenced preview per candidate table.

    WHY: the whole point of the speed pass is that the agent never has to CALL
    pmc_article_context/read_table — their output must already be present, inside the
    untrusted-data fences so injection defenses still hold.
    """
    xml: Path = tmp_path / "article.xml"
    xml.write_text(
        '<?xml version="1.0"?><article><front><article-meta>'
        "<title-group><article-title>Gut microbiota study</article-title></title-group>"
        "</article-meta></front><body><sec><title>RESULTS</title><p>brca1 correlates with mapk1</p></sec></body></article>"
    )
    table: Path = _write_table(tmp_path, "gene\tpartner\nbrca1\tmapk1\n")

    out: str = render_task_context([table], xml, min_rows=0)

    assert "Gut microbiota study" in out  # article summary shipped
    assert DATA_FENCE_BEGIN in out  # article fence
    assert out.count(DATA_FENCE_BEGIN) >= 2  # article + table fences
    assert str(table) in out  # table preview carries its ABSOLUTE source path
    assert "brca1" in out  # actual cell data visible for authoring


def test_render_task_context_previews_every_excel_sheet(tmp_path: Path) -> None:
    """A multi-sheet workbook gets a head preview PER worksheet (one section per mappable sheet)."""
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl not installed")
    import openpyxl

    path: Path = tmp_path / "wb.xlsx"
    wb = openpyxl.Workbook()
    first = wb.active
    assert first is not None
    first.title = "correlations"
    first.append(["gene", "compound"])
    first.append(["brca1", "CHEBI:41774"])
    wb.create_sheet("metadata").append(["note"])
    wb.save(path)

    out: str = render_task_context([path], None, min_rows=0)

    assert "correlations" in out  # BOTH sheets previewed
    assert "metadata" in out
    assert "CHEBI:41774" in out


def test_render_task_context_skips_small_excel_sheets_and_names_focus(tmp_path: Path) -> None:
    """Only qualifying sheets are previewed, while the task names skipped sheets and focus sheets."""
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl not installed")
    import openpyxl

    path: Path = tmp_path / "filtered.xlsx"
    workbook = openpyxl.Workbook()
    first = workbook.active
    assert first is not None
    first.title = "large"
    first.append(["gene", "value"])
    for index in range(3):
        first.append([f"GENE{index}", index])
    small = workbook.create_sheet("small")
    small.append(["gene", "value"])
    small.append(["tiny", 1])
    workbook.save(path)

    out: str = render_task_context([path], None, min_rows=3)

    assert "focus on qualifying worksheets" in out
    assert "'large'=3" in out
    assert "skipped below 3 rows" in out
    assert "'small'=1" in out
    assert "GENE0" in out
    assert "tiny" not in out


def test_render_task_context_skips_small_delimited_file(tmp_path: Path) -> None:
    """A direct context render does not spend preview space on a small CSV/TSV."""
    table: Path = _write_table(tmp_path, "value\n1\n2\n")

    out: str = render_task_context([table], None, min_rows=3)

    assert "table data.tsv skipped: 2 rows < 3 minimum" in out
    assert DATA_FENCE_BEGIN not in out


def test_render_task_context_all_small_workbook_is_excluded(tmp_path: Path) -> None:
    """A direct context render makes an all-small workbook's exclusion explicit."""
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl not installed")
    import openpyxl

    path: Path = tmp_path / "all-small.xlsx"
    workbook = openpyxl.Workbook()
    first = workbook.active
    assert first is not None
    first.title = "metadata"
    first.append(["note"])
    first.append(["only one row"])
    workbook.save(path)

    out: str = render_task_context([path], None, min_rows=2)

    assert "NO sheet has >= 2 rows" in out
    assert "excluded from candidates" in out
    assert DATA_FENCE_BEGIN not in out


def test_render_task_context_sheet_cap_ignores_small_sheets(tmp_path: Path) -> None:
    """The worksheet preview cap is spent on qualifying sheets, not tiny metadata sheets."""
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl not installed")
    import openpyxl

    path: Path = tmp_path / "capped.xlsx"
    workbook = openpyxl.Workbook()
    first = workbook.active
    assert first is not None
    first.title = "tiny"
    first.append(["value"])
    first.append(["not-a-candidate"])
    for name, value in (("first", "FIRST_DATA"), ("second", "SECOND_DATA")):
        sheet = workbook.create_sheet(name)
        sheet.append(["value"])
        for _ in range(2):
            sheet.append([value])
    workbook.save(path)

    out: str = render_task_context([path], None, min_rows=2, max_sheets=1)

    assert "FIRST_DATA" in out
    assert "SECOND_DATA" not in out
    assert "not-a-candidate" not in out
    assert "+1 more qualifying worksheets not previewed" in out


def test_render_task_context_unreadable_table_is_fail_visible(tmp_path: Path) -> None:
    """An unreadable table renders a VISIBLE note naming the fallback — it must never raise."""
    garbage: Path = tmp_path / "garbage.xlsx"
    garbage.write_bytes(b"not a real xlsx")

    out: str = render_task_context([garbage], None, min_rows=0)

    assert "could not be previewed" in out
    assert "read_table" in out  # the agent is told exactly which fallback to use


def test_render_task_context_truncates_at_max_chars(tmp_path: Path) -> None:
    """The joined block is capped so a pathological article cannot flood the context."""
    tables: list[Path] = [_write_table(tmp_path, "a\tb\n" * 50) for _ in range(5)]

    out: str = render_task_context(tables, None, max_chars=500, min_rows=0)

    assert len(out) <= 500 + 200  # cap + one explicit marker line
    assert "task context truncated" in out


# --------------------------------------------------------------------------- #
# Prompt + planner defaults (pure)
# --------------------------------------------------------------------------- #


def test_instructions_target_short_workflow_with_fallback_tools() -> None:
    """INSTRUCTIONS prescribe the short derive->build->edit->answer workflow.

    WHY: the old prompt MANDATED read_table/pmc_article_context first (2+ wasted steps per PMC);
    the rewrite must make those tools explicit FALLBACKS while keeping the ReAct framing and the
    final_answer gate that other tests rely on.
    """
    assert "4 steps or fewer" in INSTRUCTIONS
    assert "ReAct" in INSTRUCTIONS
    assert "final_answer" in INSTRUCTIONS
    assert "call pmc_article_context(path) FIRST" not in INSTRUCTIONS  # old mandated step gone
    assert "FALLBACKS" in INSTRUCTIONS.upper()


def test_build_agent_disables_periodic_planning_by_default() -> None:
    """planning_interval defaults to None (smolagents skips every planning turn).

    WHY: each planning turn is a full extra LLM round trip carrying the entire prompt; with a
    fixed short workflow it bought nothing but tokens/latency.
    """
    default: Any = inspect.signature(build_agent).parameters["planning_interval"].default
    assert default is None


# --------------------------------------------------------------------------- #
# build_and_audit tool memoization ([agent] extra)
# --------------------------------------------------------------------------- #


def test_build_and_audit_tool_memoizes_identical_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated identical config returns the cached report WITHOUT a second build.

    WHY: models re-run unchanged configs despite instructions; each repeat paid a full
    validate+build+coverage pass on a fresh tempdir. functools.lru_cache must collapse them,
    while a DIFFERENT config still builds for real.
    """
    pytest.importorskip("smolagents")
    from tablassert import rs

    root: Path = tmp_path / "fullmap"
    root.mkdir(parents=True)
    classes: Path = root / "classes.ndjson"
    classes.write_text(json.dumps({"id": "HGNC:1100", "equivalent_identifiers": [{"identifier": "NCBIGene:672"}]}) + "\n")
    synonyms: Path = root / "synonyms.ndjson"
    synonyms.write_text(
        json.dumps({"curie": "HGNC:1100", "preferred_name": "BRCA1", "names": ["BRCA1", "brca1"], "types": ["Gene"], "taxa": ["NCBITaxon:9606"]})
        + "\n"
    )
    output: Path = root / "data" / "fullmap.redb"
    rs.build_fullmap_db(output, [classes], [synonyms], threads=2)

    calls: list[str] = []
    real_build_and_audit = build_and_audit

    def counting_build_and_audit(config_yaml: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(config_yaml)
        return real_build_and_audit(config_yaml, **kwargs)

    monkeypatch.setattr("tablassert.agent.build_and_audit", counting_build_and_audit)

    table: Path = _write_table(tmp_path, "brca1\tbrca1\n")
    config_yaml: str = (
        "source:\n"
        f"  url:\n    - https://example.com/data.tsv\n"
        f"  local: {table}\n"
        "  kind: text\n"
        '  delimiter: "\\t"\n'
        "statement:\n"
        "  subject: {method: column, encoding: A}\n"
        "  predicate: associated_with\n"
        "  object: {method: column, encoding: B}\n"
        "provenance: {repo: PMC, publication: PMC1}\n"
    )
    tool = make_build_and_audit_tool(lambda: output)

    first: str = tool.forward(config_yaml)
    second: str = tool.forward(config_yaml)

    assert len(calls) == 1  # identical repeat served from cache, zero extra builds
    assert first == second
    assert json.loads(first)["ok"] is True

    changed: str = tool.forward(config_yaml.replace("associated_with", "causes"))
    assert len(calls) == 2  # a genuinely different config still builds
    assert json.loads(changed)["ok"] is True
