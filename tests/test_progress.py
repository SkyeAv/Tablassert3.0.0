from __future__ import annotations

from typing import Any

from tablassert.progress import PipelineProgress


def _section_task(p: PipelineProgress) -> Any:
    """helper to read the active section task fields."""
    return next(t for t in p.progress.tasks if t.id == p.section_task)


def test_start_shows_in_flight_item_without_incrementing() -> None:
    """start marks the in flight item on the detail line without incrementing the counter."""
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance, sub_step = p.section_loop(3, "Collect")
    task: Any = _section_task(p)
    assert task.completed == 0
    start("my_table.yaml · abc123de")
    task = _section_task(p)
    assert task.completed == 0
    assert "my_table.yaml" in p.detail_text.plain
    assert "abc123de" in p.detail_text.plain
    assert callable(advance)
    assert callable(sub_step)


def test_advance_ticks_counter_without_changing_detail() -> None:
    """advance ticks the counter without changing the detail line."""
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance, _ = p.section_loop(3, "Collect")
    start("my_table.yaml · abc123de")
    advance()
    task: Any = _section_task(p)
    assert task.completed == 1
    assert "my_table.yaml" in p.detail_text.plain


def test_start_advance_cycle_shows_current_not_previous() -> None:
    """repeated start advance cycles keep detail synced to the current item not the last."""
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance, _ = p.section_loop(3, "Subgraph")
    start("first.yaml · deadbeef")
    advance()
    start("second.yaml · cafebabe")
    task: Any = _section_task(p)
    assert task.completed == 1
    assert "second.yaml" in p.detail_text.plain
    assert "cafebabe" in p.detail_text.plain
    assert "first.yaml" not in p.detail_text.plain


def test_sub_step_appends_phase_to_detail() -> None:
    """sub_step appends a phase tag to the detail line without ticking the counter."""
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, _, sub_step = p.section_loop(3, "Subgraph")
    start("first.yaml · deadbeef")
    sub_step("resolve")
    task: Any = _section_task(p)
    assert task.completed == 0
    assert "→" in p.detail_text.plain
    assert "resolve" in p.detail_text.plain


def test_advance_clears_sub_step_suffix() -> None:
    """advance clears the sub_step suffix so the detail line returns to just the current item."""
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance, sub_step = p.section_loop(3, "Subgraph")
    start("first.yaml · deadbeef")
    sub_step("resolve")
    advance()
    assert "→" not in p.detail_text.plain
    assert "resolve" not in p.detail_text.plain
    assert "first.yaml" in p.detail_text.plain


def test_stage_prints_completion_line_for_previous_stage(capsys: Any) -> None:
    """stage emits a green completion line for the previous stage before advancing."""
    p: PipelineProgress = PipelineProgress(total_stages=2)
    with p:
        p.stage("First")
        p.stage("Second")
    captured: Any = capsys.readouterr()
    assert "Stage 1" in captured.err
    assert "FIRST" in captured.err
