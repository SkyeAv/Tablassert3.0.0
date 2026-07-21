from __future__ import annotations

from typing import Any

from tablassert.progress import PipelineProgress


# ? Helper To Read The Active Section Task Fields
def _section_task(p: PipelineProgress) -> Any:
    return next(t for t in p.progress.tasks if t.id == p.section_task)


# ? start Marks The In Flight Item On The Detail Line Without Incrementing The Counter
def test_start_shows_in_flight_item_without_incrementing() -> None:
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


# ? advance Ticks The Counter Without Changing The Detail Line
def test_advance_ticks_counter_without_changing_detail() -> None:
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance, _ = p.section_loop(3, "Collect")
    start("my_table.yaml · abc123de")
    advance()
    task: Any = _section_task(p)
    assert task.completed == 1
    assert "my_table.yaml" in p.detail_text.plain


# ? Repeated start advance Cycles Keep Detail Synced To The Current Item Not The Last
def test_start_advance_cycle_shows_current_not_previous() -> None:
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


# ? sub_step Appends A Phase Tag To The Detail Line Without Ticking The Counter
def test_sub_step_appends_phase_to_detail() -> None:
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, _, sub_step = p.section_loop(3, "Subgraph")
    start("first.yaml · deadbeef")
    sub_step("resolve")
    task: Any = _section_task(p)
    assert task.completed == 0
    assert "→" in p.detail_text.plain
    assert "resolve" in p.detail_text.plain


# ? advance Clears The sub_step Suffix So The Detail Line Returns To Just The Current Item
def test_advance_clears_sub_step_suffix() -> None:
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance, sub_step = p.section_loop(3, "Subgraph")
    start("first.yaml · deadbeef")
    sub_step("resolve")
    advance()
    assert "→" not in p.detail_text.plain
    assert "resolve" not in p.detail_text.plain
    assert "first.yaml" in p.detail_text.plain


# ? stage Emits A Green Completion Line For The Previous Stage Before Advancing
def test_stage_prints_completion_line_for_previous_stage(capsys: Any) -> None:
    p: PipelineProgress = PipelineProgress(total_stages=2)
    with p:
        p.stage("First")
        p.stage("Second")
    captured: Any = capsys.readouterr()
    assert "Stage 1" in captured.err
    assert "FIRST" in captured.err
