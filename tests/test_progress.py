from __future__ import annotations

from typing import Any

from tablassert.progress import PipelineProgress


# ? Helper To Read The Active Section Task Fields
def _section_task(p: PipelineProgress) -> Any:
    return next(t for t in p.progress.tasks if t.id == p.section_task)


# ? start Marks The In Flight Item In The Description Without Ticking The Counter
def test_start_shows_in_flight_item_without_incrementing() -> None:
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance = p.section_loop(3, "Collect")
    task: Any = _section_task(p)
    assert task.completed == 0
    start("CONFIG: my_table.yaml")
    task = _section_task(p)
    assert "my_table.yaml" in task.description
    assert task.completed == 0
    assert callable(advance)


# ? advance Ticks The Counter Without Changing The In Flight Description
def test_advance_ticks_counter_without_changing_description() -> None:
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance = p.section_loop(3, "Collect")
    start("CONFIG: my_table.yaml")
    advance()
    task: Any = _section_task(p)
    assert task.completed == 1
    assert "my_table.yaml" in task.description
    assert callable(start)


# ? Repeated start advance Cycles Keep Description Synced To The Current Item Not The Last
def test_start_advance_cycle_shows_current_not_previous() -> None:
    p: PipelineProgress = PipelineProgress(total_stages=1)
    start, advance = p.section_loop(3, "Subgraph")
    start("CONFIG: first.yaml")
    advance()
    start("CONFIG: second.yaml")
    task: Any = _section_task(p)
    assert task.completed == 1
    assert "second.yaml" in task.description
    assert "first.yaml" not in task.description
