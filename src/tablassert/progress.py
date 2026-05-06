from __future__ import annotations

from contextlib import AbstractContextManager
from importlib.metadata import version as get_version
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Optional

from rich import box
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from pydantic import ValidationError

    from tablassert.lib import Tcode


def format_section_oneline(x: "Tcode") -> str:
    from tablassert.models import Excel

    if isinstance(x.source, Excel):
        source_detail: str = f"EXCEL({x.source.sheet})"
    else:
        source_detail = f"TEXT({(x.source.delimiter or ',')!r})"  # pyright: ignore
    return (
        f"#{x.number} | HASH: {x.store.stem} | SOURCE: {source_detail} "
        f"| CONFIG: {x.config.name} | STATUS: {x.status}"
    )


def flatten_pydantic_error(e: "ValidationError") -> str:
    parts: list[str] = []
    for err in e.errors():
        loc: str = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        msg: str = str(err.get("msg", "")).replace("\n", " ").replace("|", "/").strip()
        kind: str = str(err.get("type", ""))
        parts.append(f"LOC: {loc} MSG: {msg} TYPE: {kind}")
    return " ; ".join(parts)


class PipelineProgress(AbstractContextManager["PipelineProgress"]):
    def __init__(self: "PipelineProgress", total_stages: int) -> None:
        self.total_stages: int = total_stages
        self.console: Console = Console(stderr=True)
        self.progress: Progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        self.panel: Panel = Panel(
            Padding(self.progress, (1, 2)),
            title=f"[bold]TABLASSERT[/bold] [dim]v{get_version('tablassert')}[/dim]",
            title_align="left",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
        )
        self.live: Live = Live(self.panel, console=self.console, refresh_per_second=10, transient=False)
        self.stage_task: Optional[TaskID] = None
        self.section_task: Optional[TaskID] = None
        self.stage_step: int = 0

    def __enter__(self: "PipelineProgress") -> "PipelineProgress":
        self.console.line(1)
        self.live.start()
        self.stage_task = self.progress.add_task(
            description=f"STAGE 0/{self.total_stages} | STARTING", total=self.total_stages
        )
        return self

    def __exit__(
        self: "PipelineProgress",
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.live.stop()
        self.console.line(1)

    def stage(self: "PipelineProgress", name: str) -> None:
        self.stage_step += 1
        assert self.stage_task is not None
        self.progress.update(
            self.stage_task,
            description=f"STAGE {self.stage_step}/{self.total_stages} | {name.upper()}",
            completed=self.stage_step,
        )
        self.end_section_task()

    def section_loop(self: "PipelineProgress", total: int, label: str) -> Callable[[str], None]:
        self.end_section_task()
        self.section_task = self.progress.add_task(description=f"{label.upper()} | WORKING", total=total)

        def advance(info: str) -> None:
            assert self.section_task is not None
            self.progress.update(self.section_task, description=f"{label.upper()} | {info}", advance=1)

        return advance

    def end_section_task(self: "PipelineProgress") -> None:
        if self.section_task is not None:
            self.progress.update(self.section_task, visible=False)
            self.section_task = None

    def log_sink(self: "PipelineProgress", message: str) -> None:
        self.console.print(message, end="", highlight=False, markup=False)
