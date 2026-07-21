from __future__ import annotations

from contextlib import AbstractContextManager
from importlib.metadata import version as get_version
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.text import Text

if TYPE_CHECKING:
    from pydantic import ValidationError

    from tablassert.lib import Tcode


def format_section_compact(x: "Tcode") -> str:
    # ? One Identifier Per Row: Config Stem Plus 8 Char Hash Stem
    return f"{Path(x.config.name).stem} · {x.store.stem[:8]}"


def flatten_pydantic_error(e: "ValidationError") -> str:
    parts: list[str] = []
    for err in e.errors():
        loc: str = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        msg: str = str(err.get("msg", "")).replace("\n", " ").replace("|", "/").strip()
        kind: str = str(err.get("type", ""))
        parts.append(f"LOC: {loc} MSG: {msg} TYPE: {kind}")
    return " ; ".join(parts)


def _truncate(s: str, max_width: int) -> str:
    # ? Truncates To max_width With A Trailing Ellipsis If Too Long
    if len(s) <= max_width:
        return s
    if max_width <= 1:
        return "…"
    return s[: max_width - 1] + "…"


class PipelineProgress(AbstractContextManager["PipelineProgress"]):
    def __init__(self: "PipelineProgress", total_stages: int) -> None:
        self.total_stages: int = total_stages
        self.console: Console = Console(stderr=True)
        # ? Middle Row: Label + Bar + Count + Elapsed + ETA. No Description, No Spinner.
        self.progress: Progress = Progress(
            TextColumn("[bold cyan]{task.fields[label]}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]·[/dim]"),
            TimeElapsedColumn(),
            TextColumn("[dim]·[/dim]"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            expand=True,
        )
        # ? Top Row: Stage Header
        self.stage_text: Text = Text("")
        # ? Bottom Row: Indented Item Detail With Optional Phase Tag
        self.detail_text: Text = Text("")
        self._group: Group = Group(self.stage_text, Padding(self.progress, (0, 0)), self.detail_text)
        self.live: Live = Live(self._group, console=self.console, refresh_per_second=10, transient=False)
        self.stage_task: Optional[TaskID] = None
        self.section_task: Optional[TaskID] = None
        self.stage_step: int = 0
        self._stage_name: str = ""
        self._stage_started_at: float = monotonic()
        self._current_detail: str = ""
        self._current_phase: str = ""

    def __enter__(self: "PipelineProgress") -> "PipelineProgress":
        self.console.line(1)
        self.live.start()
        self._render_stage(f"Stage 0 of {self.total_stages}  ·  STARTING")
        return self

    def __exit__(
        self: "PipelineProgress", exc_type: Optional[type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        # ? Print A Final Completion Line For The Last Stage Before Tearing Down
        if self.stage_step >= 1:
            self._print_completion()
        self.live.stop()
        self.console.line(1)

    def stage(self: "PipelineProgress", name: str) -> None:
        # ? Emit A Completion Line For The Previous Stage Before Advancing
        if self.stage_step >= 1:
            self._print_completion()
        self.stage_step += 1
        self._stage_name = name
        self._stage_started_at = monotonic()
        self.end_section_task()
        self._clear_detail()
        self._render_stage(f"Stage {self.stage_step} of {self.total_stages}  ·  {name.upper()}")

    def section_loop(self: "PipelineProgress", total: int, label: str) -> tuple[Callable[[str], None], Callable[[], None], Callable[[str], None]]:
        # ? Returns (start, advance, sub_step) callbacks. Detail lives on its own row, NOT in the bar description.
        self.end_section_task()
        self.section_task = self.progress.add_task(description="", total=total, label=label.upper())
        self._clear_detail()

        def start(detail: str) -> None:
            assert self.section_task is not None
            self._current_detail = detail
            self._current_phase = ""
            self._render_detail()

        def sub_step(phase: str) -> None:
            assert self.section_task is not None
            self._current_phase = phase
            self._render_detail()

        def advance() -> None:
            assert self.section_task is not None
            self._current_phase = ""
            self.progress.update(self.section_task, advance=1)
            self._render_detail()

        return start, advance, sub_step

    def end_section_task(self: "PipelineProgress") -> None:
        if self.section_task is not None:
            self.progress.update(self.section_task, visible=False)
            self.section_task = None

    def log_sink(self: "PipelineProgress", message: str) -> None:
        self.console.print(message, end="", highlight=False, markup=False)

    def _render_stage(self: "PipelineProgress", text: str) -> None:
        # ? Stage Header: Bold Stage Count + Dimmed Version, Single Line, No Metadata
        self.stage_text.plain = ""
        self.stage_text.append(text, style="bold")
        self.stage_text.append(f"   tablassert v{get_version('tablassert')}", style="dim")

    def _render_detail(self: "PipelineProgress") -> None:
        # ? Compose Indented Item Detail With Optional Phase Suffix, Truncated To Console Width
        body: str = f"  ↳ {self._current_detail}" if self._current_detail else "  ↳ …"
        if self._current_phase:
            body = f"{body}  →  {self._current_phase}"
        max_width: int = max(20, self.console.width - 2)
        self.detail_text.plain = ""
        self.detail_text.append(_truncate(body, max_width), style="dim")

    def _clear_detail(self: "PipelineProgress") -> None:
        self._current_detail = ""
        self._current_phase = ""
        self.detail_text.plain = ""

    def _print_completion(self: "PipelineProgress") -> None:
        # ? Print Above The Live Region: ✓ Stage N · NAME · 0:00:42
        elapsed: float = monotonic() - self._stage_started_at
        mins: int = int(elapsed) // 60
        secs: float = elapsed - mins * 60
        line: str = f"✓ Stage {self.stage_step} · {self._stage_name.upper()} · {mins}:{secs:05.2f}"
        self.console.print(line, style="green", highlight=False, markup=False)
