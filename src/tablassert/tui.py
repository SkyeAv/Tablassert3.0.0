from __future__ import annotations

from typing import Any, Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Header, Label, ProgressBar, RichLog, Static


class StageProgress(Static):
    stage: reactive[str] = reactive("")
    progress: reactive[float] = reactive(0.0)
    total: reactive[float] = reactive(0.0)

    def __init__(self: StageProgress, stage: str = "", total: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stage = stage
        self.total = total

    def compose(self: StageProgress) -> ComposeResult:
        yield Label(self.stage)
        yield ProgressBar(total=self.total, show_eta=False)

    def watch_stage(self: StageProgress, stage: str) -> None:
        if not self.is_mounted:
            return
        label: Label = self.query_one(Label)
        label.update(stage)

    def watch_total(self: StageProgress, total: float) -> None:
        if not self.is_mounted:
            return
        bar: ProgressBar = self.query_one(ProgressBar)
        bar.total = total

    def watch_progress(self: StageProgress, progress: float) -> None:
        if not self.is_mounted:
            return
        bar: ProgressBar = self.query_one(ProgressBar)
        bar.update(progress=progress)


class SectionDetails(Static):
    section_info: reactive[str] = reactive("Waiting...")

    def compose(self: SectionDetails) -> ComposeResult:
        yield Label("Section Details", classes="title")
        yield Static(self.section_info, id="section-content")

    def watch_section_info(self: SectionDetails, info: str) -> None:
        if not self.is_mounted:
            return
        content: Static = self.query_one("#section-content", Static)
        content.update(info)


class LogPanel(RichLog):
    pass


class StatsFooter(Static):
    stats: reactive[str] = reactive("")

    def render(self: StatsFooter) -> str:
        return self.stats


class TablassertApp(App[None]):
    CSS_PATH = "tui.tcss"

    TITLE = "Tablassert"

    stage: reactive[str] = reactive("")
    progress: reactive[float] = reactive(0.0)
    total: reactive[float] = reactive(0.0)
    section_info: reactive[str] = reactive("")
    stats: reactive[str] = reactive("")

    def __init__(self: TablassertApp, worker_func: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.worker_func = worker_func

    def on_mount(self: TablassertApp) -> None:
        if self.worker_func is not None:
            self.run_worker(self.worker_func, thread=True)

    def compose(self: TablassertApp) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-body"):
            with VerticalScroll(id="left-column"):
                yield StageProgress(id="stage-progress")
                yield SectionDetails(id="section-details")
            with VerticalScroll(id="right-column"):
                yield LogPanel(id="log-panel", highlight=True, markup=True)
        yield StatsFooter(id="stats-footer")

    def watch_stage(self: TablassertApp, stage: str) -> None:
        if not self.is_mounted:
            return
        sp: StageProgress = self.query_one("#stage-progress", StageProgress)
        sp.stage = stage

    def watch_progress(self: TablassertApp, progress: float) -> None:
        if not self.is_mounted:
            return
        sp: StageProgress = self.query_one("#stage-progress", StageProgress)
        sp.progress = progress

    def watch_total(self: TablassertApp, total: float) -> None:
        if not self.is_mounted:
            return
        sp: StageProgress = self.query_one("#stage-progress", StageProgress)
        sp.total = total

    def watch_section_info(self: TablassertApp, info: str) -> None:
        if not self.is_mounted:
            return
        sd: SectionDetails = self.query_one("#section-details", SectionDetails)
        sd.section_info = info

    def watch_stats(self: TablassertApp, stats: str) -> None:
        if not self.is_mounted:
            return
        sf: StatsFooter = self.query_one("#stats-footer", StatsFooter)
        sf.stats = stats

    def write_log(self: TablassertApp, message: str) -> None:
        if not self.is_mounted:
            return
        lp: LogPanel = self.query_one("#log-panel", LogPanel)
        lp.write(message)
