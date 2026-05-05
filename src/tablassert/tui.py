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
        label: Optional[Label] = self.query_one(Label)
        if label is not None:
            label.update(stage)

    def watch_total(self: StageProgress, total: float) -> None:
        bar: Optional[ProgressBar] = self.query_one(ProgressBar)
        if bar is not None:
            bar.total = total

    def watch_progress(self: StageProgress, progress: float) -> None:
        bar: Optional[ProgressBar] = self.query_one(ProgressBar)
        if bar is not None:
            bar.update(progress=progress)


class SectionDetails(Static):
    section_info: reactive[str] = reactive("Waiting...")

    def compose(self: SectionDetails) -> ComposeResult:
        yield Label("Section Details", classes="title")
        yield Static(self.section_info, id="section-content")

    def watch_section_info(self: SectionDetails, info: str) -> None:
        content: Optional[Static] = self.query_one("#section-content", Static)
        if content is not None:
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
        sp: Optional[StageProgress] = self.query_one("#stage-progress", StageProgress)
        if sp is not None:
            sp.stage = stage

    def watch_progress(self: TablassertApp, progress: float) -> None:
        sp: Optional[StageProgress] = self.query_one("#stage-progress", StageProgress)
        if sp is not None:
            sp.progress = progress

    def watch_total(self: TablassertApp, total: float) -> None:
        sp: Optional[StageProgress] = self.query_one("#stage-progress", StageProgress)
        if sp is not None:
            sp.total = total

    def watch_section_info(self: TablassertApp, info: str) -> None:
        sd: Optional[SectionDetails] = self.query_one("#section-details", SectionDetails)
        if sd is not None:
            sd.section_info = info

    def watch_stats(self: TablassertApp, stats: str) -> None:
        sf: Optional[StatsFooter] = self.query_one("#stats-footer", StatsFooter)
        if sf is not None:
            sf.stats = stats

    def write_log(self: TablassertApp, message: str) -> None:
        lp: Optional[LogPanel] = self.query_one("#log-panel", LogPanel)
        if lp is not None:
            lp.write(message)
