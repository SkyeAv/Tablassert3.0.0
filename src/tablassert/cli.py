from __future__ import annotations

from contextlib import ExitStack
from importlib.metadata import version as get_version
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import cyclopts
import lazy_loader as Lazy
from loguru import logger
from pydantic import ValidationError

from tablassert.fullmap import SHARDS
from tablassert.ingests import from_yaml, to_sections
from tablassert.lib import Tcode, compile_graph, compile_subgraph
from tablassert.models import Excel, Graph
from tablassert.tui import TablassertApp
from tablassert.utils import STORE, mkhash

if TYPE_CHECKING:
    import duckdb
else:
    duckdb = Lazy.load("duckdb")

APP: cyclopts.App = cyclopts.App(
    version=f"tablassert {get_version('tablassert')}",
    help="Extract Knowledge Assertions From Tabular Data Into KGX NDJSON",
)

APP_REF: Optional[TablassertApp] = None


def format_section(x: Tcode) -> str:
    source_detail: str = x.source.sheet if isinstance(x.source, Excel) else (x.source.delimiter or ",")  # pyright: ignore
    return (
        f"[bold]#{x.number}[/bold]  {x.config.name}\n"
        f"  Hash:      {x.store.stem}\n"
        f"  Source:    {x.source.kind} ({source_detail})\n"
        f"  Predicate: {x.statement.predicate}\n"
        f"  Repo:      {x.provenance.repo}\n"
        f"  Status:    {x.status}"
    )


def log_sink(message: str) -> None:
    app: Optional[TablassertApp] = APP_REF
    if app is not None:
        app.call_from_thread(app.write_log, str(message))


def build_pipeline(graph_configuration_file: Path, app: TablassertApp) -> None:
    # ? Build A Knowledge Graph From A Configuration File
    # * Load Tables
    app.call_from_thread(app.watch_stats, "Stage: Loading Tables")
    r: object = from_yaml(graph_configuration_file)
    try:
        g: Graph = Graph.model_validate(r)
    except ValidationError as e:
        raise RuntimeError(
            f"02 | FAILED VALIDATION | CONFIG: {graph_configuration_file} | KIND: graph | PYDANTIC: {e}"
        ) from e

    with Pool() as pool:
        raw: list[object] = pool.map(from_yaml, g.tables)
    app.call_from_thread(app.watch_stats, "Stage: Loading Tables | Sections: — | 1/6")

    # * Extract Sections
    app.call_from_thread(app.watch_stats, "Stage: Extracting Sections")
    with Pool() as pool:
        temp: list[list[dict[str, Any]]] = pool.starmap(to_sections, zip(raw, g.tables))  # pyright: ignore
    sections: list[dict[str, Any]] = list(chain.from_iterable(temp))
    n: int = len(sections)
    app.call_from_thread(app.watch_stats, f"Stage: Extracting Sections | Sections: {n} | 2/6")

    # * Build TCode
    app.call_from_thread(app.watch_stage, "Building TCode")
    app.call_from_thread(app.watch_total, float(n))
    app.call_from_thread(app.watch_progress, 0.0)
    app.call_from_thread(app.watch_stats, f"Stage: Building TCode | Sections: {n} | 3/6")
    tcode: list[Tcode] = []
    for idx, s in enumerate(sections, start=1):
        try:
            tcode.append(Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{mkhash(s)}.parquet")}))
        except ValidationError as e:
            raise RuntimeError(
                f"02 | FAILED VALIDATION | CONFIG: {graph_configuration_file} | IDX: {idx} | HASH: {mkhash(s)} | PYDANTIC: {e}"
            ) from e
        app.call_from_thread(app.watch_progress, float(idx))
        app.call_from_thread(app.watch_section_info, format_section(tcode[-1]))

    with ExitStack() as stack:
        conns: list[object] = [
            stack.enter_context(duckdb.connect(g.datassert / "data" / f"{x}.duckdb", read_only=True))
            for x in range(SHARDS)
        ]

        # * Collect Instructions
        app.call_from_thread(app.watch_stage, "Collecting Instructions")
        app.call_from_thread(app.watch_total, float(n))
        app.call_from_thread(app.watch_progress, 0.0)
        app.call_from_thread(app.watch_stats, f"Stage: Collecting Instructions | Sections: {n} | 4/6")
        instructions: list[Any] = []
        for x in tcode:
            app.call_from_thread(app.watch_section_info, format_section(x))
            instructions.append(x.collect(conns, g.pubmed_db, g.pmc_db))  # pyright: ignore
            app.call_from_thread(app.watch_progress, float(len(instructions)))

        # * Build Subgraphs
        app.call_from_thread(app.watch_stage, "Building Subgraphs")
        app.call_from_thread(app.watch_total, float(n))
        app.call_from_thread(app.watch_progress, 0.0)
        app.call_from_thread(app.watch_stats, f"Stage: Building Subgraphs | Sections: {n} | 5/6")
        subgraphs: list[Path] = []
        for x, op in zip(tcode, instructions):
            app.call_from_thread(app.watch_section_info, format_section(x))
            subgraphs.append(op if isinstance(op, Path) else compile_subgraph(op))
            app.call_from_thread(app.watch_progress, float(len(subgraphs)))

    # * Compile Graph
    app.call_from_thread(app.watch_stage, "Compiling Graph")
    app.call_from_thread(app.watch_total, 1.0)
    app.call_from_thread(app.watch_progress, 0.0)
    app.call_from_thread(app.watch_stats, f"Stage: Compiling Graph | Sections: {n} | 6/6")
    compile_graph(subgraphs, g.name, g.version)
    app.call_from_thread(app.watch_progress, 1.0)

    app.call_from_thread(app.watch_stage, "Finished")
    app.call_from_thread(
        app.watch_section_info, f"[bold green]Built {n} sections into graph '{g.name}' v{g.version}[/bold green]"
    )
    app.call_from_thread(app.watch_stats, f"Done | Sections: {n} | Graph: {g.name}")


def validate_pipeline(table_configuration_file: Path, app: TablassertApp) -> None:
    # ? Validate Section Syntax From A Configuration File
    # * Load Tables
    app.call_from_thread(app.watch_stats, "Stage: Loading Tables")
    r: object = from_yaml(table_configuration_file)
    app.call_from_thread(app.watch_stats, "Stage: Loading Tables | Sections: — | 1/3")

    # * Extract Sections
    app.call_from_thread(app.watch_stats, "Stage: Extracting Sections")
    sections: list[dict[str, Any]] = to_sections(r, table_configuration_file)  # pyright: ignore
    n: int = len(sections)
    app.call_from_thread(app.watch_stats, f"Stage: Extracting Sections | Sections: {n} | 2/3")

    # * Validate Section Syntax
    app.call_from_thread(app.watch_stage, "Validating Section Syntax")
    app.call_from_thread(app.watch_total, float(n))
    app.call_from_thread(app.watch_progress, 0.0)
    app.call_from_thread(app.watch_stats, f"Stage: Validating Syntax | Sections: {n} | 3/3")
    for idx, s in enumerate(sections, start=1):
        h: str = mkhash(s)
        app.call_from_thread(
            app.watch_section_info, f"[bold]#{idx}[/bold]  {table_configuration_file.name}\n  Hash: {h}"
        )
        try:
            Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{h}.parquet")})
        except ValidationError as e:
            raise RuntimeError(
                f"02 | FAILED VALIDATION | CONFIG: {table_configuration_file} | IDX: {idx} | HASH: {h} | PYDANTIC: {e}"
            ) from e
        app.call_from_thread(app.watch_progress, float(idx))

    app.call_from_thread(app.watch_stage, "Finished")
    app.call_from_thread(app.watch_section_info, f"[bold green]Validated {n} sections[/bold green]")
    app.call_from_thread(app.watch_stats, f"Done | Sections: {n} | All Valid")


@APP.command
def build(graph_configuration_file: Path) -> None:
    """Build a knowledge graph from a YAML configuration file."""
    global APP_REF

    TUI: TablassertApp = TablassertApp()
    APP_REF = TUI

    def worker() -> None:
        build_pipeline(graph_configuration_file, TUI)

    TUI.worker_func = worker
    sink_id: int = logger.add(log_sink, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    try:
        TUI.run()
    finally:
        logger.remove(sink_id)
        APP_REF = None


@APP.command
def validate(table_configuration_file: Path) -> None:
    """Validate section syntax from a YAML configuration file."""
    global APP_REF

    TUI: TablassertApp = TablassertApp()
    APP_REF = TUI

    def worker() -> None:
        validate_pipeline(table_configuration_file, TUI)

    TUI.worker_func = worker
    sink_id: int = logger.add(log_sink, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    try:
        TUI.run()
    finally:
        logger.remove(sink_id)
        APP_REF = None
