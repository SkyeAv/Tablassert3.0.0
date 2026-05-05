from __future__ import annotations

from contextlib import ExitStack
from importlib.metadata import version as get_version
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lazy_loader as Lazy
from pydantic import ValidationError

from tablassert.fullmap import SHARDS
from tablassert.ingests import from_yaml, to_sections
from tablassert.lib import Tcode, compile_graph, compile_subgraph
from tablassert.models import Graph, Section
from tablassert.utils import STORE, mkhash

if TYPE_CHECKING:
    import duckdb
    import typer
else:
    duckdb = Lazy.load("duckdb")
    typer = Lazy.load("typer")

from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

CLI: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)
PROGRESS: Progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
)


@CLI.command()
def version() -> None:
    """Print The Tablassert Version"""
    v: str = get_version("tablassert")
    typer.echo(f"tablassert {v}")


def track(task_id: Any, iterable: Any) -> Any:
    for item in iterable:
        yield item
        PROGRESS.advance(task_id)


class InFlight:
    # ? Tracks Sections Currently Being Processed For The Live Panel
    def __init__(self: InFlight) -> None:
        self.items: dict[int, tuple[str, str]] = {}

    def add(self: InFlight, number: int, config: str, section_hash: str) -> None:
        self.items[number] = (config, section_hash)

    def remove(self: InFlight, number: int) -> None:
        self.items.pop(number, None)

    def render(self: InFlight) -> Table:
        t: Table = Table(title="In-Flight Sections", expand=True)
        t.add_column("IDX", justify="right")
        t.add_column("CONFIG")
        t.add_column("HASH")
        for number, (config, h) in sorted(self.items.items()):
            t.add_row(str(number), config, h)
        return t


@CLI.command()
def build_knowledge_graph(
    graph_configuration_file: Path = typer.Argument(..., help="Knowledge Graph Configuration -- See Docs"),
) -> None:
    """Build A KGX Compliant Knowledge Graph From A Graph Configuration File"""
    # TODO: Make MeSH A Node (Micro Version)
    # TODO: Add FullMap Column Context Flag"
    r: object = from_yaml(graph_configuration_file)
    try:
        g: Graph = Graph.model_validate(r)
    except ValidationError as e:
        raise RuntimeError(
            f"02 | FAILED VALIDATION | CONFIG: {graph_configuration_file} | KIND: graph | PYDANTIC: {e}"
        ) from e

    inflight: InFlight = InFlight()

    def render() -> Group:
        return Group(PROGRESS, inflight.render())

    with Live(render(), refresh_per_second=4) as live:
        # ? Load Tables
        t1: Any = PROGRESS.add_task("Loading Tables...", total=None)
        with Pool() as pool:
            raw: list[object] = pool.map(from_yaml, g.tables)
        PROGRESS.update(t1, total=1, completed=1)

        # ? Extract Sections
        t2: Any = PROGRESS.add_task("Extracting Sections...", total=None)
        with Pool() as pool:
            temp: list[list[dict[str, Any]]] = pool.starmap(to_sections, zip(raw, g.tables))  # pyright: ignore
        PROGRESS.update(t2, total=1, completed=1)
        sections: list[dict[str, Any]] = list(chain.from_iterable(temp))
        n: int = len(sections)

        # ? Build Tcodes
        t3: Any = PROGRESS.add_task("Building TCode...", total=n)
        tcode: list[Tcode] = []
        for idx, s in track(t3, enumerate(sections, start=1)):
            try:
                tcode.append(Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{mkhash(s)}.parquet")}))
            except ValidationError as e:
                raise RuntimeError(
                    f"02 | FAILED VALIDATION | CONFIG: {graph_configuration_file} | IDX: {idx} | HASH: {mkhash(s)} | PYDANTIC: {e}"
                ) from e

        with ExitStack() as stack:
            conns: list[object] = [
                stack.enter_context(duckdb.connect(g.datassert / "data" / f"{x}.duckdb", read_only=True))
                for x in range(SHARDS)
            ]
            # ? Collect Instructions
            t4: Any = PROGRESS.add_task("Collecting Instructions...", total=n)
            instructions: list[Any] = []
            for x in tcode:
                inflight.add(x.number, x.config.name, x.store.stem)
                live.update(render())
                instructions.append(x.collect(conns, g.pubmed_db, g.pmc_db))  # pyright: ignore
                inflight.remove(x.number)
                PROGRESS.advance(t4)
                live.update(render())

            # ? Build Subgraphs
            t5: Any = PROGRESS.add_task("Building Subgraphs...", total=n)
            subgraphs: list[Path] = []
            for x, op in zip(tcode, instructions):
                inflight.add(x.number, x.config.name, x.store.stem)
                live.update(render())
                subgraphs.append(op if isinstance(op, Path) else compile_subgraph(op))
                inflight.remove(x.number)
                PROGRESS.advance(t5)
                live.update(render())

        # ? Compile Graph
        t6: Any = PROGRESS.add_task("Compiling Graph...", total=None)
        compile_graph(subgraphs, g.name, g.version)
        PROGRESS.update(t6, total=1, completed=1)

        PROGRESS.add_task("[bold green]Finished!", total=1, completed=1)
        live.update(render())


@CLI.command()
def verify_table_configuration_syntax(
    table_configuration_file: Path = typer.Argument(..., help="Table Configuration -- See Docs"),
) -> None:
    """Verify The Syntax Of A Declarative Table Configuration File"""
    inflight: InFlight = InFlight()

    def render() -> Group:
        return Group(PROGRESS, inflight.render())

    with Live(render(), refresh_per_second=4) as live:
        # ? Load Tables
        t1: Any = PROGRESS.add_task("Loading Tables...", total=None)
        r: object = from_yaml(table_configuration_file)
        PROGRESS.update(t1, total=1, completed=1)

        # ? Extract Sections
        t2: Any = PROGRESS.add_task("Extracting Sections...", total=None)
        sections: list[dict[str, Any]] = to_sections(r)  # pyright: ignore
        n: int = len(sections)
        PROGRESS.update(t2, total=1, completed=1)

        # ? Validating Section Syntax
        t3: Any = PROGRESS.add_task("Validating Section Syntax...", total=n)
        for idx, s in enumerate(sections, start=1):
            h: str = mkhash(s)
            inflight.add(idx, table_configuration_file.name, h)
            live.update(render())
            try:
                Section.model_validate(s)
            except ValidationError as e:
                raise RuntimeError(
                    f"02 | FAILED VALIDATION | CONFIG: {table_configuration_file} | IDX: {idx} | HASH: {h} | PYDANTIC: {e}"
                ) from e
            inflight.remove(idx)
            PROGRESS.advance(t3)
            live.update(render())

        PROGRESS.add_task("[bold green]Finished!", total=1, completed=1)
        live.update(render())
