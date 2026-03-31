from __future__ import annotations

from contextlib import ExitStack
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lazy_loader as Lazy

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

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

CLI: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)
PROGRESS: Progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
)


def track(task_id: Any, iterable: Any) -> Any:
    for item in iterable:
        yield item
        PROGRESS.advance(task_id)


@CLI.command()
def build_knowledge_graph(
    graph_configuration_file: Path = typer.Argument(..., help="Knowledge Graph Configuration -- See Docs"),
) -> None:
    """Build A KGX Compliant Knowledge Graph From A Graph Configuration File"""
    # TODO: Make MeSH A Node (Micro Version)
    # TODO: Add FullMap Column Context Flag"
    r: object = from_yaml(graph_configuration_file)
    g: Graph = Graph.model_validate(r)

    with PROGRESS:
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
        tcode: list[Tcode] = [
            Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{mkhash(s)}.parquet")})
            for idx, s in track(t3, enumerate(sections, start=1))
        ]
        with ExitStack() as stack:
            conns: list[object] = [
                stack.enter_context(duckdb.connect(g.datassert / "data" / f"{x}.duckdb", read_only=True))
                for x in range(SHARDS)
            ]
            # ? Collect Instructions
            t4: Any = PROGRESS.add_task("Collecting Instructions...", total=n)
            instructions: list[Any] = [x.collect(conns, g.pubmed_db, g.pmc_db) for x in track(t4, tcode)]  # pyright: ignore

            # ? Build Subgraphs
            t5: Any = PROGRESS.add_task("Building Subgraphs...", total=n)
            subgraphs: list[Path] = [
                op if isinstance(op, Path) else compile_subgraph(op) for op in track(t5, instructions)
            ]  # pyright: ignore

        # ? Compile Graph
        t6: Any = PROGRESS.add_task("Compiling Graph...", total=None)
        compile_graph(subgraphs, g.name, g.version)
        PROGRESS.update(t6, total=1, completed=1)

        PROGRESS.add_task("[bold green]Finished!", total=1, completed=1)


@CLI.command()
def verify_table_configuration_syntax(
    table_configuration_file: Path = typer.Argument(..., help="Table Configuration -- See Docs"),
) -> None:
    """Verify The Syntax Of A Declarative Table Configuration File"""
    with PROGRESS:
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
        for s in track(t3, sections):
            Section.model_validate(s)
            PROGRESS.update(t3, total=1, completed=1)

        PROGRESS.add_task("[bold green]Finished!", total=1, completed=1)
