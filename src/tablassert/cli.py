from __future__ import annotations

from contextlib import ExitStack
from importlib.metadata import version as get_version
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cyclopts
import lazy_loader as Lazy

from tablassert.log import logger

if TYPE_CHECKING:
    import duckdb
    import pydantic

    from tablassert.lib import Tcode  # noqa: F401
    from tablassert.models import Graph  # noqa: F401
    from tablassert.progress import PipelineProgress
else:
    duckdb = Lazy.load("duckdb")
    pydantic = Lazy.load("pydantic")

APP: cyclopts.App = cyclopts.App(
    version=f"tablassert {get_version('tablassert')}",
    help="Extract Knowledge Assertions From Tabular Data Into KGX NDJSON",
)


def build_pipeline(graph_configuration_file: Path, progress: "PipelineProgress") -> None:
    # ? Build A Knowledge Graph From A Configuration File
    from tablassert.fullmap import SHARDS
    from tablassert.ingests import from_yaml, to_sections
    from tablassert.lib import Tcode, compile_graph, compile_subgraph
    from tablassert.models import Graph
    from tablassert.progress import flatten_pydantic_error, format_section_oneline
    from tablassert.utils import STORE, mkhash

    # * Load Tables (1/6)
    progress.stage("Loading Tables")
    r: object = from_yaml(graph_configuration_file)
    try:
        g: Graph = Graph.model_validate(r)
    except pydantic.ValidationError as e:
        raise RuntimeError(
            f"02 | FAILED VALIDATION | CONFIG: {graph_configuration_file} | KIND: graph | PYDANTIC: {flatten_pydantic_error(e)}"
        ) from e
    with Pool() as pool:
        raw: list[object] = pool.map(from_yaml, g.tables)

    # * Extract Sections (2/6)
    progress.stage("Extracting Sections")
    with Pool() as pool:
        temp: list[list[dict[str, Any]]] = pool.starmap(to_sections, zip(raw, g.tables))  # pyright: ignore
    sections: list[dict[str, Any]] = list(chain.from_iterable(temp))
    n: int = len(sections)

    # * Build TCode (3/6)
    progress.stage(f"Building TCode | Sections: {n}")
    start, advance = progress.section_loop(n, "TCode")
    tcode: list[Tcode] = []
    for idx, s in enumerate(sections, start=1):
        h: str = mkhash(s)
        start(f"#{idx} | CONFIG: {Path(s['config']).name} | HASH: {h}")
        try:
            tcode.append(
                Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{h}.parquet"), "log": g.log, "qc": g.qc})
            )
        except pydantic.ValidationError as e:
            raise RuntimeError(
                f"02 | FAILED VALIDATION | CONFIG: {graph_configuration_file} | IDX: {idx} | HASH: {h} | PYDANTIC: {flatten_pydantic_error(e)}"
            ) from e
        advance()

    with ExitStack() as stack:
        conns: list[object] = [
            stack.enter_context(duckdb.connect(g.datassert / "data" / f"{x}.duckdb", read_only=True))
            for x in range(SHARDS)
        ]

        # * Collect Instructions (4/6)
        progress.stage(f"Collecting Instructions | Sections: {n}")
        start, advance = progress.section_loop(n, "Collect")
        instructions: list[Any] = []
        for x in tcode:
            start(format_section_oneline(x))
            instructions.append(x.collect(conns, g.pubmed_db, g.pmc_db))  # pyright: ignore
            advance()

        # * Build Subgraphs (5/6)
        progress.stage(f"Building Subgraphs | Sections: {n}")
        start, advance = progress.section_loop(n, "Subgraph")
        subgraphs: list[Path] = []
        for x, op in zip(tcode, instructions):
            start(format_section_oneline(x))
            subgraphs.append(op if isinstance(op, Path) else compile_subgraph(op))
            advance()

    # * Compile Graph (6/6)
    progress.stage(f"Compiling Graph | Sections: {n}")
    start, advance = progress.section_loop(1, "Graph")
    start(f"NAME: {g.name} | VERSION: {g.version}")
    compile_graph(subgraphs, g.name, g.version)
    advance()

    logger.info(f"BUILD DONE | SECTIONS: {n} | NAME: {g.name} | VERSION: {g.version}")


def validate_pipeline(table_configuration_file: Path, progress: "PipelineProgress") -> None:
    # ? Validate Section Syntax From A Configuration File
    from tablassert.ingests import from_yaml, to_sections
    from tablassert.lib import Tcode
    from tablassert.progress import flatten_pydantic_error
    from tablassert.utils import STORE, mkhash

    # * Load Tables (1/3)
    progress.stage("Loading Tables")
    r: object = from_yaml(table_configuration_file)

    # * Extract Sections (2/3)
    progress.stage("Extracting Sections")
    sections: list[dict[str, Any]] = to_sections(r, table_configuration_file)  # pyright: ignore
    n: int = len(sections)

    # * Validate Section Syntax (3/3)
    progress.stage(f"Validating Section Syntax | Sections: {n}")
    start, advance = progress.section_loop(n, "Validate")
    for idx, s in enumerate(sections, start=1):
        h: str = mkhash(s)
        start(f"#{idx} | HASH: {h}")
        try:
            Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{h}.parquet")})
        except pydantic.ValidationError as e:
            raise RuntimeError(
                f"02 | FAILED VALIDATION | CONFIG: {table_configuration_file} | IDX: {idx} | HASH: {h} | PYDANTIC: {flatten_pydantic_error(e)}"
            ) from e
        advance()

    logger.info(f"VALIDATE DONE | SECTIONS: {n} | CONFIG: {table_configuration_file.name}")


def run(stages: int, fn: Any, arg: Path) -> None:
    from tablassert.log import LOG_FORMAT, logger
    from tablassert.progress import PipelineProgress

    with PipelineProgress(total_stages=stages) as progress:
        sink_id: int = logger.add(progress.log_sink, level="INFO", format=LOG_FORMAT)
        try:
            fn(arg, progress)
        finally:
            logger.remove(sink_id)


@APP.command
def build(graph_configuration_file: Path) -> None:
    """Build a knowledge graph from a YAML configuration file."""
    run(6, build_pipeline, graph_configuration_file)


@APP.command
def validate(table_configuration_file: Path) -> None:
    """Validate section syntax from a YAML configuration file."""
    run(3, validate_pipeline, table_configuration_file)
