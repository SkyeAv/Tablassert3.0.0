from src.tablassert.utils.io import load_yaml, load_model, build_sections
from src.tablassert.core.tabular import dataframing, initialize_logger
from src.tablassert.core.export import save, savepath, aggregate
from src.tablassert.models.graph import GraphConfig
from src.tablassert.models.table import Section
from multiprocessing import Pool
from typing import Any, Optional
from pathlib import Path
import polars as pl
import typer

app = typer.Typer()


def subgraph(SubSection: Section, graphmodel: dict[str, Any], idx: int) -> None:
    subsectionmodel: dict[str, Any] = SubSection.model_dump()
    posix_filepath: str = subsectionmodel["posix_filepath"]
    sheetname: Optional[str] = subsectionmodel["location"][
        "download_hyperparameters"
    ].get("which_excel_sheet_to_use")
    metadata: dict[str, Any] = graphmodel["metadata"]
    exportpath: Path = savepath(posix_filepath, sheetname, metadata, idx)
    if not exportpath.exists():
        df: pl.DataFrame = dataframing(subsectionmodel, graphmodel)
        save(df, exportpath)
    return None


@app.command()
def build(graphconfig: str) -> None:
    # typer uses docstrings for command descriptions
    """Build a Knowledge Graph with a GraphConfig"""
    graphconfigpath: Path = Path(graphconfig)
    graph_yaml: Any = load_yaml(graphconfigpath)
    Graph: GraphConfig = load_model(graph_yaml, GraphConfig)
    graphmodel: dict[str, Any] = Graph.model_dump()
    sections: list[tuple[Section, dict[str, Any], int]] = build_sections(graphmodel)
    workers: int = graphmodel["hyperparameters"]["number_of_parallel_processes_to_run"]
    with Pool(processes=workers, initializer=initialize_logger) as pool:
        _ = pool.starmap(subgraph, sections)
    aggregate(graphmodel)
    return None


# wrapper for poety entrypoint
def cli() -> None:
    app()
    return None
