from src.utils.io import load_yaml, load_model, build_sections
from src.core.tabular import dataframing
from src.models.graph import GraphConfig
from src.models.table import Section
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import polars as pl
import typer

app = typer.Typer()


def subgraph(SubSection: Section, Graph: GraphConfig, index: int) -> None:
    subsectionmodel: dict[str, Any] = SubSection.model_dump()
    graphmodel: dict[str, Any] = Graph.model_dump()
    df: pl.DataFrame = dataframing(subsectionmodel, graphmodel)
    df.write_csv("TEST.csv", separator="\t")
    return None


@app.command()
def build(graphconfig: str) -> None:
    # typer uses docstrings for command descriptions
    """Build a Knowledge Graph with a GraphConfig"""
    graphconfigpath: Path = Path(graphconfig)
    graph_yaml: Any = load_yaml(graphconfigpath)
    Graph: GraphConfig = load_model(graph_yaml, GraphConfig)
    sections: list[tuple[Section, GraphConfig, int]] = build_sections(Graph)
    workers: int = Graph.hyperparameters.number_of_parallel_processes_to_run
    with Pool(processes=workers) as pool:
        _ = pool.starmap(subgraph, sections)
    return None


# wrapper for poety entrypoint
def cli() -> None:
    app()
    return None
