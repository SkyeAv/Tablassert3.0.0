from tablassert.src.tablassert.utils.io import load_yaml, load_model, get_sections, download_from_link
from tablassert.src.tablassert.models.graph_config import GraphConfig
from tablassert.src.tablassert.models.table_config import Section
from pydantic import FilePath, HttpUrl
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import asyncio

def build_subgraphs(Table: Section, index: int) -> None:
    download_link: HttpUrl = Table.location.where_to_download_data_from
    filepath: Path = Table.filepath
    datapath: Path = filepath / "data".upper()
    asyncio.run(download_from_link(download_link, datapath))

def build_graph(graph_config_path: Path) -> None:
    graph_yaml: Any = load_yaml(graph_config_path)
    Graph: GraphConfig = load_model(graph_yaml, GraphConfig)
    table_config_locations: set[FilePath] = Graph.location.table_config_containing_directories
    sections: list[tuple[Section, int]] = get_sections(table_config_locations)
    with Pool() as pool:
        _ = pool.starmap(build_subgraphs, sections)
        