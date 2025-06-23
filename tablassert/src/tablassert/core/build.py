from tablassert.src.tablassert.utils.io import load_yaml, load_model, get_sections
from tablassert.src.tablassert.models.table_config import GraphConfig
from tablassert.src.tablassert.models.table_config import Section
from pydantic import FilePath
from pathlib import Path
from Typing import Any

def build(graph_config_path: Path) -> None:
    graph_yaml: Any = load_yaml(graph_config_path)
    Graph: GraphConfig = load_model(graph_yaml, GraphConfig)
    table_config_locations: set[FilePath] = Graph.location.table_config_containing_directories
    sections: set[Optional[Section]] = get_sections(table_config_locations)
    