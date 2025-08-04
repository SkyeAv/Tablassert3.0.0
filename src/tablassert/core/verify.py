from src.tablassert.utils.io import load_yaml, load_model, build_sections
from src.tablassert.models.graph import GraphConfig
from pathlib import Path
from typing import Any
import tempfile
import shutil


def verify_graphconfig(graphconfig: str) -> None:
    graphconfigpath: Path = Path(graphconfig)
    if not graphconfigpath.exists():
        raise RuntimeError(f"CODE:301B | {graphconfig} does not exist")
    graph_yaml: Any = load_yaml(graphconfigpath)
    _ = load_model(graph_yaml, GraphConfig)
    print(f"Congrats! {graphconfig} is a valid GraphConfig!!")
    return None


def verify_tableconfig(tableconfig: str) -> None:
    tableconfigpath: Path = Path(tableconfig)
    if not tableconfigpath.exists():
        raise RuntimeError(f"CODE:301C | {tableconfig} does not exist")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdirpath: Path = Path(tmpdir)
        _ = shutil.copy(tableconfigpath, tmpdirpath)
        fakegraphmodel = {
            "location": {"table_config_containing_directories": tmpdirpath}
        }
        _ = build_sections(fakegraphmodel)
    print(f"Congrats! {tableconfig} is a valid TableConfig!!")
    return None
