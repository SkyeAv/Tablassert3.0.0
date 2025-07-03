from os.path import basename
from pathlib import Path
from typing import Any
import polars as pl

EXPORT_PATH: Path = Path("TABLASSERT/EXPORT").resolve()


def savepath(posix_filepath: str, metadata: dict[str, Any], idx: int) -> Path:
    graphname: str = metadata["knowledge_graph_name"]
    version: str = metadata["graph_version"]
    savepath: Path = (
        EXPORT_PATH
        / graphname
        / version
        / f"SECTION_{str(idx)}_{basename(posix_filepath)}"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    return savepath.with_suffix(".tsv")


def save(df: pl.DataFrame, savepath: Path) -> None:
    df.write_csv(savepath, separator="\t", float_scientific=None, float_precision=4)
    return None


def aggregate(graphmodel: dict[str, Any]) -> None:
    metadata: dict[str, Any] = graphmodel["metadata"]
    graphname: str = metadata["knowledge_graph_name"]
    version: str = metadata["graph_version"]
    graphdir: Path = EXPORT_PATH / graphname / version

    lazy_frames: list[pl.LazyFrame] = []
    for tsv in graphdir.rglob("*"):
        if tsv.is_file():
            lf: pl.LazyFrame = pl.read_csv(
                tsv, separator="\t", infer_schema=False, low_memory=True
            ).lazy()
            lazy_frames.append(lf)

    if lazy_frames:
        df: pl.LazyFrame = pl.concat(lazy_frames, rechunk=False)
        finalpath: Path = Path(f"{graphname}_{version}.tsv").resolve()
        save(df.collect(), finalpath)
    return None
