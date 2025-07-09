from typing import Any, Optional, Union
from os.path import basename
from pathlib import Path
import polars as pl

EXPORT_PATH: Path = Path("TABLASSERT/EXPORT").resolve()


def savepath(
    posix_filepath: str, sheetname: Optional[str], metadata: dict[str, Any], idx: int
) -> Path:
    graphname: str = metadata["knowledge_graph_name"]
    version: str = metadata["graph_version"]
    savepath: Path = (
        EXPORT_PATH
        / graphname
        / version
        / f"SECTION_{str(idx)}{f"_{sheetname}" if sheetname else ""}_{basename(posix_filepath)}"
    )
    savepath.parent.mkdir(parents=True, exist_ok=True)
    return savepath.with_suffix(".tsv")


def save(df: Union[pl.DataFrame, pl.LazyFrame], savepath: Path) -> None:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
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
        edges: pl.LazyFrame = pl.concat(lazy_frames, rechunk=False)
        edgespath: Path = Path(f"{graphname}_{version}_edges.tsv").resolve()
        save(edges, edgespath)
        subjectnodes: pl.LazyFrame = edges.select(
            [
                pl.col("subject").alias("id"),
                pl.col("subject_name").alias("name"),
                pl.col("subject_category").alias("category"),
            ]
        )
        objectnodes: pl.LazyFrame = edges.select(
            [
                pl.col("object").alias("id"),
                pl.col("object_name").alias("name"),
                pl.col("object_category").alias("category"),
            ]
        )
        nodes: pl.LazyFrame = pl.concat(
            [subjectnodes, objectnodes], how="vertical"
        ).unique(maintain_order=True)
        nodespath: Path = Path(f"{graphname}_{version}_nodes.tsv").resolve()
        save(nodes, nodespath)
    return None
