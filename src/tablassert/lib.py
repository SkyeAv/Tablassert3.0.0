from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from contextlib import ExitStack
from functools import reduce
from operator import add, eq, le
from os.path import basename
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Self, Union

import lazy_loader as Lazy
from pydantic import Field, NonNegativeInt, PositiveInt
from sqlite_utils import Database

from tablassert.downloader import from_url
from tablassert.enums import Categories, EncodingMethods, Files, Tokens
from tablassert.fullmap import SHARDS, resolve
from tablassert.log import logger
from tablassert.models import Encoding, NodeEncoding, Section
from tablassert.nlp import level_one, level_two
from tablassert.qc import fullmap_audit
from tablassert.utils import namespace_uuid

if TYPE_CHECKING:
    import duckdb
    import orjson
    import polars as pl
    import xxhash
else:
    duckdb = Lazy.load("duckdb")
    orjson = Lazy.load("orjson")
    pl = Lazy.load("polars")
    xxhash = Lazy.load("xxhash")


def value(lf: pl.LazyFrame, col: str, x: str) -> pl.LazyFrame:
    # ? Creates A New Column With A Literal Value
    return lf.with_columns(pl.lit(x).alias(col))


def contributor_values(lf: pl.LazyFrame, col: str, contributors: list[dict[str, Any]]) -> pl.LazyFrame:
    # ? Adds Nested Contributors Fields To Column
    return lf.with_columns(pl.lit([x.model_dump() for x in contributors]).alias(col))  # pyright: ignore


def column(lf: pl.LazyFrame, col: str, x: str) -> pl.LazyFrame:
    # ? Creates A New Column With From An Old Column
    return lf.with_columns(pl.col(x).alias(col))


def math_op(
    lf: pl.LazyFrame, col: str, func: str, args: list[Union[Literal[Tokens.VALUES], float, int]]
) -> pl.LazyFrame:
    # ? Transform Values In A Column With The Math Module
    # ! Collection Point: Required For map_elements
    df: pl.DataFrame = lf.collect()
    expr: pl.Expr = pl.col(col).cast(pl.Float64)
    attr: Callable[[Any], Any] = getattr(math, func)
    df = df.with_columns(
        expr.map_elements(
            lambda x: attr(*(x if eq(a, Tokens.VALUES) else a for a in args)), return_dtype=pl.Float64
        ).alias(col)
    )
    return df.lazy()


def prefix(lf: pl.LazyFrame, col: str, prefix: str) -> pl.LazyFrame:
    expr: pl.Expr = add(pl.lit(prefix), pl.col(col).cast(pl.String))
    return lf.with_columns(expr.alias(col))


def suffix(lf: pl.LazyFrame, col: str, suffix: str) -> pl.LazyFrame:
    expr: pl.Expr = add(pl.col(col).cast(pl.String), pl.lit(suffix))
    return lf.with_columns(expr.alias(col))


def regex(lf: pl.LazyFrame, col: str, pattern: str, replacement: str = "") -> pl.LazyFrame:
    expr: pl.Expr = pl.col(col).cast(pl.String).str.replace_all(pattern, replacement)
    return lf.with_columns(expr.alias(col))


def fill(lf: pl.LazyFrame, col: str, method: str) -> pl.LazyFrame:
    expr: pl.Expr = pl.col(col).fill_null(strategy=method)  # pyright: ignore
    return lf.with_columns(expr.alias(col))


def explode(lf: pl.LazyFrame, col: str, delimiter: str) -> pl.LazyFrame:
    # ? Explodes A Row With Items Into Many Unique Rows By A Delimiter
    expr: pl.Expr = pl.col(col).cast(pl.String).str.split(delimiter)
    lf = lf.with_columns(expr.alias(col))
    return lf.explode(col)


def sig(
    lf: pl.LazyFrame,
    cutoff: float = 0.05,  # pyright: ignore
    col: str = "p value",
    out: str = "significant",
) -> pl.LazyFrame:
    # ? Creates The "significant" Column
    if col in lf.collect_schema().names():
        expr: pl.Expr = pl.col(col).cast(pl.Float64, strict=False)
        cond: pl.Expr = le(expr, cutoff)
        cutoff: pl.Expr = (
            pl.when(expr.is_null()).then(pl.lit("UNSURE")).when(cond).then(pl.lit("YES")).otherwise(pl.lit("NO"))
        )
        return lf.with_columns(cutoff.alias(out))

    else:
        return lf.with_columns(pl.lit("UNSURE").alias(out))


def idx(lf: pl.LazyFrame, col: str = "row number") -> pl.LazyFrame:
    # ? Creates An Index Column Of Row Numbers
    return lf.with_row_index(col)


def csv(p: Path, sep: str) -> pl.LazyFrame:
    # ? Reads Source From CSV And TSV As LazyFrame
    return pl.scan_csv(source=p, separator=sep, has_header=False, infer_schema_length=None, truncate_ragged_lines=True)


def excel(p: Path, sheet: str, engine: str = "calamine") -> pl.LazyFrame:
    # ? Reads Source From Excel As LazyFrame
    df: pl.DataFrame = pl.read_excel(
        source=p,
        sheet_name=sheet,
        engine=engine,  # pyright: ignore
        has_header=False,
        infer_schema_length=None,
    )
    return df.lazy()


def crop(lf: pl.LazyFrame, row_slice: list[Union[NonNegativeInt, Literal[Tokens.AUTO]]]) -> pl.LazyFrame:
    # ? Takes A Slice From A LazyFrame
    # ! Collection Point: Requires Height Calculation
    df: pl.DataFrame = lf.collect()
    n: int = df.select(pl.len()).item()
    start: Union[int, Literal[Tokens.AUTO]] = row_slice[0]
    stop: Union[int, Literal[Tokens.AUTO]] = row_slice[1]
    offset: int = 0 if eq(start, Tokens.AUTO) else start  # pyright: ignore
    length: int = n if eq(stop, Tokens.AUTO) else (stop - offset)  # pyright: ignore
    df = df.slice(offset=offset, length=length)
    return df.lazy()


def pick(lf: pl.LazyFrame, rows: list[int]) -> pl.LazyFrame:
    # ? Picks A List Of Rows From A LazyFrame
    # ! Collection Point: take() Requires Eager, Relazy After
    df: pl.DataFrame = lf.collect()
    df = df.select(pl.all().take(indices=rows))  # pyright: ignore
    return df.lazy()


def reindex(df: pl.LazyFrame, col: str, op: Callable, comp: Union[str, int, float], cast: bool = True) -> pl.LazyFrame:
    # ? Reindex A LazyFrame Based On A Condition
    expr: pl.Expr = pl.col(col).cast(pl.Float64) if cast else pl.col(col)
    return df.filter(op(expr, comp))


def idxname(col: Any) -> str:
    # ? Converts Excel Style Column Names To Polars Column Names
    scol: str = str(col)
    idx: int = 0
    for char in scol:
        idx = idx * 26 + (ord(char) - 65 + 1)

    return f"column_{idx}"


def trim(lf: pl.LazyFrame, regex: str = r"^column_\d+$") -> pl.LazyFrame:
    # ? Removes Columns With The Excel Naming Conventions From LazyFrame
    return lf.select(pl.exclude(regex))


def to_store(lf: pl.LazyFrame, p: Path, config_name: str) -> Path:
    # ? collect and write section parquet; warn if result is empty
    df: pl.DataFrame = lf.collect()

    if df.height == 0:
        logger.warning(f"EMPTY SUBGRAPH | STORE: {p.stem} | CONFIG: {config_name}")
    df.write_parquet(p)

    return p


def with_mesh(lf: pl.LazyFrame, pubmed_db: Path, curie: str) -> pl.LazyFrame:
    # ? Adds PubMedDB Related MeSH Annotations To LazyFrame
    # ! Collection Point: SQLite Query Then Per-Row Literal Assignment
    df: pl.DataFrame = lf.collect()
    db: object = Database(pubmed_db)
    query: str = """
SELECT
  mesh.mesh_major,
  mesh.mesh,
  info.firstauthor,
  info.journal,
  info.title,
  info.year
FROM ids
INNER JOIN mesh ON ids.pmid = mesh.pmid
INNER JOIN info ON ids.pmid = info.pmid
WHERE ids.alt = :curie OR ids.pmid = :curie
LIMIT 1
"""
    rows: list[dict[str, str]] = list(db.query(query, {"curie": curie})) or []
    all_ids: list[str] = [add("MESH:", x["mesh"]) for x in rows if x]
    is_major: list[bool] = [eq(x["mesh_major"], "Y") for x in rows]
    domain: list[str] = [x for x, y in zip(all_ids, is_major) if y]
    mesh: list[str] = [x for x in all_ids if x not in domain]

    row: dict[str, str] = rows[0] if rows else {}
    first_author: Optional[str] = row.get("firstauthor")
    journal: Optional[str] = row.get("journal")
    title: Optional[str] = row.get("title")
    year: Optional[str] = row.get("year")

    if domain:
        df = df.with_columns(pl.lit(domain).alias("domain"))
    if mesh:
        df = df.with_columns(pl.lit(mesh).alias("mesh"))
    if first_author:
        df = df.with_columns(pl.lit(first_author).alias("first author"))
    if journal:
        df = df.with_columns(pl.lit(journal).alias("journal"))
    if title:
        df = df.with_columns(pl.lit(title).alias("title"))
    if year:
        df = df.with_columns(pl.lit(year).alias("year published"))

    return df.lazy()


def with_captions(lf: pl.LazyFrame, pmc_db: Path, curie: str, url: str) -> pl.LazyFrame:
    # ? Adds PMC Caption Annotations To LazyFrame With Filename Heuristic
    # ! Collection Point: SQLite Query Then Literal Assignment
    df: pl.DataFrame = lf.collect()
    db: object = Database(pmc_db)
    filename: str = basename(url)
    query: str = """
SELECT caption
FROM captions
WHERE pmc = :curie AND file = :filename
LIMIT 1
"""
    rows: list[dict[str, str]] = list(db.query(query, {"curie": curie, "filename": filename})) or []
    row: dict[str, str] = rows[0] if rows else {}

    caption: Optional[str] = row.get("caption")
    if caption:
        df = df.with_columns(pl.lit(caption).alias("file caption"))

    return df.lazy()


class Tcode(Section):
    # ? Extends Section To Compile A KG
    number: PositiveInt = Field(...)
    config: Path = Field(...)
    store: Path = Field(...)

    def encoding(self: Self, x: Encoding, col: str) -> list[Any]:
        # ? Collect Helper For Encoding Classes
        return [
            (value, (col, x.encoding)) if eq(x.method, EncodingMethods.VALUE) else None,
            (column, (col, idxname(x.encoding))) if eq(x.method, EncodingMethods.COLUMN) else None,
            (fill, (col, x.fill)) if x.fill else None,
            (explode, (col, x.explode_by)) if x.explode_by else None,
            [(regex, (col, r.pattern, r.replacement)) for r in x.regex] if x.regex else None,
            [(regex, (col, r)) for r in x.remove] if x.remove else None,
            (prefix, (col, x.prefix)) if x.prefix else None,
            (suffix, (col, x.suffix)) if x.suffix else None,
            [(math_op, (col, t.function, t.arguments)) for t in x.transformations] if x.transformations else None,
        ]

    def node(self: Self, x: NodeEncoding, col: str, conns: list[object]) -> list[Any]:
        # ? Collect Helper For NodeEncoding Classes
        encoding: list[Any] = self.encoding(x, col)
        node: list[Any] = [
            (column, (add("original ", col), col)),
            (level_one, (col,)),
            (level_two, (col,)),
            (resolve, (col, conns, x.taxon, x.prioritize, x.avoid, True, self.store.stem, self.config.name, True)),
            (fullmap_audit, (col, self.store.stem, self.config.name)),
        ]
        return add(encoding, node)

    def clean(self: Self, tcode: list[tuple[Callable, Any]]) -> list[tuple[Callable, tuple[Any]]]:
        # ? Cleans Tcode So It Can Be Used With reduce From functools
        result: list[tuple[Callable, tuple[Any]]] = []
        for x in tcode:
            if not x:
                continue
            elif isinstance(x, list):
                result.extend(self.clean(x))
            else:
                result.append(x)
        return result

    def collect(
        self: Self, conns: list[object], pubmed_db: Optional[Path], pmc_db: Optional[Path]
    ) -> Union[list[tuple[Callable, tuple[Any]]], Path]:
        # ? Code That Tells Tablassert What Actions To While Transforming Data

        if self.store.is_file():
            # * Quick Exit If Subgraph Already Exists
            return self.store

        else:
            # * Returns A List Of: (Function, (Arguments))
            tcode: Optional[list[Any]] = [
                (from_url, (str(self.source.url), self.source.local, self.config.name, self.store.stem)),
                (csv, (self.source.delimiter,)) if eq(self.source.kind, Files.TEXT) else None,  # pyright: ignore
                (excel, (self.source.sheet,)) if eq(self.source.kind, Files.EXCEL) else None,  # pyright: ignore
                (idx, ()),
                (crop, (self.source.row_slice,)) if self.source.row_slice else None,
                (pick, (self.source.rows,)) if self.source.rows else None,
                [
                    (reindex, (idxname(x.column), getattr(operator, x.comparison), x.comparator))
                    if x.comparison not in ["ne", "eq"]
                    else (reindex, (idxname(x.column), getattr(operator, x.comparison), x.comparator, False))
                    for x in self.source.reindex
                ]
                if self.source.reindex
                else None,
                [op for x in self.annotations for op in self.encoding(x, x.annotation)] if self.annotations else None,
                self.node(self.statement.subject, "subject", conns),
                self.node(self.statement.object, "object", conns),
                (value, ("predicate", self.statement.predicate)),
                [op for x in self.statement.qualifiers for op in self.node(x, x.qualifier, conns)]
                if self.statement.qualifiers
                else None,
                (value, ("syntax", self.syntax)),
                (value, ("configuration file", self.config.name)),
                (value, ("section number", self.number)),
                (value, ("status", self.status)),
                (value, ("repository", self.provenance.repo)),
                (value, ("publication", (self.provenance.repo + ":" + self.provenance.publication))),
                (contributor_values, ("contributors", self.provenance.contributors)),
                (value, ("url", str(self.source.url))),
                (value, ("section hash", self.store.stem)),
                (with_mesh, (pubmed_db, self.provenance.publication)) if pubmed_db else None,
                (with_captions, (pmc_db, self.provenance.publication, str(self.source.url))) if pmc_db else None,
                (sig, ()),
                (trim, ()),
                (to_store, (self.store, self.config.name)),
            ]
            return self.clean(tcode)


def compile_subgraph(tcode: list[tuple[Callable, tuple[Any]]]) -> Path:
    # ? Executes Tcode To Build Subgraphs As Parquets
    return reduce(lambda acc, op: op[0](acc, *op[1]) if acc is not None else op[0](*op[1]), tcode, None)  # pyright: ignore


def normalize(
    edges: pl.LazyFrame, col: str, names: list[str] = ["id", "name", "category", "taxon", "source", "source version"]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    # ? Normalized Disparate Node Columns To A Unified Format And Removes Them From Edges
    # * Returns Partial Nodes And Modified Edges As LazyFrames
    cols: list[str] = [
        col,
        add(col, " name"),
        add(col, " category"),
        add(col, " taxon"),
        add(col, " source"),
        add(col, " source version"),
    ]
    nodes: pl.LazyFrame = edges.select(cols).unique().rename({k: v for k, v in zip(cols, names)})
    edges_out: pl.LazyFrame = edges.drop(cols[1:])
    return nodes, edges_out


def publications(
    edges: pl.LazyFrame, names: list[str] = ["id", "name", "first author", "journal", "year published"]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    cols: list[str] = ["publication", "title", "first author", "journal", "year published"]
    cols = [x for x in cols if x in edges.collect_schema().names()]
    nodes: pl.LazyFrame = edges.select(cols).unique().rename({k: v for k, v in zip(cols, names)})
    nodes = nodes.with_columns(pl.lit("biolink:Publication").alias("category"))
    edges_out: pl.LazyFrame = edges.drop(cols[1:])
    return nodes, edges_out


def label_edge(r: object, domain: str = "TABLASSERT", out: str = "uuid") -> object:
    # ? Gives Edges A Unique UUID In The Tablassert Namespace
    r[out] = namespace_uuid(domain, *r.values())  # pyright: ignore
    return r


def strip_nulls(r: object, bad: set[str] = {"na", "nan", "null", "none", ""}) -> dict:
    # ? Removes Null Keys From NDJSON
    return {
        k: [strip_nulls(i) if isinstance(i, dict) else i for i in v]
        if isinstance(v, list)
        else strip_nulls(v)
        if isinstance(v, dict)
        else v
        for k, v in r.items()  # pyright: ignore
        if v and str(v).strip().lower() not in bad
    }


def dedup_stream(p_in: Path, is_edges: bool) -> None:
    # ? Removes Null Values From And Deduplicates NDJSON
    # * Also Adds UUIDs To Edges
    p_out: Path = p_in.with_suffix("")

    if p_out.is_file():
        p_out.unlink()

    seen: set[bytes] = set()
    with p_in.open("rb") as f_in, p_out.open("wb") as f_out:
        for line in f_in:
            r: object = orjson.loads(line)  # pyright: ignore
            r = strip_nulls(r)

            if r:
                b: bytes = orjson.dumps(r)
                h: bytes = xxhash.xxh64(b).digest()
                if h not in seen:
                    seen |= {h}

                    if is_edges:
                        r = label_edge(r)
                        b = orjson.dumps(r)

                    b = b + ("\n").encode("utf-8")
                    f_out.write(b)

    p_in.unlink()


def compile_graph(subgraphs: list[Path], name: str, version: str, fmt: str = "mixed", precision: int = 4) -> None:
    # ? Aggregates Parquets For NDJSON KGX Export Using Lazy Scan
    p: Path = Path(f"./{name}_{version}")

    e: Path = p.with_suffix(".edges.ndjson.temp")  # ! For Labeling
    if e.exists():
        e.unlink()

    n: Path = p.with_suffix(".nodes.ndjson.temp")
    if n.exists():
        n.unlink()

    subnodes: list[pl.LazyFrame] = []
    subedges: list[pl.LazyFrame] = []
    for s in subgraphs:
        lf: pl.LazyFrame = pl.scan_parquet(s)

        node_cols: list[str] = [
            col.replace("original ", "") for col in lf.collect_schema().names() if "original " in col
        ]
        for col in node_cols:
            partial, lf = normalize(lf, col)
            subnodes.append(partial)

        partial, lf = publications(lf)
        subnodes.append(partial)
        subedges.append(lf)

    # ! Collection Point: Appending To Output Files
    with n.open("a") as f:
        for subnode in subnodes:
            eagernode: pl.DataFrame = subnode.collect().unique()
            eagernode.write_ndjson(f)

    with e.open("a") as f:
        with pl.Config(set_fmt_float=fmt):  # pyright: ignore
            with pl.Config(float_precision=precision):
                for subedge in subedges:
                    eageredge: pl.DataFrame = subedge.collect().unique()
                    eageredge.write_ndjson(f)

    dedup_stream(e, is_edges=True)
    dedup_stream(n, is_edges=False)


def resolve_many(
    col: str,
    entities: Iterable[str],
    datassert: Path,
    taxon: Optional[str] = None,
    prioritize: Optional[list[Categories]] = None,
    avoid: Optional[list[Categories]] = None,
    column_context: bool = True,
) -> list[dict[str, Any]]:
    series: pl.Series = pl.Series(col, entities)
    lf: pl.LazyFrame = series.to_frame().lazy()

    lf = column(lf, add("original ", col), col)
    lf = level_one(lf, col)
    lf = level_two(lf, col)

    with ExitStack() as stack:
        conns: list[object] = [
            stack.enter_context(duckdb.connect(datassert / "data" / f"{x}.duckdb", read_only=True))
            for x in range(SHARDS)
        ]

        lf = resolve(lf, col, conns, taxon=taxon, prioritize=prioritize, avoid=avoid, column_context=column_context)

    df: pl.DataFrame = lf.collect()
    return df.to_dicts()
