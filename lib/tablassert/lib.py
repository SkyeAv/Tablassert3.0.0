from tablassert.enums import EncodingMethods
from tablassert.utils import namespace_uuid
from tablassert.models import NodeEncoding
from tablassert.downloader import from_url
from tablassert.ingests import to_sections
from tablassert.ingests import from_yaml
from tablassert.qc import fullmap_audit
from tablassert.fullmap import version4
from tablassert.models import Encoding
from tablassert.models import Section
from tablassert.utils import samphash
from tablassert.utils import mkhash
from tablassert.enums import Tokens
from pydantic import NonNegativeInt
from tablassert.models import Graph
from tablassert.enums import Files
from tablassert.utils import STORE
from sqlite_utils import Database
from pydantic import PositiveInt
from multiprocessing import Pool
from tempfile import gettempdir
from functools import reduce
from os.path import basename
from itertools import chain
from typing import Callable
from typing import Optional
from typing import Literal
from pydantic import Field
from pathlib import Path
from typing import Union
from operator import add
from typing import Self
from operator import eq
from operator import le
from typing import Any
from os import environ
import polars as pl
import subprocess
import operator
import orjson
import typer
import math

def value(df: pl.DataFrame, col: Any, x: str) -> pl.DataFrame:
  # ? Creates A New Column With A Literal Value
  return df.with_columns(pl.lit(x).alias(col))

def column(df: pl.DataFrame, col: str, x: str) -> pl.DataFrame:
  # ? Creates A New Column With From An Old Column
  return df.with_columns(pl.col(x).alias(col))

def math_op(
  df: pl.DataFrame,
  col: str,
  func: str,
  args: list[Union[Tokens.VALUES, float, int]]
) -> pl.DataFrame:
  # ? Transform Values In A Column With The Math Module
  expr: pl.Expr = pl.col(col).cast(pl.Float64)
  attr: Callable[[Any], Any] = getattr(math, func)
  transform: Callable[[float], float] = lambda x: attr(x if eq(a, Tokens.VALUES) else a for a in args)
  return df.with_columns(expr.map_elements(transform).alias(col))

def zero(df: pl.DataFrame, col: str) -> pl.DataFrame:
  # ? Level Zero Text Processing
  expr: pl.Expr = pl.col(col).cast(pl.String).str.strip_chars().str.to_lowercase()
  return df.with_columns(expr.alias(col))

def one(
  df: pl.DataFrame,
  col: str,
  regex: str = r"\W+",
  tag: str = " one"
) -> pl.DataFrame:
  # ? Level One Text Processing
  expr: pl.Expr = pl.col(col).str.replace_all(regex, "")
  col: str = add(col, tag)
  return df.with_columns(expr.alias(col))

def prefix(df: pl.DataFrame, col: str, prefix: str) -> pl.DataFrame:
  expr: pl.Expr = add(pl.lit(prefix), pl.col(col).cast(pl.String))
  return df.with_columns(expr.alias(col))

def suffix(df: pl.DataFrame, col: str, suffix: str) -> pl.DataFrame:
  expr: pl.Expr = add(pl.col(col).cast(pl.String), pl.lit(suffix))
  return df.with_columns(expr.alias(col))

def regex(
  df: pl.DataFrame,
  col: str,
  pattern: str,
  replacement: str = ""
) -> pl.DataFrame:
  expr: pl.Expr = pl.col(col).cast(pl.String).str.replace_all(pattern, replacement)
  return df.with_columns(expr.alias(col))

def fill(df: pl.DataFrame, col: str, method: str) -> pl.DataFrame:
  expr: pl.Expr = pl.col(col).fill_null(strategy=method)
  return df.with_columns(expr.alias(col))

def explode(df: pl.DataFrame, col: str, delimiter: str) -> pl.DataFrame:
  # ? Explodes A Row With Items Into Many Unique Rows By A Delimiter
  expr: pl.Expr = pl.col(col).cast(pl.String).str.split(delimiter)
  return df.with_columns(expr.explode(col).alias(col))

def sig(
  df: pl.DataFrame,
  cutoff: float = 0.05,
  col: str = "p value",
  out: str = "significant",
) -> pl.DataFrame:
  # ? Creates The "significant" Column
  print("PreSig", df.shape)

  if col in df.columns:
    expr: pl.Expr = pl.col(col).cast(pl.Float64)
    cond: pl.Expr = le(expr, cutoff)
    cutoff: pl.Expr = pl.when(expr.is_null()).then(pl.lit("UNSURE")).when(cond).then(pl.lit("YES")).otherwise(pl.lit("NO"))
    return df.with_columns(cutoff.alias(out))

  else:
    return df.with_columns(pl.lit("UNSURE").alias(out))

def idx(df: pl.DataFrame, col: str = "row number") -> pl.DataFrame:
  # ? Creates An Index Column Of Row Numbers
  return df.with_row_index(col)

def csv(p: Path, sep: str) -> pl.DataFrame:
  # ? Reads Source From CSV And TSV
  return pl.read_csv(
    source=p,
    separator=sep,
    has_header=False,
    infer_schema_length=None,
    truncate_ragged_lines=True
  )

def excel(p: Path, sheet: str, engine: str = "calamine") -> pl.DataFrame:
  # ? Reads Source From Excel
  return pl.read_excel(
    source=p,
    sheet_name=sheet,
    engine=engine,
    has_header=False,
    infer_schema_length=None
  )

def crop(df: pl.DataFrame, row_slice: Optional[list[Union[NonNegativeInt, Tokens.AUTO]]]) -> pl.DataFrame:
  # ? Takes A Slice From A DataFrame
  n: int = df.height
  start: Union[int, Literal[Tokens.AUTO]] = row_slice[0]
  stop: Union[int, Literal[Tokens.AUTO]] = row_slice[1]
  offset: int = 0 if eq(start, Tokens.AUTO) else start
  length: int = n if eq(stop, Tokens.AUTO) else (stop - offset)
  return df.slice(offset=offset, length=length)

def pick(df: pl.DataFrame, rows: list[int]) -> pl.DataFrame:
  # ? Picks A List Of Rows From A DataFrame
  return df.select(pl.all().take(indices=rows))

def reindex(
  df: pl.DataFrame,
  col: str,
  op: operator,
  comp: Union[str, int, float],
  cast: bool = True
) -> pl.DataFrame:
  # ? Reindex A DataFrame Based On A Condition
  expr: pl.Expr = pl.col(col).cast(pl.Float64) if cast else pl.col(col)
  return df.filter(op(expr, comp))

def idxname(col: str) -> str:
  # ? Converts Excel Style Column Names To Polars Column Names
  idx: int = 0
  for char in col:
    idx = idx * 26 + (ord(char) - 65 + 1)

  return f"column_{idx}"

def trim(df: pl.DataFrame, regex: str = r"^column_\d+$") -> pl.DataFrame:
  # ? Removes Columns With The Excel Naming Conventions From DataFrame
  return df.select(pl.exclude(regex))

def to_temp(df: pl.DataFrame, tmp: Path = Path(gettempdir())) -> Path:
  # ? Writes DF To A Tempfile To Be Used In Fullmap
  p: Path = tmp / samphash(df)
  p = p.with_suffix(".parquet")
  df.write_parquet(p)
  return p

def to_store(df: pl.DataFrame, p: Path) -> Path:
  # ? Writes A DF To Store To Later Be Aggregated
  df.write_parquet(p)
  return p

def with_mesh(df: pl.DataFrame, pubmed_db: Path, curie: str) -> pl.DataFrame:
  # ? Adds PubMedDB Related MeSH Annotations To DF
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
  all_ids: list[str] = [x["mesh"] for x in rows if x]
  is_major: list[bool] = [eq(x["mesh_major"], "Y") for x in rows]
  domain: list[str] = [x for x, y in zip(all_ids, is_major) if y]
  mesh: list[str] = [x for x in all_ids if x not in domain]

  row: dict[str, str] = rows[0] if rows else {}
  first_author: str = row.get("firstauthor")
  journal: str = row.get("journal")
  title: str = row.get("title")
  year: str = row.get("year")

  if domain:
    df = df.with_columns(pl.lit(",".join(domain)).alias("domain"))
  if mesh:
    df = df.with_columns(pl.lit(",".join(mesh)).alias("mesh"))
  if first_author:
    df = df.with_columns(pl.lit(first_author).alias("first author"))
  if journal:
    df = df.with_columns(pl.lit(journal).alias("journal"))
  if title:
    df = df.with_columns(pl.lit(title).alias("title"))
  if year:
    df = df.with_columns(pl.lit(year).alias("year published"))

  return df

def with_captions(df: pl.DataFrame, pmc_db: Path, curie: str, url: str) -> pl.DataFrame:
  # ? Adds PMC Caption Annotations To DF With Filename Heuristic
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

  caption: str = row.get("caption")
  if caption:
    df = df.with_columns(pl.lit(caption).alias("file caption"))

  return df

class Tcode(Section):
  # ? Extends Section To Compile A KG
  number: PositiveInt = Field(...)
  store: Path = Field(...)

  def encoding(self: Self, x: Encoding, col: str) -> list[Any]:
    # ? Collect Helper For Encoding Classes
    return [
      (value, (col, x.encoding,)) if eq(x.method, EncodingMethods.VALUE) else None,
      (column, (col, idxname(x.encoding),)) if eq(x.method, EncodingMethods.COLUMN) else None,
      (fill, (col, x.fill,)) if x.fill else None,
      (explode, (col, x.explode_by,)) if x.explode_by else None,
      [(regex, (col, r.pattern, r.replacement,)) for r in x.regex] if x.regex else None,
      [(regex, (col, r,)) for r in x.remove] if x.remove else None,
      (prefix, (col, x.prefix,)) if x.prefix else None,
      (suffix, (col, x.suffix,)) if x.suffix else None,
      [(math_op, (col, col, t.function, t.arguments,)) for t in x.transformations] if x.transformations else None
    ]

  def node(self: Self, x: NodeEncoding, col: str, dbssert: Path) -> list[Any]:
    # ? Collect Helper For NodeEncoding Classes
    encoding: list[Any] = self.encoding(x, col)
    node: list[Any] = [
      (column, (add("original ", col), col,)),
      (zero, (col,)),
      (one, (col,)),
      (to_temp, ()),
      (version4, (col, dbssert, x.taxon, x.prioritize, x.avoid,)),
      (fullmap_audit, (col,))
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

  def collect(self: Self, dbssert: Path, pubmed_db: Path, pmc_db: Path) -> Union[list[tuple[Callable, tuple[Any]]], Path]:
    # ? Code That Tells Tablassert What Actions To While Transforming Data

    if self.store.is_file():
      # * Quick Exit If Subgraph Already Exists
      return self.store

    else:
      # * Returns A List Of: (Function, (Arguments))
      tcode: Optional[list[Any]] = [
        (from_url, (str(self.source.url), self.source.local,)),
        (csv, (self.source.delimiter,)) if eq(self.source.kind, Files.TEXT) else None,
        (excel, (self.source.sheet,)) if eq(self.source.kind, Files.EXCEL) else None,
        (idx, ()),
        (crop, (self.source.row_slice,)) if self.source.row_slice else None,
        (pick, (self.source.rows,)) if self.source.rows else None,
        [(reindex, (idxname(x.column), getattr(operator, x.comparison), x.comparator,)) for x in self.source.reindex] if self.source.reindex else None,
        [op for x in self.annotations for op in self.encoding(x, x.annotation)] if self.annotations else None,
        self.node(self.statement.subject, "subject", dbssert),
        self.node(self.statement.object, "object", dbssert),
        (value, ("predicate", self.statement.predicate,)),
        [op for x in self.statement.qualifiers for op in self.node(x, x.qualifier, dbssert)] if self.statement.qualifiers else None,
        (value, ("syntax", self.syntax,)),
        (value, ("section number", self.number,)),
        (value, ("status", self.status,)),
        (value, ("repository", self.provenance.repo,)),
        (value, ("publication", self.provenance.publication,)),
        (value, ("contributors", [{k: v} for x in self.provenance.contributors for k, v in x.model_dump().items() if v],)),
        (value, ("url", str(self.source.url),)),
        (value, ("section md5", self.store.stem,)),
        (with_mesh, (pubmed_db, self.provenance.publication,)),
        (with_captions, (pmc_db, self.provenance.publication, str(self.source.url),)),
        (sig, ()),
        (trim, ()),
        (to_store, (self.store,))
      ]
      return self.clean(tcode)

def compile_subgraph(tcode: list[tuple[Callable, tuple[Any]]]) -> Path:
  # ? Executes Tcode To Build Subgraphs As Parquets
  return reduce(lambda acc, op: op[0](acc, *op[1]) if acc is not None else op[0](*op[1]), tcode, None)

def normalize_node(edges: pl.DataFrame, col: str, names: list[str] = ["id", "name", "category", "taxon", "source", "source version"]) -> pl.DataFrame:
  # ? Converts Disparate Columns Containing Nodes Into A Unified Column
  cols: list[str] = [col, add(col, " name"), add(col, " category"), add(col, " taxon"), add(col, " source"), add(col, " source version")]
  edges = edges.select(cols).unique()
  return edges.rename({k: v for k, v in zip(cols, names)})

def label_edges(e_in: Path, domain: str = "MOKG", out: str = "uuid") -> None:
  # ? Gives Each Edge In MOKG A UUID
  e_out = e_in.with_suffix(".edges.ndjson")
  with e_in.open("rb") as f_in, e_out.open("wb") as f_out:
    for line in f_in:
      r: object = orjson.loads(line)
      r[out] = namespace_uuid(domain, *r.values())

      b: bytes = orjson.dumps(r) + b"\n"
      f_out.write(b)

  e_in.unlink()

def compile_graph(subgraphs: list[Path], name: str, version: str) -> tuple[Path]:
  # ? Aggregates Parquets For NDJSON KGX Export
  p: Path = Path(f"./{name}_{version}")
  e: Path = p.with_suffix(".edges.ndjson.temp") # ! For Labeling
  n: Path = p.with_suffix(".nodes.ndjson")

  for s in subgraphs:
    edges: pl.DataFrame = pl.read_parquet(s)

    with e.open("a") as f:
      edges.write_ndjson(f)

    node_cols: list[str] = [col.replace("original ", "") for col in edges.columns if "original " in col]
    nodes: pl.DataFrame = pl.concat([normalize_node(edges, col) for col in node_cols], how="vertical")
    nodes = nodes.unique()

    with n.open("a") as f:
      nodes.write_ndjson(f)

  awk: Path = environ.get("AWK_PATH")
  jq: Path = environ.get("JQ_PATH")

  for x in [e, n]:
    temp: Path = x.with_suffix(add(x.suffix, ".tmp"))
    command = f"{jq} -c 'walk(if type == \"object\" then with_entries(select(.value != null)) else . end)' {x} | {awk} '!seen[$0]++' > {temp} && mv {temp} {x}"
    subprocess.run(command, shell=True, check=True)

  label_edges(e)

CLI: typer.Typer = typer.Typer(pretty_exceptions_show_locals=False)

@CLI.command()
def main(
  ingest: Path = typer.Option(..., "-i", "-ingest", help="Knowledge Graph Configuration -- See Docs")
) -> None:
  """Tablassert Builds Knowledge Graphs From Declarative Configuration"""
  r: object = from_yaml(ingest)
  g: Graph = Graph.model_validate(r)
  with Pool() as pool:
    raw: list[object] = pool.map(from_yaml, g.tables)
    temp: list[list[dict[str, Any]]] = pool.map(to_sections, raw)
    sections: list[dict[str, Any]] = list(chain.from_iterable(temp))
    print(sections)

    tcode: list[Tcode] = [Tcode.model_validate({**s, "number": idx, "store": (STORE / f"{mkhash(s)}.parquet")}) for idx, s in enumerate(sections, start=1)]
    instructions: Union[list[tuple[Callable, tuple[Any]]], Path] = [x.collect(g.dbssert, g.pubmed_db, g.pmc_db) for x in tcode]
    print(instructions)

  subgraphs: list[Path] = [op if isinstance(op, Path) else compile_subgraph(op) for op in instructions]
  compile_graph(subgraphs, g.name, g.version)
