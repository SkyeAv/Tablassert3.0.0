from tablassert.enums import Categories
from tablassert.utils import samphash
from tablassert.log import logger
from tempfile import gettempdir
from typing import Optional
from pathlib import Path
from operator import add
import polars as pl

def distinct(lf: pl.LazyFrame, l0: str, l1: str) -> pl.LazyFrame:
  # ? Extract Unique Terms From Two Text Normalization Columns As LazyFrame
  t0: pl.LazyFrame = lf.select(pl.col(l0).alias("term")).unique()
  t0 = t0.with_columns(pl.lit(0).alias("nlp level"))

  t1: pl.LazyFrame = lf.select(pl.col(l1).alias("term")).unique()
  t1 = t1.with_columns(pl.lit(1).alias("nlp level"))

  terms: pl.LazyFrame = pl.concat([t0, t1]).unique(subset=["term"])

  bad: str = r"^\d+$|^(none|nan|na|null|unknown)$|^$"
  return terms.filter(~pl.col("term").str.contains(bad))

def to_temp(lf: pl.LazyFrame, tmp: Path = Path(gettempdir())) -> Path:
  # ? Writes LazyFrame To A Tempfile To Be Used In Fullmap
  # ! Collection Point: samphash And write_parquet Require Eager
  df: pl.DataFrame = lf.collect()
  p: Path = tmp / samphash(df)
  p = p.with_suffix(".parquet")
  df.write_parquet(p)
  return p

def query_builder(
  p: Path,
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]],
  taxon: Optional[str]
) -> str:
  # ? Build Query With UNION For Better Index Utilization
  base: str = """
  SELECT
    PA.term,
    CU.CURIE,
    CU.PREFERRED_NAME,
    CA.CATEGORY_NAME,
    CU.TAXON_ID,
    SO.SOURCE_NAME,
    SO.SOURCE_VERSION,
    PA."nlp level" AS NLP_LEVEL,
    CASE
      {priority_case}
      ELSE 50
    END AS PR
  FROM SYNONYMS SY
  JOIN SOURCES SO ON SY.SOURCE_ID = SO.SOURCE_ID
  JOIN CURIES CU ON SY.CURIE_ID = CU.CURIE_ID
  JOIN CATEGORIES CA ON CU.CATEGORY_ID = CA.CATEGORY_ID
    {avoid_filter}
  JOIN read_parquet('{parquet}') PA ON PA.term = SY.SYNONYM
  {taxon_filter}
"""

  priority_case: str = f"WHEN CA.CATEGORY_NAME IN ({", ".join(f"'{x}'" for x in prioritize)}) THEN 1" if prioritize else "WHEN TRUE THEN 50"
  avoid_filter: str = f"AND CA.CATEGORY_NAME NOT IN ({", ".join(f"'{x}'" for x in avoid)})" if avoid else ""
  taxon_filter: str = f"WHERE CU.TAXON_ID = {taxon} OR CA.CATEGORY_NAME != 'Gene'" if taxon else ""

  return base.format(
    priority_case=priority_case,
    avoid_filter=avoid_filter,
    taxon_filter=taxon_filter,
    parquet=p
  )

def query_distinct(
  p: Path,
  conn: object,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]]
) -> pl.DataFrame:
  # ? Query Database For Distinct Terms Only Using Persistent Connection
  # * Added Column Prioritization Logic From 4.2.0
  query: str = query_builder(p, prioritize, avoid, taxon)
  results: pl.DataFrame = conn.execute(query).pl() # pyright: ignore

  frequency: pl.DataFrame = results.group_by("CATEGORY_NAME").agg(pl.len().alias("FREQUENCY"))
  results = results.join(frequency, on="CATEGORY_NAME", how="left")

  results = results.sort(["term", "PR", "NLP_LEVEL", "FREQUENCY"], descending=[False, False, False, True])
  results = results.unique(subset=["term"], keep="first")

  p.unlink(missing_ok=True)
  return results

def version4(
  lf: pl.LazyFrame,
  col: str,
  conn: object,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]],
  section_hash: str,
  config_file: str,
  tag: str = " one"
) -> pl.LazyFrame:
  # ? Case Dependant, Provenance Rich Name Entity Recognition
  l0: str = col
  l1: str = add(l0, tag)

  terms: pl.LazyFrame = distinct(lf, l0, l1)
  p: Path = to_temp(terms)
  matches: pl.DataFrame = query_distinct(p, conn, taxon, prioritize, avoid)

  # * Log Unmatched Entities
  antimatches: pl.LazyFrame = terms.join(matches.lazy().select("term"), left_on="term", right_on="term", how="anti")

  # ! Collection Point: Requires Eager
  unnmatched: pl.DataFrame = antimatches.select("term").unique().collect()
  if unnmatched.height > 0:
    for term in unnmatched.get_column("term").to_list():
      logger.info(f"FAILED FULLMAP | STORE: {section_hash} | CONFIG: {config_file} | COL: {col} | VALUE: {term!r}")

  # ! Collection Point: Join After DuckDB Query, Then Re-Lazy
  df: pl.DataFrame = lf.collect()
  result: pl.DataFrame = df.join(
    matches.filter(pl.col("NLP_LEVEL").eq(0)),
    left_on=l0,
    right_on="term",
    how="left",
    suffix=" l0"
  )

  l1_matches: pl.DataFrame = matches.filter(pl.col("NLP_LEVEL").eq(1))
  result = result.join(
    l1_matches,
    left_on=l1,
    right_on="term",
    how="left",
    suffix=" l1"
  )

  result = result.with_columns([
    pl.when(pl.col("CURIE").is_not_null())
      .then(pl.col("CURIE"))
      .otherwise(pl.col("CURIE l1"))
      .alias(col),
    pl.when(pl.col("PREFERRED_NAME").is_not_null())
      .then(pl.col("PREFERRED_NAME"))
      .otherwise(pl.col("PREFERRED_NAME l1"))
      .alias(add(col, " name")),
    pl.when(pl.col("CATEGORY_NAME").is_not_null())
      .then(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME")))
      .otherwise(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME l1")))
      .alias(add(col, " category")),
    pl.when(pl.col("TAXON_ID").is_not_null())
      .then(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID").cast(pl.String)))
      .otherwise(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID l1").cast(pl.String)))
      .alias(add(col, " taxon")),
    pl.when(pl.col("SOURCE_NAME").is_not_null())
      .then(pl.col("SOURCE_NAME"))
      .otherwise(pl.col("SOURCE_NAME l1"))
      .alias(add(col, " source")),
    pl.when(pl.col("SOURCE_VERSION").is_not_null())
      .then(pl.col("SOURCE_VERSION"))
      .otherwise(pl.col("SOURCE_VERSION l1"))
      .alias(add(col, " source version")),
    pl.when(pl.col("NLP_LEVEL").is_not_null())
      .then(pl.col("NLP_LEVEL"))
      .otherwise(pl.col("NLP_LEVEL l1"))
      .alias(add(col, " nlp level"))
  ])

  result = result.select(pl.exclude(r"^(CURIE|PREFERRED_NAME|CATEGORY_NAME|TAXON_ID|SOURCE_NAME|SOURCE_VERSION|NLP_LEVEL|PR|FREQUENCY)( l1)?$"))
  result = result.select(pl.exclude(add(col, " one")))
  result = result.with_columns(pl.col(add(col, " taxon")).replace("NCBITaxon:0", None))
  result = result.filter(pl.col(col).is_not_null())

  return result.lazy()
