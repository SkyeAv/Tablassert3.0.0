from tablassert.enums import Categories
from tablassert.utils import samphash
from tempfile import gettempdir
from typing import Optional
from pathlib import Path
from operator import add
import polars as pl
import duckdb

def distinct(df: pl.LazyFrame, l0: str, l1: str) -> pl.LazyFrame:
  # ? Extract Unique Terms From Two Text Normalization Columns As LazyFrame
  t0: pl.LazyFrame = df.select(pl.col(l0).alias("term")).unique()
  t0 = t0.with_columns(pl.lit(0).alias("nlp level"))

  t1: pl.LazyFrame = df.select(pl.col(l1).alias("term")).unique()
  t1 = t1.with_columns(pl.lit(1).alias("nlp level"))

  terms: pl.LazyFrame = pl.concat([t0, t1]).unique(subset=["term"])
  return terms.with_row_index("term id")

def to_temp(df: pl.LazyFrame, tmp: Path = Path(gettempdir())) -> Path:
  # ? Writes LazyFrame To A Tempfile To Be Used In Fullmap
  # ! Collection Point: samphash and write_parquet require eager
  eager_df: pl.DataFrame = df.collect()
  p: Path = tmp / samphash(eager_df)
  p = p.with_suffix(".parquet")
  eager_df.write_parquet(p)
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
  taxon_filter: str = f"WHERE CU.TAXON_ID = {taxon}" if taxon else ""

  return base.format(
    priority_case=priority_case,
    avoid_filter=avoid_filter,
    taxon_filter=taxon_filter,
    parquet=p
  )

def query_distinct(
  p: Path,
  dbssert: Path,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]]
) -> pl.DataFrame:
  # ? Query Database For Distinct Terms Only
  try:
    with duckdb.connect(dbssert) as conn:
      query: str = query_builder(p, prioritize, avoid, taxon)
      results: pl.DataFrame = conn.execute(query).pl()

      results = results.sort(["term", "PR", "NLP_LEVEL"])
      results = results.unique(subset=["term", "CURIE"], keep="first")
      return results

  finally:
    p.unlink(missing_ok=True)

def version4(
  df: pl.DataFrame,
  col: str,
  dbssert: Path,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]],
  tag: str = " one"
) -> pl.DataFrame:
  # ? Case Dependant, Provenance Rich Name Entity Recognition
  l0: str = col
  l1: str = add(l0, tag)

  terms: pl.DataFrame = distinct(df, l0, l1)
  p: Path = to_temp(terms)
  matches: pl.DataFrame = query_distinct(p, dbssert, taxon, prioritize, avoid)
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

  result = result.select(pl.exclude(r"^(CURIE|PREFERRED_NAME|CATEGORY_NAME|TAXON_ID|SOURCE_NAME|SOURCE_VERSION|NLP_LEVEL|PR)( l1)?$"))
  result = result.select(pl.exclude(add(col, " one")))
  result = result.with_columns(pl.col(add(col, " taxon")).replace("NCBITaxon:0", None))
  return result
