from tablassert.enums import Categories
from typing import Optional
from pathlib import Path
from operator import add
import polars as pl
import duckdb

def distinct(df: pl.DataFrame, l0: str, l1: str) -> pl.DataFrame:
  # ? Extract Unique Terms From Two Text Normalization Columns
  t0: pl.DataFrame = df.select(pl.col(l0).alias("term")).unique()
  t0 = t0.with_columns(pl.lit(0).alias("nlp level"))

  t1: pl.DataFrame = df.select(pl.col(l1).alias("term")).unique()
  t1 = t1.with_columns(pl.lit(1).alias("nlp level"))

  terms: pl.DataFrame = pl.concat([t0, t1]).unique(subset=["term"])
  return terms.with_row_index("term id")

def query_builder(
  p: Path,
  l0: str,
  l1: str,
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
  terms: pl.DataFrame,
  dbssert: Path,
  l0: str,
  l1: str,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]]
) -> pl.DataFrame:
  # ? Query Database For Distinct Terms Only
  from tempfile import gettempdir
  from tablassert.utils import samphash

  tmp: Path = Path(gettempdir())
  p: Path = tmp / samphash(terms)
  p = p.with_suffix(".parquet")
  terms.write_parquet(p)

  try:
    with duckdb.connect(dbssert) as conn:
      query: str = query_builder(p, l0, l1, prioritize, avoid, taxon)
      results: pl.DataFrame = conn.execute(query).pl()

      # ? Deduplicate By Keeping Best Match Per Term
      results = results.sort(["term", "PR", "NLP_LEVEL"])
      results = results.unique(subset=["term", "CURIE"], keep="first")

      return results
  finally:
    p.unlink(missing_ok=True)

def version4(
  p: Path,
  col: str,
  dbssert: Path,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]],
  tag: str = " one"
) -> pl.DataFrame:
  # ? Case Dependant, Provenance Rich Name Entity Recognition
  try:
    l0: str = col
    l1: str = add(l0, tag)

    # ? Read Input Parquet
    df: pl.DataFrame = pl.read_parquet(p)

    # ? Extract Distinct Terms
    terms: pl.DataFrame = distinct(df, l0, l1)

    # ? Query Database For Distinct Terms Only
    matches: pl.DataFrame = query_distinct(
      terms,
      dbssert,
      l0,
      l1,
      taxon,
      prioritize,
      avoid
    )

    # ? Join Matches Back To Original DataFrame
    # ? First Try l0 (Original Text)
    result: pl.DataFrame = df.join(
      matches.filter(pl.col("NLP_LEVEL").eq(0)),
      left_on=l0,
      right_on="term",
      how="left",
      suffix=" l0"
    )

    # ? Then Try l1 (Normalized Text) For Rows Without Matches
    l1_matches: pl.DataFrame = matches.filter(pl.col("NLP_LEVEL").eq(1))
    result = result.join(
      l1_matches,
      left_on=l1,
      right_on="term",
      how="left",
      suffix=" l1"
    )

    # ? Merge Results: Prefer l0, Fallback To l1
    result = result.with_columns([
      pl.when(pl.col("CURIE l0").is_not_null())
        .then(pl.col("CURIE l0"))
        .otherwise(pl.col("CURIE l1"))
        .alias(col),
      pl.when(pl.col("PREFERRED_NAME l0").is_not_null())
        .then(pl.col("PREFERRED_NAME l0"))
        .otherwise(pl.col("PREFERRED_NAME l1"))
        .alias(add(col, " name")),
      pl.when(pl.col("CATEGORY_NAME l0").is_not_null())
        .then(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME l0")))
        .otherwise(add(pl.lit("biolink:"), pl.col("CATEGORY_NAME l1")))
        .alias(add(col, " category")),
      pl.when(pl.col("TAXON_ID l0").is_not_null())
        .then(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID l0").cast(pl.String)))
        .otherwise(add(pl.lit("NCBITaxon:"), pl.col("TAXON_ID l1").cast(pl.String)))
        .alias(add(col, " taxon")),
      pl.when(pl.col("SOURCE_NAME l0").is_not_null())
        .then(pl.col("SOURCE_NAME l0"))
        .otherwise(pl.col("SOURCE_NAME l1"))
        .alias(add(col, " source")),
      pl.when(pl.col("SOURCE_VERSION l0").is_not_null())
        .then(pl.col("SOURCE_VERSION l0"))
        .otherwise(pl.col("SOURCE_VERSION l1"))
        .alias(add(col, " source version")),
      pl.when(pl.col("NLP_LEVEL l0").is_not_null())
        .then(pl.col("NLP_LEVEL l0"))
        .otherwise(pl.col("NLP_LEVEL l1"))
        .alias(add(col, " nlp level")),
      pl.when(pl.col("term l0").is_not_null())
        .then(pl.col("term l0"))
        .otherwise(pl.col("term l1"))
        .alias(add(col, " synonym"))
    ])

    # ? Clean Up Intermediate Columns
    result = result.select(pl.exclude(r"^(CURIE|PREFERRED_NAME|CATEGORY_NAME|TAXON_ID|SOURCE_NAME|SOURCE_VERSION|NLP_LEVEL|term|PR) (l0|l1)$"))

    # ? Replace NCBITaxon:0 With None
    result = result.with_columns(pl.col(add(col, " taxon")).replace("NCBITaxon:0", None))

    return result
  finally:
    p.unlink()
