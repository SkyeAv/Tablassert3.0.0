from tablassert.enums import Categories
from typing import Optional
from pathlib import Path
from operator import add
import polars as pl
import duckdb

def version4(
  p: Path,
  col: str,
  dbssert: Path,
  taxon: Optional[str],
  prioritize: Optional[list[Categories]],
  avoid: Optional[list[Categories]],
  tag: str = " one"
) -> pl.DataFrame:
  # ? Case Dependant, Provenance Rich Name Entitiy Recognition
  try:
    with duckdb.connect(dbssert) as conn:
      l0: str = col
      l1: str = add(l0, tag)
      query: str = f"""
SELECT DISTINCT ON (RANKED."row number", RANKED.CURIE)
  RANKED.CURIE AS '{col}',
  RANKED.PREFERRED_NAME AS '{add(col, " name")}',
  'biolink:' || RANKED.CATEGORY_NAME AS '{add(col, " category")}',
  'NCBITaxon:' || RANKED.TAXON_ID AS '{add(col, " taxon")}',
  RANKED.SOURCE_NAME AS '{add(col, " source")}',
  RANKED.SOURCE_VERSION AS '{add(col, " source version")}',
  RANKED.NLP_LEVEL AS '{add(col, " nlp level")}',
  RANKED.SYNONYM AS '{add(col, " synonym")}',
  RANKED.* EXCLUDE (
    SYNONYM,
    CURIE,
    PREFERRED_NAME,
    CATEGORY_NAME,
    TAXON_ID, 
    SOURCE_NAME,
    SOURCE_VERSION,
    NLP_LEVEL,
    SOURCE_ID, 
    SOURCE_ID_1,
    CURIE_ID,
    CURIE_ID_1,
    CATEGORY_ID,
    CATEGORY_ID_1,
    PR,
    '{l0}',
    '{l1}'
  )
FROM (
  SELECT
    SY.*,
    SO.*,
    CU.*,
    CA.*,
    PA.*,
    CASE
      {f"WHEN CA.CATEGORY_NAME IN ({", ".join(f"'{x}'" for x in prioritize)}) THEN 1" if prioritize else "WHEN TRUE THEN 50"}
      ELSE 50
    END AS PR
  FROM SYNONYMS SY
  JOIN SOURCES SO
    ON SY.SOURCE_ID = SO.SOURCE_ID
  JOIN CURIES CU
    ON SY.CURIE_ID = CU.CURIE_ID
  JOIN CATEGORIES CA
    ON CU.CATEGORY_ID = CA.CATEGORY_ID
    {f"AND CA.CATEGORY_NAME NOT IN ({", ".join(f"'{x}'" for x in avoid)})" if avoid else ""}
  JOIN read_parquet('{p}') PA
    ON (PA."{l0}" = SY.SYNONYM OR PA."{l1}" = SY.SYNONYM)
  {f"WHERE CU.TAXON_ID = {taxon}" if taxon else ""}
) AS RANKED
ORDER BY (RANKED."row number", RANKED.CURIE, RANKED.PR);
"""
      df: pl.DataFrame = conn.execute(query).pl()
      print("PostFullmap", col, df.shape)
      return df
  finally:
    p.unlink()
