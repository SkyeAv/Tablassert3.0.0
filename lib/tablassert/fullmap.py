from __future__ import annotations
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
  tag: str = "_one"
) -> pl.DataFrame:
  prepare: str = """
  PREPARE version4 AS
  JOIN LATERAL read_parquet($parquet) p
    ON [[]] = p.($zero)
    OR [[]] = p.($one)
  """
  one: str = add(col, tag)
  execute: str = f"""
  EXECUTE version4(parquet := {p}, zero := {col}, one := {one})
  """
  return 