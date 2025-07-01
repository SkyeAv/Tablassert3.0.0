from diskcache import Cache
from loguru import logger
from typing import Any
import polars as pl

# diskcache setup (sqlite caches)
fullmap3cache: Cache = Cache("TABLASSERT/CACHE/FULLMAP3", max_size=1e9)
babelcache: Cache = Cache("TABLASSERT/CACHE/BABEL", max_size=1e10)
kg2cache: Cache = Cache("TABLASSERT/CACHE/KG2", max_size=1e9)

def connect(sqlitepath: str) -> Database:
    db: Database = Database(sqlitepath)
    db.enable_wal()
    return db

EXCEL_ENGINE: str = "xlsx2csv"

def dataframing(subsectionmodel: dict[str, Any], graphmodel: dict[str, dict[str, Any]]) -> pl.DataFrame:
    