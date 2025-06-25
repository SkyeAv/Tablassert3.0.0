from sqlite_utils import Database
from typing import Any, Optional
from collections import Counter
from functools import lru_cache
from diskcache import Cache
from pathlib import Path

# all of these are hardcoded because of how hard everything got with circular imports
fullmapcache: Cache = Cache("/tablassert/cache/.fullmap".upper(), max_size=1e10)
babelcache: Cache = Cache("/tablassert/cache/.babel".upper(), max_size=1e10)
kg2cache: Cache = Cache("/tablassert/cache/.kg2".upper(), max_size=1e9)
metadatacache: Cache = Cache("/tablassert/cache/.pubmed_metadata".upper(), max_size=1e6)
captionscache: Cache = Cache("/tablassert/cache/.pubmed_captions".upper(), max_size=1e6)

def new_connection(sqlitepath: Path) -> Database:
    return Database(sqlitepath.as_posix()).enable_wal()

# pubmed lookups aren't frequent enough to justify a combined cache

@metadatacache.memorize()
def pubmed_metadata(db: Database, article_curie: str) -> Optional[dict[str, object]]:
    sql: str = """
    FROM
    """
    return db.query(sql)

@captionscache.memorize()
def file_caption(db: Database, article_curie: str, filename: str) -> Optional[str]:
    sql: str = """
    FROM
    """
    return db.query(sql)

@lru_cache(maxsize=1024)
def cached_babel_lookup(db: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str) -> Optional[]:
    return babel_lookup(db, unprocessed_input, prioritize, avoid, taxon)

@babelcache.memorize()
def babel_lookup(db: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str) -> Optional[]:
    sql: str = """
    FROM
    """
    return db.query(sql)

@lru_cache(maxsize=512)
def cached_kg2_lookup(db: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    return kg2_lookup(db, unprocessed_input, prioritize, avoid, taxon)

@kg2cache.memorize()
def kg2_lookup(db: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    sql: str = """
    FROM
    """
    return db.query(sql)

# patch lookups aren't frequent enough to justify a combined cache

@lru_cache(maxsize=16)
def override_lookup(db: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    sql: str = """
    FROM
    """
    return db.query(sql)

@lru_cache(maxsize=32)
def supplement_lookup(db: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    sql: str = """
    FROM
    """
    return db.query(sql)

@lru_cache(maxsize=2048)
def cached_fullmap3(babel: Database, kg2: Database, mapping_patch: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str, column_context: Counter) -> Optional[]:
    return fullmap3(babel, kg2, mapping_patch, unprocessed_input, prioritize, avoid, taxon, column_context)

@fullmapcache.memorize()
def fullmap3(babel: Database, kg2: Database, mapping_patch: Database, unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str, column_context: Counter) -> Optional[]:
    return