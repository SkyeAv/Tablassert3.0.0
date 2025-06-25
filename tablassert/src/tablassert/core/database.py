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
def pubmed_metadata(article_curie: str) -> Optional[dict[str, object]]:
    sql: str = """
    FROM
    """
    return pubmed.query(sql)

@captionscache.memorize()
def file_caption(article_curie: str, filename: str) -> Optional[str]:
    sql: str = """
    FROM
    """
    return pubmed.query(sql)

# lru cache is 10-100x faster so I cache twice
@lru_cache(maxsize=1024)
def cached_babel_lookup(unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str) -> Optional[]:
    return babel_lookup(unprocessed_input, prioritize, avoid, taxon)

@babelcache.memorize()
def babel_lookup(unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str) -> Optional[]:
    sql: str = """
    FROM
    """
    return babel.query(sql)

@lru_cache(maxsize=512)
def cached_kg2_lookup(unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    return kg2_lookup(unprocessed_input, prioritize, avoid, taxon)

@kg2cache.memorize()
def kg2_lookup(unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    sql: str = """
    FROM
    """
    return kg2.query(sql)

# patch lookups aren't frequent enough to justify a combined cache

@lru_cache(maxsize=16)
def override_lookup(unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    sql: str = """
    FROM
    """
    return mapping_patch.query(sql)

@lru_cache(maxsize=32)
def supplement_lookup(unprocessed_input: str, prioritize: set[str], avoid: set[str]) -> Optional[]:
    sql: str = """
    FROM
    """
    return mapping_patch.query(sql)

# counter is global because it's not hashable for the caches
column_context: Counter = Counter()

def reset_column_context() -> None:
    global column_context
    column_context = Counter()

# databases aren't hashable so I activate them all globally
def activate_sqlites() -> None:
    global pubmed
    pubmed: Database = new_connection()
    global babel
    babel: Database = new_connection()
    global kg2
    kg2: Database = new_connection()
    global mapping_patch
    mapping_patch: Database = new_connection()

@lru_cache(maxsize=2048)
def cached_fullmap3(unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str) -> Optional[]:
    return fullmap3(unprocessed_input, prioritize, avoid, taxon, column_context)

@fullmapcache.memorize()
def fullmap3(unprocessed_input: str, prioritize: set[str], avoid: set[str], taxon: str) -> Optional[]:
    return