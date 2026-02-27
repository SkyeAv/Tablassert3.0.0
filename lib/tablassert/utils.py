from diskcache import Cache
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid3
from uuid import UUID
import polars as pl
import xxhash

STORE: Path = Path("./storessert")
STORE.mkdir(parents=True, exist_ok=True)

DISKCACHE: object = Cache(
  "./cachessert",
  size_limit=100_000_000, # ~100MB
  eviction_policy="least-recently-used"
)

def mkhash(x: Any) -> str:
  b: bytes = str(x).encode("utf-8")
  return xxhash.xx64(b).hexdigest()

def samphash(df: pl.DataFrame, n: int = 20) -> str:
  # ? Hash Of Sampled DataFrame For Tempfile Naming
  # ! Requires eager DataFrame input - call .collect() before passing LazyFrame
  samp: pl.DataFrame = df.sample(min(n, df.height))
  return mkhash(samp.to_init_repr())

@cache
def basespace(domain: str) -> UUID:
  namespace: UUID = UUID("00000000-0000-0000-0000-000000000000")
  return uuid3(namespace, domain)

def namespace_uuid(domain: Any, *values: list[Any]) -> str:
  domain = str(domain)
  values = [str(x) for x in values if x] # pyright: ignore
  domainspace: UUID = basespace(domain)
  joined: str = "\t".join(values) # pyright: ignore
  return str(uuid3(domainspace, joined))
