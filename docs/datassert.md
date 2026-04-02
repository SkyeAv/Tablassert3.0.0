# Datassert

Datassert is the entity-resolution database used by Tablassert. It contains biological synonyms, CURIEs, Biolink categories, taxon IDs, and source provenance, enabling `resolve()` to map free-text strings to standardized identifiers.

## Installation

```bash
git clone https://github.com/SkyeAv/datassert
```

## Structure

Datassert is split into 16 DuckDB shard files for parallel querying:

```
datassert/
  data/
    0.duckdb
    1.duckdb
    ...
    15.duckdb
```

Terms are routed to shards deterministically via `xxhash64(term) % 16`, so a given string always hits the same shard.

### Schema

Each shard contains four tables:

| Table | Key Columns | Description |
|-------|-------------|-------------|
| `SYNONYMS` | `SYNONYM`, `CURIE_ID`, `SOURCE_ID` | Text synonym → CURIE mapping |
| `CURIES` | `CURIE_ID`, `CURIE`, `PREFERRED_NAME`, `TAXON_ID`, `CATEGORY_ID` | Canonical identifiers and preferred names |
| `CATEGORIES` | `CATEGORY_ID`, `CATEGORY_NAME` | Biolink category names |
| `SOURCES` | `SOURCE_ID`, `SOURCE_NAME`, `SOURCE_VERSION` | Source database and version provenance |

## Usage in Graph Config

The `datassert:` field in a GC2 graph configuration points to the directory containing the shards. Tablassert opens all 16 shards at startup and passes the connections to `resolve()`.

```yaml
# graph-config.yaml (GC2)
syntax: GC2
name: my-graph
version: "1.0"
datassert: /path/to/datassert/   # directory containing data/0..15.duckdb
tables:
  - ./TABLE/my-table.yaml
```

## Programmatic Usage

When calling `resolve()` directly, open the shard connections yourself:

```python
import duckdb
from tablassert.fullmap import resolve

datassert_dir = "/path/to/datassert"
conns = [
    duckdb.connect(f"{datassert_dir}/data/{i}.duckdb", read_only=True)
    for i in range(16)
]
```

See [Entity Resolution](api/fullmap.md) for the full `resolve()` API.
