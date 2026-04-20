# Datassert

Datassert is a high-performance CLI for building a DuckDB-backed assertion store from NCATS Translator BABEL export files, with a focus on fast local builds and simple command-driven workflows. It produces the entity-resolution database used by Tablassert, containing biological synonyms, CURIEs, Biolink categories, taxon IDs, and source provenance, enabling `resolve()` to map free-text strings to standardized identifiers.

## Installation

```bash
# Install CLI from GitHub
go install github.com/SkyeAv/datassert@latest

# Verify install
datassert --help
```

## Build Command

```bash
# Build a Datassert database (downloads BABEL data automatically)
datassert build
```

The build command automatically downloads BABEL exports from RENCI (`https://stars.renci.org/var/babel_outputs`), processes them, and produces sharded DuckDB databases.

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--skip-downloads` / `-s` | No | `false` | Skip the BABEL download phase (use previously downloaded files) |
| `--use-existing-parquets` / `-p` | No | `false` | Use existing Parquet files to rebuild DuckDB databases |

### Data Pipeline

1. **Download** — BABEL class and synonym files are downloaded from RENCI and split into LZ4-compressed NDJSON chunks under `./datassert/downloads/`.
2. **Lookup** — Class files (`*.ndjson.lz4`) are read to build an in-memory equivalent-identifier lookup.
3. **Parquet Staging** — Synonym files are processed with the lookup, quality-controlled, and written as sharded Parquet files to `./datassert/parquets/`.
4. **DuckDB Generation** — Parquet files are loaded into 12 sharded DuckDB databases under `./datassert/data/`.

### Examples

```bash
# Full build (download, process, and generate databases)
datassert build

# Skip downloads if BABEL files were already fetched
datassert build --skip-downloads

# Rebuild DuckDB databases from existing Parquet files
datassert build --use-existing-parquets
```

### Runtime Behavior

- Displays progress bars for download, class lookup, synonym processing, and DuckDB build phases.
- Uses 90% of available CPUs for concurrent processing.
- Downloads are retried up to 3 times on failure with a 10-second backoff.
- All working files are stored under `./datassert/`.

## Output Artifacts

- 12 sharded DuckDB databases are written to `./datassert/data/{0..11}.duckdb`.
- Each shard contains `SOURCES`, `CATEGORIES`, `CURIES`, and `SYNONYMS` tables, deduplicated, sorted, and indexed for query performance.
- Staging Parquet files are written to `./datassert/parquets/{0..11}/`.

Terms are routed to shards deterministically via `xxhash64(term) % 12`, so a given string always hits the same shard.

### Schema

Each shard contains four tables:

| Table | Key Columns | Description |
|-------|-------------|-------------|
| `SYNONYMS` | `SYNONYM`, `CURIE_ID`, `SOURCE_ID` | Text synonym → CURIE mapping |
| `CURIES` | `CURIE_ID`, `CURIE`, `PREFERRED_NAME`, `TAXON_ID`, `CATEGORY_ID` | Canonical identifiers and preferred names |
| `CATEGORIES` | `CATEGORY_ID`, `CATEGORY_NAME` | Biolink category names |
| `SOURCES` | `SOURCE_ID`, `SOURCE_NAME`, `SOURCE_VERSION` | Source database and version provenance |

## Usage in Graph Config

The `datassert:` field in a GC2 graph configuration points to the directory containing the shards. Tablassert opens all 12 shards at startup and passes the connections to `resolve()`.

```yaml
# graph-config.yaml (GC2)
syntax: GC2
name: my-graph
version: "1.0"
datassert: /path/to/datassert/   # directory containing data/0..11.duckdb
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
    for i in range(12)
]
```

See [Entity Resolution](api/fullmap.md) for the full `resolve()` API.
