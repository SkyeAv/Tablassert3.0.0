# Fullmap

Fullmap is Tablassert's embedded entity-resolution database: a single [redb](https://github.com/cberner/redb) file containing biological synonyms, CURIEs, Biolink categories, taxon IDs, and source provenance, built from NCATS Translator BABEL export files. It powers `resolve()` / `resolve_many()`, mapping free-text strings to standardized identifiers.

Unlike the DuckDB-shard architecture used in earlier versions (built by a separate external `datassert` Go CLI), Fullmap is built entirely in-process by Tablassert's own Rust extension — no external tool or install step is required.

## Build Command

```bash
# Build a fullmap database (downloads BABEL data automatically)
tablassert build-fullmap
```

### Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--output` | No | `./fullmap/data/fullmap.redb` | Path to write the built redb file |
| `--cache` | No | `./fullmap/downloads/fullmap` | Directory for downloaded BABEL files |
| `--version` | No | current BABEL release (see `cli.py`) | BABEL release version to fetch |
| `--threads` | No | `None` (~90% of available CPUs) | Worker threads for the parallel build |

### Data Pipeline

The build is an in-memory, parallel pipeline (rayon) executed by the Rust extension — there is no staging database and no temporary-file copy:

1. **Download** — BABEL class and synonym files are downloaded from RENCI (`https://stars.renci.org/var/babel_outputs`) into `--cache` (resumable, range-request downloads; cached files are reused).
2. **Equivalents map** — Class files are parsed in parallel into an in-memory map of each primary CURIE to its equivalent identifiers.
3. **Synonym pass** — Synonym files are parsed in parallel. For each row, the build collects dimension sets (CURIE prefixes, Biolink categories, sources), assigns compact integer CURIE IDs, and accumulates normalized-term → (CURIE, source) postings in a sharded in-memory map.
4. **Write** — A single redb write transaction emits the dimension tables (`prefixes`, `categories`, `sources`), the `curies` table, and the `meta` schema tag, followed by the `records` table (normalized term → serialized postings) written to `--output`.

### Examples

```bash
# Full build (download, process, and generate the database)
tablassert build-fullmap

# Custom output location and BABEL version
tablassert build-fullmap --output /data/fullmap/fullmap.redb --version 2025sep1

# Tune concurrency for large builds
tablassert build-fullmap --threads 8
```

## Output Artifact

A single redb file (default `./fullmap/data/fullmap.redb`) containing six tables (see `rust/src/fullmap.rs`):

| Table | Description |
|-------|-------------|
| `records` | Normalized term (xxhash `u64`) → serialized list of resolution postings (CURIE id, source id) |
| `prefixes` | Compact `u16` id → CURIE prefix string |
| `categories` | Compact `u16` id → Biolink category string |
| `sources` | Compact `u8` id → source metadata (name/version) |
| `curies` | Compact `u32` id → CURIE record (CURIE, preferred name, category, taxon, source) |
| `meta` | Schema version tag (`tablassert.fullmap.v3`) and the BABEL `source_version` used to build the file |

Lookups (`lookup_fullmap_terms`) check the `meta` schema tag before reading `records`; a mismatched or missing tag raises rather than silently reading incompatible data. Databases built under the older `v1`/`v2` schemas are rejected — there is no automatic schema migration, so a schema bump requires rebuilding via `tablassert build-fullmap`.

## Usage in Graph Config

The `fullmap:` field in a graph configuration points at either the redb file directly or a base directory. Tablassert resolves it via `fullmap_db_path()`:

- If the path is a file or already ends in `.redb`, it's used as-is.
- Else if `<path>/fullmap.redb` exists, that's used.
- Else it falls back to `<path>/data/fullmap.redb` (the `build-fullmap` default layout).

```yaml
# graph-config.yaml
name: my-graph
version: "1.0"
description: Example graph backed by a fullmap entity-resolution database.
fullmap: /path/to/fullmap/   # directory containing data/fullmap.redb, or a direct .redb file
tables:
  - ./TABLE/my-table.yaml
```

## Programmatic Usage

When calling `resolve_many()` directly, pass the fullmap path (file or base directory) as the `fullmap` argument:

```python
from pathlib import Path
from tablassert.lib import resolve_many

results = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1"],
    fullmap=Path("/path/to/fullmap"),
    taxon="9606",
)
```

See [Entity Resolution](api/fullmap.md) for the full `resolve()` API.
