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
| `--threads` | No | `None` (single-threaded) | Worker threads for staging/writing |
| `--write-batch-size` | No | `50000` | Row batch size for staged writes |

### Data Pipeline

1. **Download** — BABEL class and synonym files are downloaded from RENCI (`https://stars.renci.org/var/babel_outputs`) into `--cache`.
2. **Equivalents** — Class files are staged into an `equivalents` table used to resolve equivalent-identifier groups.
3. **Synonyms** — Synonym files are staged into a `term_records` table alongside per-term metadata.
4. **Finalize** — Staged tables are reduced into a single `records` table keyed by normalized term, written to a temporary redb file, then atomically renamed to `--output`.

### Examples

```bash
# Full build (download, process, and generate the database)
tablassert build-fullmap

# Custom output location and BABEL version
tablassert build-fullmap --output /data/fullmap/fullmap.redb --version 2025sep1

# Tune concurrency and write batching for large builds
tablassert build-fullmap --threads 8 --write-batch-size 100000
```

## Output Artifact

A single redb file (default `./fullmap/data/fullmap.redb`) containing four tables (see `rust/src/fullmap.rs`):

| Table | Description |
|-------|-------------|
| `records` | Normalized term → serialized list of `FullmapRecord` (CURIE, preferred name, category, taxon ID, source name/version) |
| `meta` | Schema version tag (`tablassert.fullmap.v1`) and the BABEL `source_version` used to build the file |
| `equivalents` | Staged equivalent-identifier groups from BABEL class files |
| `term_records` | Staged per-term synonym records prior to finalization |

Lookups (`lookup_fullmap_terms`) check the `meta` schema tag before reading `records`; a mismatched or missing tag raises rather than silently reading incompatible data — there is no automatic schema migration, so a schema bump requires rebuilding via `tablassert build-fullmap`.

## Usage in Graph Config

The `fullmap:` field in a graph configuration points at either the redb file directly or a base directory. Tablassert resolves it via `fullmap_db_path()`:

- If the path is a file or already ends in `.redb`, it's used as-is.
- Else if `<path>/fullmap.redb` exists, that's used.
- Else it falls back to `<path>/data/fullmap.redb` (the `build-fullmap` default layout).

```yaml
# graph-config.yaml (GC3)
syntax: GC3
name: my-graph
version: "1.0"
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
