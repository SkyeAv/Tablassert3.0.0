# Graph Configuration Reference

Graph configurations orchestrate the processing of multiple table configurations into a single knowledge graph output.

## Purpose

A graph configuration file specifies:
- Output knowledge graph name and version
- Whether QC auditing runs during the build
- List of table configurations to process
- Database locations for entity resolution and provenance

## Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Knowledge graph name (used in output filename) |
| `version` | String | Knowledge graph version (used in output filename) |
| `tables` | List[Path] | Paths to table configuration YAML files |
| `datassert` | Path | Path to datassert directory containing DuckDB shards |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `syntax` | String | Configuration version. Defaults to `"GC2"` (also accepts `"TC3"`); overriding is not recommended. |
| `pubmed_db` | Path | Path to SQLite PubMed metadata database |
| `pmc_db` | Path | Path to SQLite PMC figure captions database |
| `log` | Boolean | Log unmatched entities and audit details during graph builds |
| `qc` | Boolean | Enable the QC audit stage during graph builds |

### Field Details

**`syntax: "GC2"`**

Configuration syntax version. Defaults to `"GC2"` (the type also accepts `"TC3"`); overriding the default is not recommended.

**`name: string`**

Output knowledge graph name. Used as prefix for NDJSON files.

Example: `name: MULTIOMICS_KG` produces `MULTIOMICS_KG_{version}.nodes.ndjson`

**`version: string`**

Output knowledge graph version. Used as suffix for NDJSON files.

Common values: `"1.0.0"`, `"UNSTABLE"`, `"BETA"`

**`log: bool = false`**

When `true`, Tablassert logs unmatched entities and audit details during the graph build. When `false`, unmatched entities are silently filtered.

**`qc: bool = false`**

When `true`, Tablassert runs the QC audit stage after entity resolution for node-like columns. When `false`, the build skips QC entirely.

This field only controls whether QC runs. Install `tablassert[qc]` or `tablassert[qc-cuda]` if you plan to enable it.

**`tables: list[path]`**

List of table configuration file paths. Can be absolute or relative to the current working directory (paths are resolved against the process CWD, not the graph-config file location).

Each table config defines:
- Data source (Excel/CSV/TSV)
- Entity resolution rules
- Provenance information
- Edge annotations

See [Table Configuration](table.md) for details.

**`datassert: path`**

Path to the [datassert](../datassert.md) directory for entity resolution. Tablassert opens 10 shard files from `datassert/data/{0..9}.duckdb`. This database contains:
- Synonym mappings (text → CURIE)
- Biolink categories
- Taxonomic information
- Source provenance (which database provided the mapping)

See [Datassert](../datassert.md) for installation, build commands, and database schema.

**`pubmed_db: path`**

Optional path to SQLite database with PubMed metadata:
- Authors
- Journal information
- Publication dates

When provided, this enriches edges with publication metadata.

**`pmc_db: path`**

Optional path to SQLite database with PubMed Central figure captions.

When provided, this is used when provenance specifies PMC publications.

## Path Resolution

Paths can be:
- **Absolute:** `/home/user/data/datassert`
- **Relative to the current working directory:** `./tables/table1.yaml` (note: paths are resolved against the process CWD, not the graph-config file location — there is no config-relative resolver)

## Minimal Example

```yaml
syntax: GC2
name: MY_GRAPH
version: 1.0.0
log: true
qc: true
tables:
  - ./my-table.yaml
datassert: /data/datassert
pubmed_db: /data/PubMed.db
pmc_db: /data/PMCSuppCaptions.db
```

## Multi-Table Example

```yaml
syntax: GC2
name: MULTIOMICS_KG
version: UNSTABLE
tables:
  - /configs/gene-disease-associations.yaml
  - /configs/drug-targets.yaml
  - /configs/protein-interactions.yaml
datassert: /databases/datassert
pubmed_db: /databases/PubMed.db
pmc_db: /databases/PMCSuppCaptions.db
```

## Processing Flow

When you run `tablassert build graph.yaml`:

1. **Load graph configuration** - Parse YAML, validate schema
2. **Load table configurations** - Parse each YAML in `tables`
3. **Extract sections** - Expand templates into per-section `Tcode` instances
4. **Collect instructions (per section):**
   - Download source file (if URL specified)
   - Apply transformations and resolve entities using `datassert`
   - Validate with the QC audit when `qc: true`
   - Enrich with provenance: query `pubmed_db` (publication metadata) and `pmc_db` (captions) when configured
5. **Build subgraphs** - Compile each section's resolved data into a parquet file
6. **Compile graph** - Aggregate all subgraph parquets and export `{name}_{version}.nodes.ndjson` / `.edges.ndjson`

> Note: provenance enrichment (`with_publication`/`with_captions`) and QC both run during the per-section Collect Instructions stage, *before* subgraphs are built — not as a separate post-aggregation step.

## Output Files

Given this configuration:
```yaml
name: EXAMPLE_KG
version: 2.0.0
```

Produces:
- `EXAMPLE_KG_2.0.0.nodes.ndjson`
- `EXAMPLE_KG_2.0.0.edges.ndjson`

## Real-World Example

From MOKGV6.yaml:

```yaml
syntax: GC2
name: MULTIOMICS_KG
version: UNSTABLE
tables:
  - /local_raid1/sgoetz/STORE/CONFIG/TABLASSERT/TABLE/V6/ALAMV6.yaml
datassert: /local_raid1/sgoetz/CODE/DATASSERT/datassert
pubmed_db: /local_raid1/sgoetz/DBSTORE/PUBMED/PubMed.db
pmc_db: /local_raid1/sgoetz/DBSTORE/CAPTIONS/PMCSuppCaptions.db
```

This processes a single table configuration (ALAMV6.yaml) into a knowledge graph named `MULTIOMICS_KG_UNSTABLE`.

## Next Steps

- **[Table Configuration](table.md)** - Learn how to define table transformations
- **[Datassert](../datassert.md)** - Entity-resolution database installation and build
- **[Tutorial](../tutorial.md)** - Complete example walkthrough
