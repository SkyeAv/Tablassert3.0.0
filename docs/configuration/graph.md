# Graph Configuration Reference

Graph configurations orchestrate the processing of multiple table configurations into a single knowledge graph output.

## Purpose

A graph configuration file specifies:
- Output knowledge graph name and version
- List of table configurations to process
- Database locations for entity resolution and provenance

## Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `syntax` | String | Configuration version (must be `"GC2"`) |
| `name` | String | Knowledge graph name (used in output filename) |
| `version` | String | Knowledge graph version (used in output filename) |
| `tables` | List[Path] | Paths to table configuration YAML files |
| `dbssert` | Path | Path to DuckDB entity resolution database |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `pubmed_db` | Path | Path to SQLite PubMed metadata database |
| `pmc_db` | Path | Path to SQLite PMC figure captions database |

### Field Details

**`syntax: "GC2"`**

Configuration syntax version. Must be `"GC2"`.

**`name: string`**

Output knowledge graph name. Used as prefix for NDJSON files.

Example: `name: MULTIOMICS_KG` produces `MULTIOMICS_KG_{version}.nodes.ndjson`

**`version: string`**

Output knowledge graph version. Used as suffix for NDJSON files.

Common values: `"1.0.0"`, `"UNSTABLE"`, `"BETA"`

**`tables: list[path]`**

List of table configuration file paths. Can be absolute or relative to graph config location.

Each table config defines:
- Data source (Excel/CSV/TSV)
- Entity resolution rules
- Provenance information
- Edge annotations

See [Table Configuration](table.md) for details.

**`dbssert: path`**

Path to DuckDB database for entity resolution. This database contains:
- Synonym mappings (text → CURIE)
- Biolink categories
- Taxonomic information
- Source provenance (which database provided the mapping)

**`pubmed_db: path`**

Optional path to SQLite database with PubMed metadata:
- MeSH terms
- Authors
- Journal information
- Publication dates

When provided, this enriches edges with MeSH annotations.

**`pmc_db: path`**

Optional path to SQLite database with PubMed Central figure captions.

When provided, this is used when provenance specifies PMC publications.

## Path Resolution

Paths can be:
- **Absolute:** `/home/user/data/dbssert.duckdb`
- **Relative to graph config:** `./tables/table1.yaml`
- **Relative to current directory:** `../configs/table.yaml`

## Minimal Example

```yaml
syntax: GC2
name: MY_GRAPH
version: 1.0.0
tables:
  - ./my-table.yaml
dbssert: /data/dbssert.duckdb
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
dbssert: /databases/dbssert.duckdb
pubmed_db: /databases/PubMed.db
pmc_db: /databases/PMCSuppCaptions.db
```

## Processing Flow

When you run `tablassert-cli build-knowledge-graph graph.yaml`:

1. **Load graph configuration** - Parse YAML, validate schema
2. **For each table in `tables`:**
   - Load table configuration
   - Download source file (if URL specified)
   - Apply transformations
   - Resolve entities using `dbssert`
   - Validate with QC pipeline
   - Create subgraph parquet file
3. **Aggregate subgraphs** - Merge all parquet files
4. **Add provenance (optional)** - Query `pubmed_db` and `pmc_db` for metadata when configured
5. **Export NDJSON** - Generate `{name}_{version}.nodes.ndjson` and `.edges.ndjson`

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
dbssert: /local_raid1/sgoetz/CODE/DBSSERT/dbssert.duckdb
pubmed_db: /local_raid1/sgoetz/DBSTORE/local_raid1/sgoetz/DBSTORE/PUBMED/PubMed.db
pmc_db: /local_raid1/sgoetz/DBSTORE/local_raid1/sgoetz/DBSTORE/CAPTIONS/PMCSuppCaptions.db
```

This processes a single table configuration (ALAMV6.yaml) into a knowledge graph named `MULTIOMICS_KG_UNSTABLE`.

## Next Steps

- **[Table Configuration](table.md)** - Learn how to define table transformations
- **[Tutorial](../tutorial.md)** - Complete example walkthrough
