# Graph Configuration Reference

Graph configurations orchestrate the processing of multiple table configurations into a single knowledge graph output.

## Purpose

A graph configuration file specifies:
- Output knowledge graph name, version, and description
- List of table configurations to process
- Database location for entity resolution
- Resource Ingest Guide (RIG) metadata (contributions and UI explanation)

QC auditing and verbose logging are controlled at build time via the `build-graph --qc` and `build-graph --log` flags — they are **not** graph-config fields.

## Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Knowledge graph name (used in output filenames and the RIG) |
| `version` | String | Knowledge graph version (used in output filenames) |
| `description` | String | Source-scope description written into the generated RIG |
| `tables` | List[Path] | Paths to table configuration YAML files |
| `fullmap` | Path | Path to the fullmap redb file, or a base directory containing it |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `contributions` | List[String] | RIG contribution statements for graph provenance. Defaults to `["Tablassert: KGX and RIG generation"]` |
| `ui_explanation` | String | RIG explanation applied to generated edge-type metadata. Defaults to a built-in description of how Tablassert transforms source records into Biolink associations |

### Field Details

**`name: string`**

Output knowledge graph name. Used as the prefix for the NDJSON and RIG output files.

Example: `name: MULTIOMICS_KG` produces `MULTIOMICS_KG_{version}.nodes.ndjson`

**`version: string`**

Output knowledge graph version. Used as the suffix for output files.

Common values: `"1.0.0"`, `"UNSTABLE"`, `"BETA"`

**`description: string`**

A short human-readable description of the graph's source scope. Written into the generated Resource Ingest Guide (RIG). Required.

**`contributions: list[string]`**

Contribution statements recorded in the RIG `provenance_info`. Defaults to `["Tablassert: KGX and RIG generation"]` when omitted.

**`ui_explanation: string`**

Human-readable explanation applied to each generated edge type's metadata in the RIG. Defaults to a built-in description when omitted.

**`tables: list[path]`**

List of table configuration file paths. Can be absolute or relative to the current working directory (paths are resolved against the process CWD, not the graph-config file location).

Each table config defines:
- Data source (Excel/CSV/TSV)
- Entity resolution rules
- Provenance information
- Edge annotations

See [Table Configuration](table.md) for details.

**`fullmap: path`**

Path to the [fullmap](../fullmap.md) redb file for entity resolution, or a base directory containing it. Tablassert resolves the actual file via `fullmap_db_path()`. This database contains:
- Synonym mappings (text → CURIE)
- Biolink categories
- Taxonomic information
- Source provenance (which database provided the mapping)

See [Fullmap](../fullmap.md) for build commands and database schema.

## Path Resolution

Paths can be:
- **Absolute:** `/home/user/data/fullmap`
- **Relative to the current working directory:** `./tables/table1.yaml` (note: paths are resolved against the process CWD, not the graph-config file location — there is no config-relative resolver)

## Minimal Example

```yaml
name: MY_GRAPH
version: 1.0.0
description: Knowledge graph built from configured tabular source data.
tables:
  - ./my-table.yaml
fullmap: /data/fullmap
```

## Multi-Table Example

```yaml
name: MULTIOMICS_KG
version: UNSTABLE
description: Multi-omics knowledge graph integrating gene-disease, drug-target, and protein-interaction tables.
tables:
  - /configs/gene-disease-associations.yaml
  - /configs/drug-targets.yaml
  - /configs/protein-interactions.yaml
fullmap: /databases/fullmap
```

## Processing Flow

When you run `tablassert build-graph graph.yaml`:

1. **Load graph configuration** - Parse YAML, validate schema
2. **Load table configurations** - Parse each YAML in `tables`
3. **Extract sections** - Expand templates into per-section `Tcode` instances
4. **Collect instructions (per section):**
   - Read the source file from disk (`source.local`)
   - Apply transformations and resolve entities using `fullmap`
   - Validate with the QC audit when `build-graph --qc` is passed
5. **Build subgraphs** - Compile each section's resolved data into a parquet file
6. **Compile graph** - Aggregate all subgraph parquets and export `{name}_{version}.nodes.ndjson` / `.edges.ndjson` / `.RIG.yaml`

## Output Files

Given this configuration:
```yaml
name: EXAMPLE_KG
version: 2.0.0
```

Produces:
- `EXAMPLE_KG_2.0.0.nodes.ndjson`
- `EXAMPLE_KG_2.0.0.edges.ndjson`
- `EXAMPLE_KG_2.0.0.RIG.yaml`

## Real-World Example

From MOKGV6.yaml:

```yaml
name: MULTIOMICS_KG
version: UNSTABLE
description: Multi-omics knowledge graph derived from the ALAMV6 tabular source.
tables:
  - /local_raid1/sgoetz/STORE/CONFIG/TABLASSERT/TABLE/V6/ALAMV6.yaml
fullmap: /local_raid1/sgoetz/CODE/FULLMAP/fullmap
```

This processes a single table configuration (ALAMV6.yaml) into a knowledge graph named `MULTIOMICS_KG_UNSTABLE`.

## Next Steps

- **[Table Configuration](table.md)** - Learn how to define table transformations
- **[Fullmap](../fullmap.md)** - Entity-resolution database build and schema
- **[Tutorial](../tutorial.md)** - Complete example walkthrough
