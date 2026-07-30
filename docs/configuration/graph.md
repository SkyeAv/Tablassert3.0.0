# Graph Configuration Reference

Graph configurations orchestrate one or more [table configurations](table.md) into a single knowledge-graph build — author one whenever you run `tablassert build-kg` to produce KGX output.

## Purpose

A graph configuration file specifies:
- Output knowledge graph name, version, and description
- List of table configurations to process
- Database location for entity resolution
- Resource Ingest Guide (RIG) metadata (contributions and UI explanation)

QC auditing and verbose logging are controlled at build time via the `build-kg --qc` and `build-kg --log` flags — they are **not** graph-config fields.

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
| `infores` | String | Graph-level `infores:` CURIE emitted as the default Biolink `primary_knowledge_source` and RIG `source_info.infores_id`. Defaults to `infores:<kebab-name>` derived from `name` |

### Field Details

Notes beyond the tables above:

- **`version`** — common values: `"1.0.0"`, `"UNSTABLE"`, `"BETA"`.
- **`infores`** — must start with `infores:`; a name like `MULTIOMICS_KG` derives `infores:multiomics-kg`. Set it when the graph's Translator information resource differs from the output name, or a non-PMC/PMID source KG needs a stable manually-curated infores.
- **`tables`** — paths are absolute or relative to the process CWD (not the graph-config file location; there is no config-relative resolver). See [Table Configuration](table.md).
- **`fullmap`** — a redb file or a base directory containing it, resolved via `fullmap_db_path()`. See [Fullmap](../fullmap.md) for build commands and schema.

## Minimal Example

```yaml
name: MY_GRAPH
version: 1.0.0
description: Knowledge graph built from configured tabular source data.
infores: infores:my-graph
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

When you run `tablassert build-kg graph.yaml`:

1. **Load graph configuration** - Parse YAML, validate schema
2. **Load table configurations** - Parse each YAML in `tables`
3. **Extract sections** - Expand templates into per-section `Tcode` instances
4. **Collect instructions (per section):**
   - Read the source file from disk (`source.local`)
   - Apply transformations and resolve entities using `fullmap`
   - Validate with the QC audit when `build-kg --qc` is passed
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
