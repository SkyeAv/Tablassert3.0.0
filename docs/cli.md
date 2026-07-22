# CLI Reference

Tablassert provides three commands.

## version

Display the current Tablassert package version.

### Synopsis

```bash
tablassert --version
```

### Example

```bash
tablassert --version
```

### Description

Prints the installed Tablassert version to stdout and exits. This is a flag on the main `tablassert` command, not a subcommand.

---

## build-graph

Build a knowledge graph from a YAML configuration file.

### Synopsis

```bash
tablassert build-graph <graph_configuration_file> [--release] [--qc] [--log]
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `graph_configuration_file` | Path | Yes | Knowledge Graph Configuration -- See Docs |
| `--release`, `-r` | Flag | No | Emit a slim, significant-only graph (drops `biolink:not_significant` edges before resolution) |
| `--qc`, `-q` | Flag | No | Run the QC audit stage (exact → fuzzy → BioBERT) on resolved node columns |
| `--log`, `-l` | Flag | No | Enable verbose per-section logging |

### Example

```bash
tablassert build-graph /path/to/MOKGV6.yaml --qc --log
```

### Description

This command runs the full extraction pipeline from a graph configuration file. It loads table configurations, reads each table's source file from disk, applies transformations, resolves entities through fullmap, optionally validates mappings with the QC pipeline (exact → fuzzy → BioBERT) when `--qc` is passed, and compiles subgraphs into KGX-compliant NDJSON files plus a Resource Ingest Guide (RIG).

The process executes in parallel stages with a three-row live progress block (logs print above the live block):

```
✓ Stage 1 · LOADING TABLES · 0:00:01.20
✓ Stage 2 · EXTRACTING SECTIONS · 0:00:00.80
Stage 3 of 6  ·  BUILDING TCODE   tablassert v8.0.0
TCODE ━━━━━━━━━━━━━━━━━━━━━━━━━━─── 12/47 · 0:00:42 · 0:02:11
  ↳ my_table.yaml · abc123de
```

- **Row 1** — the current stage header (`Stage N of 6 · NAME`)
- **Row 2** — the section bar: label, bar, count, elapsed, ETA
- **Row 3** — the in-flight item detail (config stem + 8-char hash)

The six stages are:

- Loading Tables
- Extracting Sections
- Building TCode
- Collecting Instructions
- Building Subgraphs
- Compiling Graph

For Building Subgraphs (the longest stage), the detail line also shows the current per-section phase as `compile_subgraph` reduces over the op-list:

```
Stage 5 of 6  ·  BUILDING SUBGRAPHS   tablassert v8.0.0
SUBGRAPH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12/47 · 0:01:24 · 0:03:02
  ↳ my_table.yaml · abc123de  →  resolve
```

Phases cycle through `load`, `filter`, `clean`, `encode`, `resolve`, `qc`, `edge`, `provenance`, `significance`, `finalize`, `write` per section. Each completed stage prints a green `✓ Stage N · NAME · elapsed` line above the live block.

Final output files are written to the current working directory as:
- `{name}_{version}.nodes.ndjson` - Node file (entities)
- `{name}_{version}.edges.ndjson` - Edge file (relationships)
- `{name}_{version}.RIG.yaml` - Resource Ingest Guide (source, provenance, and target metadata for the graph)

Intermediate parquet artifacts are written to `.tablassert/store/` during section processing.

See [Graph Configuration](configuration/graph.md) for details on the YAML schema.

---

## validate-table

Validate section syntax from a YAML configuration file.

### Synopsis

```bash
tablassert validate-table <table_configuration_file>
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `table_configuration_file` | Path | Yes | Table Configuration -- See Docs |

### Example

```bash
tablassert validate-table /path/to/table-config.yaml
```

### Description

This command validates a TC4 YAML configuration file without running the full extraction pipeline. It loads the file, extracts sections, and validates each section against the schema using Pydantic models. The command exits with a non-zero status if schema errors are detected, making it useful for CI/CD pipelines and pre-commit hooks.

Use this for:
- Quick syntax validation during development
- Pre-flight checks in CI/CD pipelines
- Verifying configuration changes before running expensive graph builds

See [Table Configuration](configuration/table.md) for details on the YAML schema.

---

## build-fullmap

Build an embedded fullmap redb database from BABEL export files.

### Synopsis

```bash
tablassert build-fullmap [--output <path>] [--cache <path>] [--version <version>] [--threads <n>] [--write-batch-size <n>]
```

### Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--output` | Path | No | `./fullmap/data/fullmap.redb` | Path to write the built redb file |
| `--cache` | Path | No | `./fullmap/downloads/fullmap` | Directory for downloaded BABEL files |
| `--version` | str | No | current BABEL release | BABEL release version to fetch |
| `--threads` | int | No | `None` | Worker threads for the parallel build (defaults to ~90% of available CPUs when unset) |
| `--write-batch-size` | int | No | `50000` | Row batch size for database writes |

### Example

```bash
tablassert build-fullmap --output /data/fullmap/fullmap.redb
```

### Description

This command downloads BABEL class and synonym files from RENCI and builds a single embedded `fullmap.redb` file (via an in-memory parallel build) used for entity resolution during `build-graph`. See [Fullmap](fullmap.md) for the full data pipeline, output schema, and graph-config usage.

---

## Examples

### Check Version

```bash
tablassert --version
```

### Build Knowledge Graph

```bash
tablassert build-graph my-graph.yaml
```

### Validate Table Configuration

```bash
tablassert validate-table table-config.yaml
```

### Build Fullmap Database

```bash
tablassert build-fullmap
```

## Workflow

1. **Create table configuration** - Define data sources and transformations
2. **Create graph configuration** - Define output name, table configs, databases
3. **Validate table config** - `tablassert validate-table table.yaml`
4. **Build knowledge graph** - `tablassert build-graph graph.yaml`
5. **Process executes:**
   - Reads each table's source file from disk
   - Applies transformations to each table
   - Resolves entities using fullmap
   - Validates mappings with the QC pipeline (when `--qc` is passed)
   - Aggregates subgraphs into NDJSON and writes the RIG

## Next Steps

- **[Tutorial](tutorial.md)** - Complete example walkthrough
- **[Configuration Guide](configuration/graph.md)** - YAML configuration reference
- **[Fullmap](fullmap.md)** - Entity-resolution database build and schema
