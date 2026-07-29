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

## build-kg

Build a knowledge graph from a YAML configuration file.

### Synopsis

```bash
tablassert build-kg <configuration_file> [--release] [--qc] [--log] [--head] [--table-config] [--fullmap <path>]
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `configuration_file` | Path | Yes | Knowledge Graph Configuration -- See Docs |
| `--release`, `-r` | Flag | No | Emit a slim, significant-only graph (drops `biolink:not_significant` edges before resolution) |
| `--qc`, `-q` | Flag | No | Run the QC audit stage (exact → fuzzy → BioBERT) on resolved node columns |
| `--log`, `-l` | Flag | No | Enable verbose per-section logging |
| `--head`, `-hd` | Flag | No | Preview a random sample of up to 5 rows per section for a fast output shape/schema check (cached separately, never clobbers a full build) |
| `--table-config`, `-tc` | Flag | No | Treat the positional config as a TABLE (Section) YAML wrapped in a throwaway `TEMP_KG` graph, instead of a Graph YAML |
| `--fullmap`, `-f` | Path | No | Fullmap path for the throwaway `TEMP_KG` graph when `--table-config` is passed (default `./fullmap`) |

### Example

```bash
tablassert build-kg /path/to/MOKGV6.yaml --qc --log
```

Build or test a single table configuration without authoring a full graph config:

```bash
tablassert build-kg /path/to/table-config.yaml --table-config --fullmap /path/to/fullmap
```

### Description

This command runs the full extraction pipeline from a graph configuration file. It loads table configurations, reads each table's source file from disk, applies transformations, resolves entities through fullmap, optionally validates mappings with the QC pipeline (exact → fuzzy → BioBERT) when `--qc` is passed, and compiles subgraphs into KGX-compliant NDJSON files plus a Resource Ingest Guide (RIG).

By default the positional config is a Graph YAML. With `--table-config`/`-tc` it is instead a table (Section) YAML that Tablassert wraps in a throwaway graph (`name: TEMP_KG`, `version: 0.0.0`) so a single table config can be built or tested without authoring a full graph config; `--fullmap`/`-f` sets the fullmap path for that throwaway graph (default `./fullmap`), and the RIG `contributions`/`ui_explanation` fall back to the Graph model defaults.

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

## validate

Validate a graph or table YAML configuration file.

### Synopsis

```bash
tablassert validate <configuration_file>
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `configuration_file` (or `--configuration-file`, `-f`) | Path | Yes | Graph OR Table Configuration -- See Docs |

### Example

```bash
tablassert validate /path/to/table-config.yaml
tablassert validate /path/to/graph.yaml
```

### Description

This command validates a YAML configuration file without running the full extraction pipeline. It detects the config kind from the YAML: a mapping with a top-level `tables` key is treated as a **graph** config (the `Graph` model is validated, then every table file it references is validated), and anything else is treated as a **table** config (sections are extracted and validated against the schema using Pydantic models). The command exits with a non-zero status if schema errors are detected, making it useful for CI/CD pipelines and pre-commit hooks.

Use this for:
- Quick syntax validation during development
- Pre-flight checks in CI/CD pipelines
- Verifying configuration changes before running expensive graph builds

See [Table Configuration](configuration/table.md) and [Graph Configuration](configuration/graph.md) for details on the YAML schemas.

---

## gen-fullmap

Build an embedded fullmap redb database from BABEL export files.

### Synopsis

```bash
tablassert gen-fullmap [--output <path>] [--cache <path>] [--version <version>] [--threads <n>]
```

### Options

| Option | Type | Required | Default | Description |
|--------|------|----------|---------|-------------|
| `--output`, `-o` | Path | No | `./fullmap/data/fullmap.redb` | Path to write the built redb file |
| `--cache`, `-c` | Path | No | `./fullmap/downloads` | Directory for downloaded BABEL files (with `classes/` and `synonyms/` subdirectories) |
| `--version`, `-v` | str | No | `BABEL_VERSION` literal (`2026jul22`) | BABEL release snapshot date to fetch (a RENCI snapshot date stamp, not Tablassert's version) |
| `--threads`, `-t` | int | No | `None` (auto) | Worker threads; when unset, auto-capped by available memory on Linux (`/proc/meminfo`), else ~90% of CPUs |

> **Note:** `--version` defaults to the `BABEL_VERSION` literal in `cli.py` (a RENCI BABEL snapshot date stamp), and `--threads` unset lets the Rust build cap workers by available memory on Linux. See [Fullmap](fullmap.md) for the full default behavior.

### Example

```bash
tablassert gen-fullmap --output /data/fullmap/fullmap.redb
```

### Description

This command downloads BABEL class and synonym files from RENCI and builds a single embedded `fullmap.redb` file (via an in-memory parallel build) used for entity resolution during `build-kg`. See [Fullmap](fullmap.md) for the full data pipeline, output schema, and graph-config usage.

---

## Examples

### Check Version

```bash
tablassert --version
```

### Build Knowledge Graph

```bash
tablassert build-kg my-graph.yaml
```

### Validate Configuration

```bash
tablassert validate table-config.yaml
```

### Generate Fullmap Database

```bash
tablassert gen-fullmap
```

## Workflow

1. **Create table configuration** - Define data sources and transformations
2. **Create graph configuration** - Define output name, table configs, databases
3. **Validate config** - `tablassert validate table.yaml` (or `tablassert validate graph.yaml`)
4. **Build knowledge graph** - `tablassert build-kg graph.yaml`
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
