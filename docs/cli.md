# CLI Reference

Tablassert provides two commands.

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

## build

Build a knowledge graph from a YAML configuration file.

### Synopsis

```bash
tablassert build <graph_configuration_file>
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `graph_configuration_file` | Path | Yes | Knowledge Graph Configuration -- See Docs |

### Example

```bash
tablassert build /path/to/MOKGV6.yaml
```

### Description

This command runs the full extraction pipeline from a graph configuration file. It loads table configurations, downloads source files, applies transformations, resolves entities through datassert, validates mappings with the QC pipeline (exact → fuzzy → BERT), and compiles subgraphs into KGX-compliant NDJSON files.

The process executes in parallel stages with rich progress bars showing:
- Loading Tables
- Extracting Sections
- Building TCode
- Collecting Instructions
- Building Subgraphs
- Compiling Graph

Final output files are written to the current working directory as:
- `{name}_{version}.nodes.ndjson` - Node file (entities)
- `{name}_{version}.edges.ndjson` - Edge file (relationships)

Intermediate parquet artifacts are written to `.storassert/` during section processing.

See [Graph Configuration](configuration/graph.md) for details on the YAML schema.

---

## validate

Validate section syntax from a YAML configuration file.

### Synopsis

```bash
tablassert validate <table_configuration_file>
```

### Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `table_configuration_file` | Path | Yes | Table Configuration -- See Docs |

### Example

```bash
tablassert validate /path/to/table-config.yaml
```

### Description

This command validates a TC4 YAML configuration file without running the full extraction pipeline. It loads the file, extracts sections, and validates each section against the schema using Pydantic models. The command exits with a non-zero status if schema errors are detected, making it useful for CI/CD pipelines and pre-commit hooks.

Use this for:
- Quick syntax validation during development
- Pre-flight checks in CI/CD pipelines
- Verifying configuration changes before running expensive graph builds

See [Table Configuration](configuration/table.md) for details on the YAML schema.

---

## Examples

### Check Version

```bash
tablassert --version
```

### Build Knowledge Graph

```bash
tablassert build my-graph.yaml
```

### Validate Table Configuration

```bash
tablassert validate table-config.yaml
```

## Workflow

1. **Create table configuration** - Define data sources and transformations
2. **Create graph configuration** - Define output name, table configs, databases
3. **Validate table config** - `tablassert validate table.yaml`
4. **Build knowledge graph** - `tablassert build graph.yaml`
5. **Process executes:**
   - Downloads files from URLs (if needed)
   - Applies transformations to each table
   - Resolves entities using datassert
   - Validates mappings with QC pipeline
   - Aggregates subgraphs into NDJSON

## Next Steps

- **[Tutorial](tutorial.md)** - Complete example walkthrough
- **[Configuration Guide](configuration/graph.md)** - YAML configuration reference
