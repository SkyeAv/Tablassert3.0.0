# CLI Reference

Tablassert provides a single command-line interface for building knowledge graphs from declarative configurations.

## Command

```bash
tablassert-cli -i <graph-config.yaml>
```

## Options

### `-i, --ingest PATH` (required)

Path to the graph configuration file (YAML format).

**Example:**
```bash
tablassert-cli -i /path/to/MOKGV6.yaml
```

The graph configuration specifies:
- Output knowledge graph name and version
- Paths to table configurations
- Database locations (dbssert, pubmed_db, pmc_db)

See [Graph Configuration](configuration/graph.md) for details.

### `--help`

Display help message and exit.

```bash
tablassert-cli --help
```

## Output Files

Tablassert generates two NDJSON files:

- `{name}_{version}.nodes.ndjson` - Node file (entities)
- `{name}_{version}.edges.ndjson` - Edge file (relationships)

Where `{name}` and `{version}` come from the graph configuration.

**Example:**
```yaml
# graph-config.yaml
name: MULTIOMICS_KG
version: UNSTABLE
```

Produces:
- `MULTIOMICS_KG_UNSTABLE.nodes.ndjson`
- `MULTIOMICS_KG_UNSTABLE.edges.ndjson`

## Output Format

Files are KGX-compliant NDJSON (newline-delimited JSON):

**Nodes:**
```json
{"id": "HGNC:1234", "name": "GENE1", "category": ["biolink:Gene"], "taxon": "NCBITaxon:9606"}
{"id": "MONDO:0005148", "name": "diabetes mellitus", "category": ["biolink:Disease"]}
```

**Edges:**
```json
{"id": "uuid:...", "subject": "HGNC:1234", "predicate": "biolink:associated_with", "object": "MONDO:0005148"}
```

## Environment Variables

When using Nix, these are configured automatically:

- `CHROMIUM_PATH` - Chromium for file downloads
- `AWK_PATH` - GNU AWK for NDJSON processing
- `JQ_PATH` - JQ for JSON cleanup

## Examples

### Basic Usage

```bash
tablassert-cli -i my-graph.yaml
```

### With Absolute Path

```bash
tablassert-cli -i /home/user/configs/graph.yaml
```

### Using Nix Run (No Installation)

```bash
nix run github:SkyeAv/Tablassert#default -- -i ./graph.yaml
```

## Workflow

1. **Create graph configuration** - Define output name, table configs, databases
2. **Create table configurations** - Define data sources and transformations
3. **Run CLI** - `tablassert-cli -i graph.yaml`
4. **Process executes:**
   - Downloads files from URLs (if needed)
   - Applies transformations to each table
   - Resolves entities using dbssert
   - Validates mappings with QC pipeline
   - Aggregates subgraphs into NDJSON

## Next Steps

- **[Tutorial](tutorial.md)** - Complete example walkthrough
- **[Configuration Guide](configuration/graph.md)** - YAML configuration reference
