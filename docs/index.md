# Tablassert

**Version 6.2.0 (Beta)**

Tablassert is a highly performant declarative knowledge graph backend that extracts knowledge assertions from tabular data and exports NCATS Translator-compliant KGX (Knowledge Graph Exchange) NDJSON.

## What is Tablassert?

Tablassert transforms biomedical tabular data (Excel, CSV, TSV) into knowledge graphs through:

- **Declarative YAML configuration** - Define data transformations without code
- **Entity resolution** - Map text to biological entities (genes, diseases, chemicals) using comprehensive databases
- **Multi-stage quality control** - Exact matching, fuzzy matching, and BioBERT semantic validation
- **KGX compliance** - Outputs NCATS Translator-compatible NDJSON for node and edge files

## Key Features

- **Named Entity Recognition**: Case-dependent, provenance-rich NER with taxonomic filtering
- **Quality Control**: Three-stage validation (exact → fuzzy → BERT embeddings)
- **Biolink Compliance**: Uses Biolink categories and predicates throughout
- **Performance**: Parallel processing with disk caching for expensive operations
- **Reproducible**: Nix-based development environment with deterministic builds

## Quick Start

```bash
# Clone and enter development environment
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
nix develop -L .

# Run with your configuration
| `tablassert-cli build-knowledge-graph <config>` | New 6.2.0 syntax |
```

## Documentation Sections

- **[Installation](installation.md)** - All Nix usage patterns
- **[CLI Reference](cli.md)** - Command-line interface usage
- **[Tutorial](tutorial.md)** - Step-by-step example with synthetic data
- **[Configuration](configuration/graph.md)** - Graph and table configuration reference
- **[API Reference](api/fullmap.md)** - Core functions documentation

## Authors

- **[Skye Lane Goetz](mailto:sgoetz@isbscience.org)** - Institute for Systems Biology, CalPoly SLO
- **[Gwênlyn Glusman](mailto:gglusman@isbscience.org)** - Institute for Systems Biology
- **Jared C. Roach** - Institute for Systems Biology

## License

See repository for license information.
