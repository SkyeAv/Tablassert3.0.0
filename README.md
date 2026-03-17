# Tablassert

### By Skye Lane Goetz, Gwênlyn Glusman, and Jared C. Roach

Tablassert is a highly performant declarative knowledge graph backend designed to extract knowledge assertions from tabular data while exporting NCATS Translator-compliant Knowledge Graph Exchange (KGX) NDJSON.

## Documentation

**[Full Documentation](https://skyeav.github.io/Tablassert/)**

Complete guides covering installation, configuration, tutorials, and API reference.

## Quick Start

```bash
# Clone repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install with UV (requires Python 3.13+)
uv sync

# Run CLI
uv run tablassert --help
```

## Usage (With UV)

### Prerequisites

- Python 3.13 or higher
- UV package manager

### Method 1: Development Installation (Recommended)

Best for exploring Tablassert or active development.

```bash
# Clone and install dependencies
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
uv sync

# Run CLI through UV
uv run tablassert build-knowledge-graph /path/to/graph-config.yaml
```

### Method 2: Tool Installation with UV

For direct CLI usage without activating a virtual environment.

```bash
# Clone repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install CLI tool from local source
uv tool install .

# CLI is now available
tablassert build-knowledge-graph /path/to/graph-config.yaml
```

You can also install directly from GitHub without cloning:

```bash
# Install from main branch
uv tool install git+https://github.com/SkyeAv/Tablassert.git@main
```

## Key Features

- **Declarative Configuration:** YAML-based, no code required
- **Entity Resolution:** Maps text to biological entities (genes, diseases, chemicals)
- **Quality Control:** Three-stage validation (exact → fuzzy → BERT embeddings)
- **KGX Compliance:** NCATS Translator-compatible NDJSON output
- **Performance:** Parallel processing with disk caching

## Contributors

[Skye Lane Goetz](mailto:sgoetz@isbscience.org) - Institute for Systems Biology, CalPoly SLO

[Gwênlyn Glusman](mailto:gglusman@isbscience.org) - Institute for Systems Biology

Jared C. Roach - Institute for Systems Biology
