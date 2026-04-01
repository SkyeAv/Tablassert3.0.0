# Tablassert

### By Skye Lane Goetz, Gwênlyn Glusman, and Jared C. Roach

Tablassert is a highly performant declarative knowledge graph backend for bioinformatics that extracts knowledge assertions from tabular data, performs entity resolution and data quality control, and exports NCATS Translator-compliant Knowledge Graph Exchange (KGX) NDJSON.

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

Or install the CLI directly from PyPI:

```bash
# Option A: UV tool install
uv tool install tablassert

# Option B: pip install
pip install tablassert

# Option C: runtime-compatible Polars build
# (for CPUs without the required Polars instructions)
uv tool install "tablassert[rtcompat]"
# or
pip install "tablassert[rtcompat]"

tablassert --help
```

## Usage (With UV)

### Prerequisites

- Python 3.13 or higher
- UV package manager
- [Datassert](https://skyeav.github.io/Tablassert/datassert/) — the entity-resolution database (`git clone https://github.com/SkyeAv/datassert`)

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

### Method 2: Install from PyPI

Recommended for most users.

```bash
# Option A: standard install (UV)
uv tool install tablassert

# Option B: standard install (pip)
pip install tablassert

# Option C: runtime-compatible Polars build
# (for CPUs without the required Polars instructions)
uv tool install "tablassert[rtcompat]"
# or
pip install "tablassert[rtcompat]"

tablassert build-knowledge-graph /path/to/graph-config.yaml
```

### Method 3: Install from GitHub main

Use this when you want the latest main-branch build before a tagged release.

```bash
uv tool install git+https://github.com/SkyeAv/Tablassert.git@main
tablassert build-knowledge-graph /path/to/graph-config.yaml
```

If your CPU does not support the instructions required by default Polars builds,
use **Method 2** with `tablassert[rtcompat]`.

### Method 4: Local source install

For contributors testing local changes.

```bash
# Clone repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install CLI tool from local source
uv tool install .

# CLI is now available
tablassert build-knowledge-graph /path/to/graph-config.yaml
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
