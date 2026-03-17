# Tablassert

## Version 7.0.0

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
uv run tablassert-cli --help
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
uv run tablassert-cli build-knowledge-graph /path/to/graph-config.yaml
```

### Method 2: Install to Virtual Environment

For a more traditional Python development environment.

```bash
# Clone repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with UV
uv pip install -e .

# CLI is now available
tablassert-cli build-knowledge-graph /path/to/graph-config.yaml
```

### Method 3: Docker

Use the unified multi-arch image from GitHub Container Registry for containerized environments or when UV is not available.

```bash
# Multi-arch image (amd64 + arm64)
docker run --rm -v $(pwd):/workdir ghcr.io/skyeav/tablassert-cli:latest tablassert-cli build-knowledge-graph /path/to/config.yaml
```

The publish workflow (`workflow.yml`) also pushes a commit-pinned tag as `ghcr.io/skyeav/tablassert-cli:sha-<commit-sha>`.

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