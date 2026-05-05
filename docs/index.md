# Tablassert

Tablassert is a highly performant declarative knowledge graph backend that extracts knowledge assertions from tabular data and exports NCATS Translator-compliant KGX (Knowledge Graph Exchange) NDJSON.

## What is Tablassert?

Tablassert transforms biomedical tabular data (Excel, CSV, TSV) into knowledge graphs through:

- **Declarative YAML configuration** - Define data transformations without code
- **Entity resolution** - Map text to biological entities (genes, diseases, chemicals) using comprehensive databases
- **Optional multi-stage quality control** - Exact matching, fuzzy matching, and BioBERT semantic validation
- **KGX compliance** - Outputs NCATS Translator-compatible NDJSON for node and edge files

## Key Features

- **Named Entity Recognition**: Case-dependent, provenance-rich NER with taxonomic filtering
- **Quality Control**: Three-stage validation (exact → fuzzy → BERT embeddings)
- **Biolink Compliance**: Uses Biolink categories and predicates throughout
- **Performance**: Lazy evaluation pipelines via Polars with DuckDB-accelerated entity resolution
- **Reproducible**: UV-based development environment with deterministic builds

## Quick Start

```bash
# Install from PyPI (UV) — minimal install
uv tool install tablassert
tablassert --help

# Install from PyPI (pip) — minimal install
pip install tablassert
tablassert --help

# Install runtime-compatible Polars build
# (for CPUs without the required Polars instructions)
uv tool install "tablassert[rt]"
# or
pip install "tablassert[rt]"

# Install CPU QC runtime
uv tool install "tablassert[qc]"
# or
pip install "tablassert[qc]"

# Install CUDA QC runtime
uv tool install "tablassert[qc-cuda]"
# or
pip install "tablassert[qc-cuda]"

# Or install latest from GitHub main
uv tool install git+https://github.com/SkyeAv/Tablassert.git@main
tablassert --help
```

QC is opt-in and requires either the `qc` or `qc-cuda` extra. CUDA support targets a single GPU on `device_id=0` and does not silently fall back to CPU if CUDA is unavailable.
See [Installation](installation.md) for details.

For development from source:

```bash
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
uv sync

# Run with your configuration
uv run tablassert build <config>
```

## Documentation Sections

- **[Installation](installation.md)** - Installation methods (PyPI, GitHub main, source development)
- **[CLI Reference](cli.md)** - Command-line interface usage
- **[Tutorial](tutorial.md)** - Step-by-step example with synthetic data
- **[Use Case Gallery](examples.md)** - Real-world configuration patterns for common data types
- **[Configuration](configuration/graph.md)** - Graph and table configuration reference
- **[API Reference](api/fullmap.md)** - Core functions documentation

## Authors

- **[Skye Lane Goetz](mailto:sgoetz@isbscience.org)** - Institute for Systems Biology, CalPoly SLO
- **[Gwênlyn Glusman](mailto:gglusman@isbscience.org)** - Institute for Systems Biology
- **Jared C. Roach** - Institute for Systems Biology

## License

See repository for license information.
