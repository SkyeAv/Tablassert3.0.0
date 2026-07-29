# Tablassert

[![PyPI](https://img.shields.io/pypi/v/tablassert.svg)](https://pypi.org/project/tablassert/)
[![Python](https://img.shields.io/pypi/pyversions/tablassert.svg)](https://pypi.org/project/tablassert/)
[![License](https://img.shields.io/pypi/l/tablassert.svg)](https://github.com/SkyeAv/Tablassert/blob/main/LICENSE)
[![Docs](https://img.shields.io/github/deployments/SkyeAv/Tablassert/github-pages?label=docs)](https://skyeav.github.io/Tablassert/)

Extract knowledge assertions from tabular data into NCATS Translator-compliant KGX NDJSON — declaratively, with entity resolution built in and optional quality control.

```bash
pip install tablassert
tablassert build-kg config.yaml
```

**[Full Documentation](https://skyeav.github.io/Tablassert/)** — installation guides, tutorials, configuration reference, and API docs.

## Installation

```bash
pip install tablassert
```

The base install includes everything needed to build knowledge graphs from CSV/TSV sources. Optional extras are available for CPU compatibility and quality control:

```bash
pip install "tablassert[rt]"  # Polars build for CPUs without the required instructions
pip install "tablassert[qc]"  # Enable QC (torch + sentence-transformers BioBERT, scikit-learn)
```

Excel (`.xlsx`) inputs are read through Polars' `calamine` engine and additionally require `python-calamine` (`pip install python-calamine`).

QC is opt-in: pass `--qc` to `build-kg` to run the three-stage audit (exact → fuzzy → BioBERT). See the [CLI Reference](https://skyeav.github.io/Tablassert/cli/) for the full flag reference.

## Quick Demo

```python
from pathlib import Path
from tablassert.lib import resolve_many

# Resolve gene names to CURIEs against a fullmap database
results = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1", "EGFR"],
    fullmap=Path("/path/to/fullmap"),
    taxon="9606",
)

for row in results:
    print(f"{row['original_gene']} → {row['gene']} ({row['gene_name']})")
# TP53 → HGNC:11998 (TP53)
# BRCA1 → HGNC:1100 (BRCA1)
# EGFR → HGNC:3236 (EGFR)
```

Point `resolve_many()` at a fullmap database and resolve any iterable of entity strings to CURIEs — no LazyFrame setup or NLP preprocessing required. For full pipeline builds with YAML configuration, use `tablassert build-kg config.yaml`.

## Key Features

- **Declarative Configuration** — YAML-based, no code required
- **Entity Resolution** — Maps text to biological entities (genes, diseases, chemicals)
- **Quality Control** — Optional three-stage validation (exact → fuzzy → BERT embeddings)
- **KGX Compliance** — NCATS Translator-compatible NDJSON output
- **Performance** — Lazy evaluation pipelines with Polars and an embedded redb-accelerated entity resolution database

## Developing

```bash
uv sync --group dev --extra qc
uv run maturin develop --manifest-path rust/Cargo.toml
make check
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full development loop, quality gates, and pull request guidelines.

## License

[Apache License 2.0](LICENSE)

## Contributors

[Skye Lane Goetz](mailto:sgoetz@isbscience.org) — Institute for Systems Biology, CalPoly SLO

[Gwênlyn Glusman](mailto:gglusman@isbscience.org) — Institute for Systems Biology

Jared C. Roach — Institute for Systems Biology
