# Tablassert

[![PyPI](https://img.shields.io/pypi/v/tablassert.svg)](https://pypi.org/project/tablassert/)
[![Python](https://img.shields.io/pypi/pyversions/tablassert.svg)](https://pypi.org/project/tablassert/)
[![License](https://img.shields.io/pypi/l/tablassert.svg)](https://github.com/SkyeAv/Tablassert/blob/main/LICENSE)
[![Docs](https://img.shields.io/github/deployments/SkyeAv/Tablassert/github-pages?label=docs)](https://skyeav.github.io/Tablassert/)

Extract knowledge assertions from tabular data into NCATS Translator-compliant KGX NDJSON — declaratively, with entity resolution built in and optional quality control.

```bash
pip install tablassert
tablassert build config.yaml
```

**[Full Documentation](https://skyeav.github.io/Tablassert/)** — installation guides, tutorials, configuration reference, and API docs.

## Installation

```bash
pip install tablassert
```

Base install includes web and Excel support. Optional extras are available for CPU compatibility and QC runtime selection:

```bash
pip install "tablassert[rt]"       # Polars build for CPUs without required instructions
pip install "tablassert[qc]"       # Enable QC with CPU ONNX Runtime
pip install "tablassert[qc-cuda]"  # Enable QC with CUDA ONNX Runtime on GPU 0
```

QC is disabled by default at the graph level. Set `qc: true` in a graph config to enable the audit stage.

<details>
<summary><strong>Docker</strong></summary>

```bash
docker pull ghcr.io/skyeav/tablassert:latest

docker run --rm \
  -v /path/to/config:/data \
  -v /path/to/datassert:/datassert \
  ghcr.io/skyeav/tablassert:latest \
  build /data/graph-config.yaml
```

</details>

## Quick Demo

```python
from pathlib import Path
from tablassert.lib import resolve_many

# Resolve gene names to CURIEs against a datassert database
results = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1", "EGFR"],
    datassert=Path("/path/to/datassert"),
    taxon="9606",
)

for row in results:
    print(f"{row['original gene']} → {row['gene']} ({row['gene name']})")
# TP53 → HGNC:11998 (TP53)
# BRCA1 → HGNC:1100 (BRCA1)
# EGFR → HGNC:3236 (EGFR)
```

Point `resolve_many()` at a datassert database and resolve any iterable of entity strings to CURIEs — no LazyFrame setup, NLP preprocessing, or DuckDB connection management required. For full pipeline builds with YAML configuration, use `tablassert build config.yaml`.

## Key Features

- **Declarative Configuration** — YAML-based, no code required
- **Entity Resolution** — Maps text to biological entities (genes, diseases, chemicals)
- **Quality Control** — Optional three-stage validation (exact → fuzzy → BERT embeddings)
- **KGX Compliance** — NCATS Translator-compatible NDJSON output
- **Performance** — Lazy evaluation pipelines with Polars and DuckDB-accelerated entity resolution

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for development setup, code style, and pull request guidelines.

## License

[Apache License 2.0](LICENSE)

## Contributors

[Skye Lane Goetz](mailto:sgoetz@isbscience.org) — Institute for Systems Biology, CalPoly SLO

[Gwênlyn Glusman](mailto:gglusman@isbscience.org) — Institute for Systems Biology

Jared C. Roach — Institute for Systems Biology
