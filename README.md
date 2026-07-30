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

The base install builds knowledge graphs from CSV/TSV/Excel sources. Optional extras (`rt`, `qc`,
`agent`) add CPU-compatible Polars, the three-stage QC audit, and the autonomous agent — see the
[Installation guide](https://skyeav.github.io/Tablassert/installation/) for the full matrix. QC is opt-in
at build time (`build-kg --qc`); see the [CLI Reference](https://skyeav.github.io/Tablassert/cli/) for the
complete flag reference.

## Quick Demo

```python
from pathlib import Path
from tablassert.lib import resolve_many

results = resolve_many(col="gene", entities=["TP53", "BRCA1"], fullmap=Path("/path/to/fullmap"), taxon="9606")
# [{"original_gene": "TP53", "gene": "HGNC:11998", "gene_name": "TP53", ...}, ...]
```

Point `resolve_many()` at a fullmap database to resolve any iterable of entity strings to CURIEs — no
LazyFrame setup or NLP preprocessing required. See the
[Batch Resolution API](https://skyeav.github.io/Tablassert/api/lib/) for the full reference; for
YAML-configured pipeline builds use `tablassert build-kg config.yaml`.

## Key Features

Declarative YAML configs, built-in entity resolution, optional three-stage QC, and KGX-compliant NDJSON
output — with lazy Polars pipelines over an embedded redb resolution database. See the
[documentation](https://skyeav.github.io/Tablassert/) for the full feature overview and use-case gallery.

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

[Skye Lane Goetz](mailto:sgoetz@isbscience.org) — Institute for Systems Biology

[Gwênlyn Glusman](mailto:gglusman@isbscience.org) — Institute for Systems Biology

Jared C. Roach — Institute for Systems Biology
