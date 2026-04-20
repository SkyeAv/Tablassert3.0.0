# Tablassert

[![PyPI](https://img.shields.io/pypi/v/tablassert.svg)](https://pypi.org/project/tablassert/)
[![Python](https://img.shields.io/pypi/pyversions/tablassert.svg)](https://pypi.org/project/tablassert/)
[![License](https://img.shields.io/pypi/l/tablassert.svg)](https://github.com/SkyeAv/Tablassert/blob/main/LICENSE)
[![Docs](https://img.shields.io/github/deployments/SkyeAv/Tablassert/github-pages?label=docs)](https://skyeav.github.io/Tablassert/)

Extract knowledge assertions from tabular data into NCATS Translator-compliant KGX NDJSON — declaratively, with entity resolution and quality control built in.

```bash
pip install tablassert
tablassert build-knowledge-graph config.yaml
```

**[Full Documentation](https://skyeav.github.io/Tablassert/)** — installation guides, tutorials, configuration reference, and API docs.

## Installation

```bash
pip install tablassert
```

All dependencies (ML, web, Excel support) are included in the base install. An optional extra is available for CPU compatibility:

```bash
pip install "tablassert[rtcompat]"  # Polars build for CPUs without required instructions
```

<details>
<summary><strong>Docker</strong></summary>

```bash
docker pull ghcr.io/skyeav/tablassert:latest

docker run --rm \
  -v /path/to/config:/data \
  -v /path/to/datassert:/datassert \
  ghcr.io/skyeav/tablassert:latest \
  build-knowledge-graph /data/graph-config.yaml
```

</details>

## Quick Demo

```bash
# Build a knowledge graph from a YAML configuration
$ tablassert build-knowledge-graph graph-config.yaml
⠋ Loading table configurations...
⠋ Resolving entities across 10 DuckDB shards...
⠋ Compiling subgraphs...
⠋ Deduplicating nodes and edges...
✓ Done — wrote nodes.ndjson and edges.ndjson to .storassert/
```

Define your entities and relationships in YAML, point tablassert at your data, and get NCATS Translator-compliant KGX NDJSON out the other side — no code required.

## Key Features

- **Declarative Configuration** — YAML-based, no code required
- **Entity Resolution** — Maps text to biological entities (genes, diseases, chemicals)
- **Quality Control** — Three-stage validation (exact → fuzzy → BERT embeddings)
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
