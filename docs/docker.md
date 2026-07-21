# Docker

Tablassert publishes a pre-built Docker image to GitHub Container Registry (ghcr.io) for containerized usage without a local Python installation.

## Image

The image is based on `python:3.14-slim` with the Tablassert CLI as the entrypoint. All dependencies are included in the base install.

```bash
docker pull ghcr.io/skyeav/tablassert:latest
```

Version-pinned tags match the git tag (e.g., `ghcr.io/skyeav/tablassert:v7.5.0`).

## Quick Start

```bash
# Show CLI help (default CMD)
docker run --rm ghcr.io/skyeav/tablassert:latest

# Check version
docker run --rm ghcr.io/skyeav/tablassert:latest --version
```

## Building a Knowledge Graph

The primary CLI command is `build-graph`, which reads a graph configuration YAML file and produces KGX-compliant NDJSON output.

```bash
docker run --rm \
  -v /path/to/config:/data \
  -v /path/to/fullmap:/fullmap \
  ghcr.io/skyeav/tablassert:latest \
  build-graph /data/graph-config.yaml
```

## Verifying Table Configuration

The `validate-table` command validates a table configuration YAML against the schema without running a full build.

```bash
docker run --rm \
  -v /path/to/config:/data \
  ghcr.io/skyeav/tablassert:latest \
  validate-table /data/table-config.yaml
```

## Included Capabilities

All dependencies ship in the base install, so the Docker image includes:

- **Quality control** — The QC pipeline in `src/tablassert/qc.py` runs a three-stage audit: exact match, then fuzzy matching via rapidfuzz (`fuzz.ratio` >= 20 or `partial_token_sort_ratio` >= 30), then BioBERT sentence embeddings with cosine similarity (threshold >= 0.2). The ONNX model is cached in `.tablassert/onnx/` (line 25).

## Persistent Data Directories

Mount a single volume at `.tablassert/` to persist all working artifacts across container runs. The subdirectories (`store/`, `log/`, `onnx/`) are auto-created on first use.

| Subdirectory | Source | Purpose |
|---|---|---|
| `.tablassert/store/` | `src/tablassert/utils.py:14` — `STORE` | Intermediate Parquet storage for compiled subgraphs |
| `.tablassert/log/` | `src/tablassert/log.py:13` — `LOGASSERT` | Loguru log files (`tablassert.log`) with 100 MB rotation |
| `.tablassert/onnx/` | `src/tablassert/qc.py:25` — `MODEL` | Cached ONNX/BioBERT model |

All three paths are derived from `utils.BASE = Path("./.tablassert")`.

Example:

```bash
docker run --rm \
  -v ./config:/data \
  -v ./fullmap:/fullmap \
  -v ./.tablassert:/app/.tablassert \
  -w /app \
  ghcr.io/skyeav/tablassert:latest \
  build-graph /data/graph-config.yaml
```

## Runtime Considerations

- **Fullmap path** — The graph configuration YAML specifies the `fullmap` path (a redb file, or base directory containing one) for the entity-resolution database. Ensure it is accessible inside the container.
- **Multiprocessing** — `src/tablassert/cli.py` uses `multiprocessing.Pool` for parallel table loading and section extraction.
- **Entity resolution** — The `fullmap` module (`src/tablassert/fullmap.py`) queries a single embedded redb file built by `tablassert build-fullmap`; see [Fullmap](fullmap.md) for the schema and build pipeline.
- **Text normalization** — `src/tablassert/nlp.py` provides `level_one` (strip + lowercase) and `level_two` (regex-based cleanup).

## CI/CD Integration

Images are built by `.github/workflows/docker.yml`, which triggers when the `Auto Tag Versions` workflow completes on `main` (also runnable manually via `workflow_dispatch`). Tags match the repository version tag (e.g., `v7.5.0`).
