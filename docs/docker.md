# Docker

Tablassert publishes a pre-built Docker image to GitHub Container Registry (ghcr.io) for containerized usage without a local Python installation.

## Image

The image is based on `python:3.14-slim` with the Tablassert CLI as the entrypoint. All dependencies are included in the base install.

```bash
docker pull ghcr.io/skyeav/tablassert:latest
```

Version-pinned tags match the git tag (e.g., `ghcr.io/skyeav/tablassert:v7.4.0`).

## Quick Start

```bash
# Show CLI help (default CMD)
docker run --rm ghcr.io/skyeav/tablassert:latest

# Check version
docker run --rm ghcr.io/skyeav/tablassert:latest --version
```

## Building a Knowledge Graph

The primary CLI command is `build`, which reads a graph configuration YAML file and produces KGX-compliant NDJSON output.

```bash
docker run --rm \
  -v /path/to/config:/data \
  -v /path/to/datassert:/datassert \
  ghcr.io/skyeav/tablassert:latest \
  build /data/graph-config.yaml
```

## Verifying Table Configuration

The `validate` command validates a table configuration YAML against the schema without running a full build.

```bash
docker run --rm \
  -v /path/to/config:/data \
  ghcr.io/skyeav/tablassert:latest \
  validate /data/table-config.yaml
```

## Included Capabilities

All dependencies ship in the base install, so the Docker image includes:

- **Quality control** — The QC pipeline in `src/tablassert/qc.py` runs a three-stage audit: exact match, then fuzzy matching via rapidfuzz (threshold >= 20), then BioBERT sentence embeddings with cosine similarity (threshold >= 0.2). The ONNX model is cached in `.onnxassert/` (line 26).
- **Web downloads** — `src/tablassert/downloader.py` uses Playwright to download remote files with retry logic.
- **Legacy Excel** — `modernize_xls()` in `src/tablassert/downloader.py` converts `.xls` files using pyexcel.

## Persistent Data Directories

Mount these volumes to persist data across container runs:

| Directory | Source | Purpose |
|---|---|---|
| `.storassert/` | `src/tablassert/utils.py:17` — `STORE` | Intermediate Parquet storage for compiled subgraphs |
| `.logassert/` | `src/tablassert/log.py` | Loguru log files with 100 MB rotation |
| `.onnxassert/` | `src/tablassert/qc.py:26` — `MODEL` | Cached ONNX/BioBERT model |

Example:

```bash
docker run --rm \
  -v ./config:/data \
  -v ./datassert:/datassert \
  -v ./.storassert:/app/.storassert \
  -v ./.logassert:/app/.logassert \
  -v ./.onnxassert:/app/.onnxassert \
  -w /app \
  ghcr.io/skyeav/tablassert:latest \
  build /data/graph-config.yaml
```

## Runtime Considerations

- **Datassert path** — The graph configuration YAML specifies the `datassert` path for the entity-resolution database. Ensure it is accessible inside the container.
- **Multiprocessing** — `src/tablassert/cli.py` uses `multiprocessing.Pool` for parallel table loading and section extraction.
- **DuckDB connections** — An `ExitStack` in `src/tablassert/cli.py` opens read-only connections to all 10 Datassert DuckDB shards concurrently.
- **Entity resolution** — The `fullmap` module (`src/tablassert/fullmap.py`) shards terms across 10 DuckDB shards (`SHARDS = 10`) using xxhash64.
- **Text normalization** — `src/tablassert/nlp.py` provides `level_one` (strip + lowercase) and `level_two` (regex-based cleanup).

## CI/CD Integration

Images are built by `.github/workflows/docker.yml`, which triggers on tag pushes (after autotag and PyPI publish complete). Tags match the repository version tag (e.g., `v7.2.2`).
