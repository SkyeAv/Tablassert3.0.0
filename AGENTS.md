# AGENTS.md — Tablassert

## Fast Start

- Python package, not a monorepo. Main code lives in `src/tablassert/`; tests live in `tests/`.
- Install with `uv sync`. QC is not available unless you install an extra: `uv sync --extra qc` or `uv sync --extra qc-cuda`.
- CLI entrypoint is `tablassert.cli:APP`. Real user commands are:
  - `uv run tablassert build <graph.yaml>`
  - `uv run tablassert validate <table.yaml>`

## Verify Changes

- Match the repo hooks before finishing: `uv run ruff check --fix .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`.
- Full hook run: `uv run pre-commit run --all-files`.
- Focused test runs:
  - Single test: `uv run pytest tests/test_lib.py::test_name`
  - By keyword: `uv run pytest -k "pattern"`
  - With print output: `uv run pytest -s tests/test_lib.py`
- Docs build: `uv run --group dev mkdocs build`

## High-Value Structure

- `src/tablassert/cli.py` is the wiring layer: `build()` calls `build_pipeline()`, `validate()` calls `validate_pipeline()`.
- `src/tablassert/ingests.py` loads YAML and expands table configs into section dicts.
- `src/tablassert/lib.py` is the core pipeline:
  - `Tcode.collect()` builds the per-section operation list.
  - `compile_subgraph()` executes that list into parquet.
  - `compile_graph()` aggregates subgraph parquets into KGX NDJSON.
  - `resolve_many()` is the direct library API for batch entity resolution.
- Entity resolution uses DuckDB shard files under `<datassert>/data/`. `src/tablassert/fullmap.py` hardcodes `SHARDS = 10`.

## Repo-Specific Gotchas

- Heavy dependencies are lazy-loaded per module with `TYPE_CHECKING` + `lazy_loader`. Follow the existing pattern instead of importing heavy packages eagerly.
- `tests/conftest.py` autouse-mocks `httpx.head`, so model URL validation tests do not hit the network unless a test is explicitly marked otherwise.
- Network-dependent tests are marked `@pytest.mark.network`; GPU QC tests are marked with both `network` and `gpu` in `tests/test_qc.py`.
- QC runtime selection is strict in `src/tablassert/qc.py`: if `onnxruntime-gpu` is installed but `CUDAExecutionProvider` is unavailable, the code raises instead of falling back to CPU.
- Downloader behavior in `src/tablassert/downloader.py` is two-path: direct `httpx` fetch for known file URLs, headless-browser fallback for browser-only sources. Keep tests around payload validation and cleanup intact when changing it.

## Conventions That Matter Here

- Start every module with `from __future__ import annotations`.
- Annotate locals, not just function signatures.
- Use `Optional[T]` / `Union[...]`, not `T | None`.
- Prefer `Path` over raw path strings.
- Function docs are usually `# ?` comments above the code, not docstrings.

## Side Effects

- The package writes working artifacts to hidden directories in the repo root: `.storassert/`, `.logassert/`, `.cachassert/`, and `.onnxassert/`.
