# AGENTS.md — Tablassert

## Fast Start

- Python package, not a monorepo. Main code lives in `src/tablassert/`; tests live in `tests/`.
- Install with `uv sync`. Three optional extras:
  - `--extra qc` / `--extra qc-cuda` — installs `onnxruntime` / `onnxruntime-gpu` for QC (strict runtime behavior below).
  - `--extra rt` — runtime-compatible Polars build for CPUs missing required SIMD instructions.
- CLI entrypoint is `tablassert.cli:APP`. Real user commands:
  - `uv run tablassert build <graph.yaml>` — 6 pipeline stages.
  - `uv run tablassert validate <table.yaml>` — 3 stages, syntax-only.

## Source of Truth

- When prose docs and code disagree, `src/tablassert/models.py` and `src/tablassert/cli.py` are authoritative. Docs drift: some pages claim shard files are `{0..11}.duckdb` or that only the `rt` extra exists; the code uses `SHARDS = 10` (so `0..9`) and exposes `qc`/`qc-cuda`/`rt` extras.

## Verify Changes

- Match the repo hooks before finishing: `uv run ruff check --fix .`, `uv run ruff format .`, `uv run pyright`, `uv run pytest`.
- After editing Rust code or Python wrappers around Rust, rebuild the extension before pytest: `uv run maturin develop --manifest-path rust/Cargo.toml`, then `uv run pytest`.
- Rust checks: `cargo fmt --check --manifest-path rust/Cargo.toml`, `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`, `cargo test --manifest-path rust/Cargo.toml`.
- Full hook run: `uv run pre-commit run --all-files` (ruff, ruff-format, pyright, pytest).
- Focused test runs:
  - Single test: `uv run pytest tests/test_lib.py::test_name`
  - By keyword: `uv run pytest -k "pattern"`
  - With print output: `uv run pytest -s tests/test_lib.py`
- Docs build: `uv run --group dev mkdocs build`.

## High-Value Structure

- `src/tablassert/cli.py` is the wiring layer: `build()` → `build_pipeline()`, `validate()` → `validate_pipeline()`.
- `src/tablassert/ingests.py` loads YAML and expands table configs into section dicts.
- `src/tablassert/lib.py` is the core pipeline:
  - `Tcode.collect()` builds the per-section operation list.
  - `compile_subgraph()` executes that list into parquet.
  - `compile_graph()` aggregates subgraph parquets into KGX NDJSON.
  - `resolve_many()` is the direct library API for batch entity resolution.
- Entity resolution uses DuckDB shard files at `<datassert>/data/{0..9}.duckdb`, opened read-only. `datassert` is a required `Path` field on the `Graph` model (not a fixed location); `src/tablassert/fullmap.py` hardcodes `SHARDS = 10`.

## Repo-Specific Gotchas

- Heavy dependencies are lazy-loaded per module with `TYPE_CHECKING` + `lazy_loader`. Follow the existing pattern instead of importing heavy packages eagerly. Lazy-loaded: polars, duckdb, xxhash, polars_hash, yaml, httpx, pyexcel, onnxruntime, sentence_transformers. The Rust extension ships prebuilt as `tablassert.tablassert_rs` (rebuild with `uv run maturin develop --manifest-path rust/Cargo.toml`).
- `tests/conftest.py` autouse-mocks `httpx.head`, so model-URL validation tests never hit the network unless a test opts in.
- Network-dependent tests are marked `@pytest.mark.network`; GPU QC tests are marked with both `network` and `gpu` in `tests/test_qc.py`.
- QC runtime selection is strict in `src/tablassert/qc.py`: if `onnxruntime-gpu` is installed but `CUDAExecutionProvider` is unavailable, the code raises (error 06) instead of falling back to CPU. Install `tablassert[qc]` for CPU-only.
- Downloader behavior in `src/tablassert/downloader.py` is two-path: `direct()` via `httpx` for known file URLs, `browser()` via Playwright fallback for browser-only sources. Keep payload-validation and cleanup tests intact when changing it.

## Conventions That Matter Here

- Start every module with `from __future__ import annotations`.
- Annotate every variable, including locals.
- Use `Optional[T]` / `Union[...]`, not `T | None`. Use `Path`, not `str`, for filesystem paths. Use `# pyright: ignore` to silence lazy-load false positives.
- No docstrings on functions. Use comment markers on the line above: `# ?` description, `# !` warning, `# *` pipeline stage, `# TODO:` todo.
- Pydantic models inherit `TablaBase` (`extra="forbid"`, `validate_assignment=True`). Required fields use `Field(...)`; optional use `Optional[T] = Field(None)`. Enums extend `str, Enum` and live in `enums.py`.
- Per-module logger: `from tablassert.log import cat; logger = cat("MODULE")` (e.g. `cat("FULLMAP")`). The CLI uses the root `from tablassert.log import logger`. Raise `RuntimeError` for failures; `logger.warning()` for non-fatal issues.

## Side Effects

- The package writes working artifacts to hidden dirs in the repo root, each defined in code: `.storassert/` (`utils.STORE`), `.logassert/` (`log.LOGASSERT`), `.cachassert/` (`models.CACHE`), `.onnxassert/` (`qc.MODEL`).
