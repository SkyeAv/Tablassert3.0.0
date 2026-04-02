# AGENTS.md — Tablassert

Guidance for AI coding agents working in this repository.

## Project Overview

Tablassert is a Python package (>=3.11) for tabular data assertion, normalization, and quality control. It builds declarative knowledge graphs from tabular data, exporting NCATS Translator-compliant KGX NDJSON. Uses **Polars** DataFrames, **DuckDB** for entity resolution, and **ONNX/BioBERT** for quality control. CLI built with **Typer**. Models built with **Pydantic v2**.

## Quick Reference

| Task | Command |
|---|---|
| Install | `uv sync` |
| Run CLI | `uv run tablassert` |
| Lint | `uv run ruff check .` |
| Lint (fix) | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Format check | `uv run ruff format --check .` |
| Type check | `uv run pyright` |
| All checks | `uv run pre-commit run --all-files` |
| Run all tests | `uv run pytest` |
| Run single test | `uv run pytest tests/test_foo.py::test_name` |
| Run by keyword | `uv run pytest -k "test_pattern"` |
| Run with print | `uv run pytest -s tests/test_foo.py` |
| Build | `uv build` |
| Build docs | `uv run --group dev mkdocs build` |
| Add dependency | `uv add <package>` |
| Add dev dependency | `uv add --group dev <package>` |

## Repository Structure

```
src/tablassert/
  cli.py          # Typer CLI (entry point: tablassert.cli:CLI)
  lib.py          # Core logic: encodings, data loading, Tcode(Section) class
  models.py       # Pydantic v2 models (TablaBase base class)
  enums.py        # str, Enum subclasses (Tokens, Repositories, Comparisons, etc.)
  fullmap.py      # NER / entity resolution (DuckDB, 16 shards)
  qc.py           # Quality control (ONNX/BioBERT, sentence_transformers)
  nlp.py          # Text normalization (level_one: strip+lowercase, level_two: regex)
  ingests.py      # YAML ingestion: from_yaml(), to_sections(), fastmerge()
  downloader.py   # Playwright-based file downloads with retries
  utils.py        # Hashing (xxhash), STORE path, namespace UUIDs
  log.py          # loguru logger → .logassert/logassert.log
  __init__.py     # Empty file (lazy loading is per-module, not here)
docs/             # MkDocs documentation source
mkdocs.yml        # MkDocs configuration
pyproject.toml    # Project config, dependencies, tool settings
tests/            # Test directory (at repo root)
```

- `conftest.py` provides a `fixtures_path` fixture returning `Path(__file__).parent / "fixtures"`.
- pytest configured via `pyproject.toml` `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.
- Test fixtures: `tests/fixtures/` contains YAML files for Section model tests.
- Test modules: `test_enums.py`, `test_fullmap.py`, `test_ingests.py`, `test_lib.py`, `test_models.py`, `test_nlp.py`, `test_utils.py`.

## Code Style

### Imports

- Every file starts with `from __future__ import annotations`
- Heavy dependencies are loaded **lazily per-module** using this pattern:
  ```python
  from typing import TYPE_CHECKING
  import lazy_loader as Lazy

  if TYPE_CHECKING:
      import polars as pl
  else:
      pl = Lazy.load("polars")
  ```
- Lazy-loaded deps: `polars`, `duckdb`, `orjson`, `typer`, `xxhash`, `polars_hash`, `yaml`
- Direct (non-lazy) heavy deps: `sqlite_utils`, `rapidfuzz`, `pydantic`, `loguru`, `yaml.CLoader`
- Previously-optional deps now in core: `sentence_transformers`, `onnxruntime`, `sklearn`, `playwright`, `pyexcel` — lazy-loaded when present
- Some modules mix direct and lazy imports for the same package (e.g., `ingests.py` does `from yaml import CLoader` directly, then lazy-loads `yaml` for `yaml.load()`)
- Import order: standard library → blank line → third-party → blank line → local
- Use `from __future__ import annotations` to enable deferred evaluation

### Type Annotations

- **Every variable** gets a type annotation, including locals: `col: str = "name"`, `df: pl.DataFrame = ...`
- Use `Optional[T]` and `Union[...]` (not `T | None` or `X | Y`)
- Use `Self` for class methods returning the class type
- Use `Path` (not `str`) for filesystem paths
- Use `# pyright: ignore` comments to suppress false positives from lazy-loaded modules

### Pydantic Models

- All models inherit from `TablaBase(BaseModel)` which sets:
  ```python
  model_config: ConfigDict = ConfigDict(  # pyright: ignore
      str_strip_whitespace=False,
      validate_assignment=True,
      use_enum_values=True,
      extra="forbid",
      populate_by_name=True,
  )
  ```
- Required fields: `Field(...)` (ellipsis sentinel)
- Optional fields: `Optional[T] = Field(None)`
- All enums are `str, Enum` subclasses (defined in `enums.py`)

### Enums

All enums live in `enums.py` and extend `str, Enum`. Key enums: `Tokens`, `Repositories`, `Contributions`, `Comparisons`, `Functions`, `Files`, `EncodingMethods`, `FillMethods`, `Syntaxes`, `Statuses`, `Categories`, `Predicates`, `Qualifiers`.

### Naming

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Module-level constants: `UPPER_CASE`

### Comments

- `# ?` — descriptions / clarifications
- `# !` — warnings / important notes
- `# *` — stage markers (pipeline steps)
- `# TODO:` — todos
- No docstrings on functions; use `# ?` comment on the line above instead

### Formatting (enforced by ruff)

- Line length: **120**
- Quote style: **double quotes**
- Indent: **4 spaces**
- `skip-magic-trailing-comma = true`
- Target: Python >=3.11

### Error Handling

- Use `RuntimeError` for exceptional cases (no custom exception classes currently)
- Use `logger.warning()` for non-fatal issues (e.g., empty subgraphs)
- Logger: `from tablassert.log import logger`

### Other Conventions

- `operator.add` for Polars string concatenation on columns (not `+` directly)
- CLI entry point: `tablassert.cli:CLI` (Typer app with `pretty_exceptions_show_locals=False`)
- Use `rich.progress` for progress tracking in CLI
- Data side-effects stored in hidden directories: `.logassert/`, `.storassert/`, `.onnxassert/`

## Tools

- **ruff** — linting (`ruff check`) and formatting (`ruff format`)
- **pyright** — type checking (no pyrightconfig.json; uses defaults)
- **pre-commit** — runs ruff fix, ruff-format, pyright, and pytest on all Python files
- **pytest** — testing (>=9.0.2)
- **uv** — package manager (use `uv run` for all commands, `uv add` for deps)
- **hatchling** — build backend

## Optional Dependency Groups

Defined in `pyproject.toml` `[project.optional-dependencies]`:
- `rtcompat` — `polars[rtcompat]` (runtime-compatible Polars build for CPUs without required instructions)
- `rt` — alias for `rtcompat`

All other dependencies (ML, web, Excel) are now in core `dependencies`.

Install with: `uv sync` or `pip install tablassert`

## CI Workflows

- **PyPI publish** (`.github/workflows/pipy.yml`): builds and publishes on push to `main`
- **MkDocs deploy** (`.github/workflows/docs.yml`): builds docs and deploys to GitHub Pages on push to `main`
- **Docker publish** (`.github/workflows/docker.yml`): builds and pushes image to GHCR on tag push (`v*`)
- **Autotag** (`.github/workflows/autotag.yml`): automatic version tagging

## Key Dependencies

polars, duckdb, orjson, pydantic, typer, xxhash, loguru, rapidfuzz, scikit-learn, sqlite-utils, pyyaml, lazy-loader, polars-hash, fastexcel, pyarrow, optimum-onnx
