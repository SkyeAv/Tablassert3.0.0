# lib/tablassert/ — SOURCE CODE

9 Python files, ~1400 LOC. This is the ENTIRE codebase.

## CODING STYLE — MANDATORY, READ FIRST

**Style violations are the #1 source of rework. Follow EVERY rule below.**

### Indentation

- **2-SPACE INDENT.** Not 4. `.flake8` ignores E111/E114 for this reason.

```python
# CORRECT
def foo(lf: LazyFrame, col: str) -> LazyFrame:
  x: int = 1
  return lf.with_columns(pl.lit(x).alias(col))

# WRONG — 4 spaces
def foo(lf: LazyFrame, col: str) -> LazyFrame:
    x: int = 1
    return lf.with_columns(pl.lit(x).alias(col))
```

### Comments Replace Docstrings

**NO docstrings.** Use comment markers instead:

| Marker | Meaning | Case | Example |
|--------|---------|------|---------|
| `# ?` | Function/section description | Title Case | `# ? Clean And Normalize Input` |
| `# *` | Stage/emphasis marker | Title Case | `# * Stage 2: Entity Resolution` |
| `# !` | Collection/eager point or warning | Title Case | `# ! Collect: Write Parquet` |
| `# TODO:` | Future work | Title Case | `# TODO: Add Retry Logic` |

```python
# CORRECT
# ? Build Query For Entity Resolution
def query_builder(p: Path, prioritize: list[str]) -> str:
  # * Construct UNION
  ...
  # ! Collect: Execute Query
  ...

# WRONG — docstring
def query_builder(p: Path, prioritize: list[str]) -> str:
  """Build query for entity resolution."""
  ...
```

### Imports

- **`from X import Y` only.** Never `import X`.
- **One import per line.** Never `from X import Y, Z`.
- Order is loose (not strictly stdlib→3rd-party→local).

```python
# CORRECT
from pathlib import Path
from polars import LazyFrame
from polars import DataFrame
from operator import add

# WRONG — bare import
import polars as pl

# WRONG — multiple per line
from operator import add, eq, le
```

### Type Annotations

- **Type EVERY variable.** No untyped locals.
- **`Union[X, Y]`** — never `X | Y`
- **`Optional[X]`** — never `X | None`
- **`Self`** for method returns
- **`# pyright: ignore`** for untyped external libraries (sentence-transformers, duckdb, etc.)

```python
# CORRECT
x: int = 5
name: str = "foo"
result: Optional[Path] = None
items: Union[str, int] = value
conn.execute(sql).pl()  # pyright: ignore

# WRONG — missing type
x = 5

# WRONG — pipe union
result: str | None = None
```

### Functions

- **No blank lines between functions.** `.flake8` ignores E302/E305.
- **Short parameter names**: `lf` (LazyFrame), `df` (DataFrame), `col` (column), `p` (Path), `x` (value).
- **LazyFrame→LazyFrame** pattern for transforms.
- **`from operator import add`** (etc.) for Polars expressions — not `+`, `==`, `<=`.

```python
# CORRECT — no blank line between functions
def value(lf: LazyFrame, col: str, x: str) -> LazyFrame:
  return lf.with_columns(pl.lit(x).alias(col))
def column(lf: LazyFrame, col: str, x: str) -> LazyFrame:
  return lf.with_columns(pl.col(x).alias(col))

# WRONG — blank line between functions
def value(lf: LazyFrame, col: str, x: str) -> LazyFrame:
  return lf.with_columns(pl.lit(x).alias(col))

def column(lf: LazyFrame, col: str, x: str) -> LazyFrame:
  return lf.with_columns(pl.col(x).alias(col))
```

### Strings and Formatting

- **Double quotes** for string literals.
- **f-strings** for formatting.
- **Trailing commas** in lists/tuples.

### Classes

- **Pydantic models only** — no plain classes (except Tcode which extends Section).
- **`ConfigDict(extra='forbid')`** on all models.
- **str+Enum** pattern for enumerations.

### Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Functions | snake_case, short | `to_store`, `sig`, `idx` |
| Variables | snake_case, abbreviated | `lf`, `df`, `col`, `p`, `x` |
| Classes | CamelCase | `Tcode`, `NodeEncoding` |
| Enum members | SCREAMING_SNAKE | `RELATED_TO`, `PMC` |
| Enum values | lowercase or PascalCase | `"treats"`, `"Gene"` |
| Constants | SCREAMING_SNAKE | `STORE`, `DISKCACHE`, `BIOBERT` |

### Error Handling

- Error messages use numbered prefixes: `'01 | Download Failed After {retries} Attempts: {last}'`

### What NOT To Do

- No docstrings (use `# ?` comments)
- No 4-space indent (use 2-space)
- No `import X` (use `from X import Y`)
- No `X | Y` unions (use `Union[X, Y]`)
- No `X | None` (use `Optional[X]`)
- No blank lines between functions
- No untyped variables
- No bare `except:` blocks

## MODULE CATALOG

| File | LOC | Purpose |
|------|-----|---------|
| `lib.py` | 444 | Core pipeline: transforms, Tcode composition, CLI entry, graph compilation |
| `enums.py` | 507 | All str+Enum types: Categories (~170), Predicates (~240), Qualifiers (~37), small enums |
| `fullmap.py` | 157 | Entity resolution: DuckDB queries against dbssert, NER via distinct→query→join |
| `models.py` | 120 | Pydantic models: TablaBase→Source→Statement→Section→Graph hierarchy |
| `qc.py` | 117 | Quality control: fuzzy matching (RapidFuzz), BERT matching (BioBERT/ONNX), diskcache |
| `ingests.py` | 42 | YAML config: CLoader, fastmerge (recursive dict merge), template+sections pattern |
| `downloader.py` | 42 | Playwright+Chromium downloads, XLS→XLSX conversion, retry+backoff |
| `utils.py` | 40 | STORE/DISKCACHE paths, MD5 hashing, namespace UUID generation |
| `__init__.py` | 0 | Empty |

## KEY PATTERNS

### Tcode Pipeline (lib.py)

Functional composition via `reduce`. Each transform is `(callable, args)`:

```python
# compile_subgraph threads LazyFrame through transforms
reduce(lambda lf, step: step[0](lf, *step[1:]), tcode, initial_lf)
```

Tcode.collect() returns cached `Path` if subgraph already exists (resumable).

### Entity Resolution (fullmap.py)

`version4()`: distinct terms → temp parquet → DuckDB UNION query with priority CASE → join back → coalesce columns.

### QC Pipeline (qc.py)

`fullmap_audit()`: exact match → fuzzy (RapidFuzz ≥80) → BERT (cosine sim ≥0.80). BioBERT lazy-loaded on first call.

### Model Hierarchy (models.py)

`TablaBase` → `{Reindex, BaseSource→Excel/Text, Regex, Math, Encoding→NodeEncoding→Qualifier, Statement, Contributor, Provenance, Annotation, Section, Graph}`. All `extra='forbid'`.

### Multiprocessing (lib.py)

`Pool.map()` for parallel YAML loading and section extraction in CLI.

## EXTERNAL DEPENDENCIES (NON-OBVIOUS)

| Dep | Used In | Why |
|-----|---------|-----|
| Polars (LazyFrame) | lib.py, fullmap.py | All data transforms — NOT pandas |
| DuckDB | lib.py, fullmap.py | Entity resolution SQL against dbssert databases |
| Playwright+Chromium | downloader.py | Headless browser file downloads |
| BioBERT (ONNX) | qc.py | Semantic similarity for QC — lazy-loaded |
| RapidFuzz | qc.py | Fuzzy string matching |
| diskcache | utils.py, qc.py | ~100MB LRU cache for expensive QC ops |
| AWK + JQ | lib.py (subprocess) | Post-processing NDJSON output |
| `from operator import` | lib.py, fullmap.py | `add`, `eq`, `le` for Polars expressions |

## NOTES

- **No tests.** No pytest, no test directory, no test CI.
- **Intermediate parquet** saved to `storessert/` — enables resumable pipeline.
- **Disk cache** at `cachessert/` — persists across runs.
- **CLI entry**: `tablassert-cli` → `tablassert.lib:CLI` (Typer).
- **Environment variables**: `CHROMIUM_PATH`, `AWK_PATH`, `JQ_PATH` — set by Nix wrapper, required at runtime.


## SEE ALSO

- `../../AGENTS.md` — Root-level project overview and quick reference
- `../../ARCHITECTURE.md` — Detailed pipeline architecture and design patterns
- `../../nix/AGENTS.md` — Nix packaging and environment setup
- `../../docs/AGENTS.md` — MkDocs documentation structure