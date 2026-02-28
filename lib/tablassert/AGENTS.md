# lib/tablassert Source Guide

## OVERVIEW

Core Python runtime for Tablassert. This directory owns CLI execution, transform pipeline, entity mapping, QC, and schema validation.

## MODULE MAP

| File | Role |
|------|------|
| `lib.py` | CLI commands and end-to-end graph pipeline |
| `models.py` | Pydantic schemas (`extra='forbid'`) |
| `enums.py` | `str, Enum` vocabularies |
| `fullmap.py` | DuckDB-backed entity resolution |
| `qc.py` | exact -> fuzzy -> BERT audit cascade |
| `ingests.py` | YAML loading + section expansion |
| `downloader.py` | Playwright URL download handling |
| `utils.py` | store/cache paths, hashing, UUID helpers |

## STYLE (AUTHORITATIVE)

- 2-space indentation.
- Comment markers instead of docstrings (`# ?`, `# *`, `# !`, `# TODO:`).
- Prefer `from X import Y`; one import target per line.
- Type locals/variables explicitly; use `Union[...]` and `Optional[...]` forms.
- Keep transform helpers in LazyFrame -> LazyFrame style where applicable.
- Avoid bare `except:` blocks.

## IMPLEMENTATION PATTERNS

- Transform composition in `lib.py` uses Tcode instructions threaded with `reduce`.
- `build_knowledge_graph` is the main runtime command behind `tablassert-cli`.
- Parallel table/section stages use `multiprocessing.Pool` in CLI flow.
- Entity mapping (`fullmap.py`) writes temporary parquet then joins DuckDB matches.
- QC (`qc.py`) escalates exact match to fuzzy then semantic matching.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add transform operation | `lib.py` | Keep helper naming short (`lf`, `col`, `x`) |
| Update schema fields | `models.py` | Preserve strict model behavior |
| Add categories/predicates | `enums.py` | Keep enum members stable and explicit |
| Tune resolver behavior | `fullmap.py` | SQL query generation and join semantics |
| Adjust QC thresholds | `qc.py` | Match stage ordering and cache behavior |

## RUNTIME FACTS

- Downloader requires `CHROMIUM_PATH` at runtime.
- No automated test suite exists in this repository currently.

## ANTI-PATTERNS

- Do not relax model strictness by allowing unknown keys.
- Do not introduce style rules that conflict with this directory's established conventions.
- Do not duplicate architecture detail from `ARCHITECTURE.md`; link to it.
