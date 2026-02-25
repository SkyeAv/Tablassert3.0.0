# TABLASSERT KNOWLEDGE BASE

## OVERVIEW

Declarative biomedical knowledge graph backend. Extracts assertions from tabular data, exports KGX NDJSON. Python 3.13+, Nix-built, no tests.

## STRUCTURE

```
./
├── AGENTS.md                 # Root-level knowledge base (this file)
├── ARCHITECTURE.md           # Detailed architecture docs (canonical reference)
├── CHANGELOG.md              # Version history
├── README.md                 # Public-facing project overview
├── flake.nix                 # Flake-parts entry, imports nix/shell + nix/overlay
├── pyproject.toml            # setuptools+wheel, entry: tablassert-cli → tablassert.lib:CLI
├── mkdocs.yml                # Site nav and theme config
├── .flake8                   # Linter configuration (ignores E111, E302, etc.)
├── .github/workflows/
│   └── docs.yml              # Only CI: builds MkDocs, deploys GH Pages
├── lib/tablassert/            # ALL Python source (9 files, ~1400 LOC) — see lib/tablassert/AGENTS.md
│   ├── AGENTS.md              # Source code style guide and module catalog
│   ├── lib.py                 # Core pipeline, transforms, Tcode composition, CLI (454 LOC)
│   ├── enums.py               # All str+Enum types: Categories (~170), Predicates (~240) (506 LOC)
│   ├── fullmap.py             # Entity resolution via DuckDB (156 LOC)
│   ├── models.py              # Pydantic data models hierarchy (119 LOC)
│   ├── qc.py                  # Quality control with fuzzy/BERT matching (89 LOC)
│   ├── ingests.py             # YAML config loading and merging (41 LOC)
│   ├── downloader.py           # Playwright file downloads (41 LOC)
│   └── utils.py              # Utilities: paths, hashing, UUID (39 LOC)
├── nix/                      # Nix packaging — see nix/AGENTS.md
│   ├── overlay.nix             # Python package build (77 lines)
│   ├── shell.nix              # Dev shell definition (17 lines)
│   └── docker.nix             # Docker image build (22 lines)
├── docs/                     # MkDocs documentation — see docs/AGENTS.md
│   ├── index.md               # Homepage
│   ├── installation.md          # Nix usage patterns
│   ├── cli.md                 # CLI reference
│   ├── tutorial.md             # Step-by-step guide
│   ├── configuration/          # YAML config reference
│   │   ├── graph.md           # Graph-level config
│   │   ├── table.md           # Table-level config
│   │   └── advanced-example.md # Complex config example
│   ├── api/                   # Function documentation
│   │   ├── fullmap.md         # Entity resolution API
│   │   ├── qc.md              # Quality control API
│   │   └── utils.md          # Utility functions API
│   └── examples/              # Tutorial assets
│       ├── tutorial-data.csv
│       ├── tutorial-graph.yaml
│       └── tutorial-table.yaml
├── storessert/               # Intermediate parquet (gitignored)
├── cachessert/               # Disk cache ~100MB LRU (gitignored)
└── onnx/                     # BioBERT model cache (gitignored)
```

```
./
├── lib/tablassert/   # ALL Python source (9 files, ~1400 LOC) — see lib/tablassert/AGENTS.md
├── nix/              # overlay.nix (package+deps+env wrapping), shell.nix (devShell)
├── docs/             # MkDocs (readthedocs theme) — configuration, API ref, tutorial
├── .github/workflows/docs.yml  # Only CI: builds MkDocs, deploys GH Pages
├── CLAUDE.md         # Detailed architecture docs (canonical reference)
├── flake.nix         # Flake-parts entry, imports nix/shell + nix/overlay
├── pyproject.toml    # setuptools+wheel, entry: tablassert-cli → tablassert.lib:CLI
└── mkdocs.yml        # Site nav and theme config
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/modify transform ops | `lib/tablassert/lib.py` | Functions are LazyFrame→LazyFrame, composed via Tcode |
| Change data models | `lib/tablassert/models.py` | Pydantic strict, extra=forbid |
| Add entity types/predicates | `lib/tablassert/enums.py` | str+Enum, 507 lines |
| Entity resolution | `lib/tablassert/fullmap.py` | DuckDB queries against dbssert |
| QC / fuzzy / BERT matching | `lib/tablassert/qc.py` | BioBERT lazy-loaded, RapidFuzz, diskcache |
| YAML config loading | `lib/tablassert/ingests.py` | Template+sections merge via fastmerge |
| File downloads | `lib/tablassert/downloader.py` | Playwright+Chromium, XLS→XLSX conversion |
| Shared utils (hashing, UUID) | `lib/tablassert/utils.py` | STORE dir, DISKCACHE, namespace UUIDs |
| Nix packaging / env vars | `nix/overlay.nix` | Wraps CHROMIUM_PATH, AWK_PATH, JQ_PATH |
| Dev shell | `nix/shell.nix` | tablassert + mkdocs |
| Documentation content | `docs/` | MkDocs markdown pages |

## CODING STYLE — READ BEFORE WRITING ANY CODE

**See `lib/tablassert/AGENTS.md` for COMPLETE style guide. Violations are the #1 source of rework.**

Quick summary:
- **2-SPACE INDENT** (not 4) — `.flake8` ignores E111
- **`# ?` comments** replace docstrings — Title Case
- **No blank lines** between functions — E302 ignored
- **`from X import Y`** only, one per line
- **Type every variable** — `Union[X, Y]` not `X | Y`

**See `lib/tablassert/AGENTS.md` for the COMPLETE style guide. Violations are the #1 source of rework.**

Quick summary:
- **2-SPACE INDENT** (not 4) — `.flake8` ignores E111
- **`# ?` comments** replace docstrings — Title Case
- **No blank lines** between functions — E302 ignored
- **`from X import Y`** only, one per line
- **Type every variable** — `Union[X, Y]` not `X | Y`

## COMMANDS

```bash
# Enter dev shell
nix develop -L .

# Run CLI
tablassert-cli -i /path/to/config.yaml

# Build package
nix build

# Docs preview
nix develop -L . -c mkdocs serve

# Lint
flake8 lib/
```

## NIX

**See `nix/AGENTS.md` for detailed Nix packaging patterns.**

- `flake.nix` → flake-parts, imports `nix/shell.nix` + `nix/overlay.nix`
- `overlay.nix` → builds `optimum-onnx` (from GitHub) + `tablassert` (17 Python deps + 3 system: chromium, gawk, jq)
- `makeWrapper` sets env vars: `CHROMIUM_PATH`, `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD`, `AWK_PATH`, `JQ_PATH`
- `doCheck = false` — no test suite
- Four install patterns: dev shell, `nix run`, `nix profile install`, flake overlay

- `flake.nix` → flake-parts, imports `nix/shell.nix` + `nix/overlay.nix`
- `overlay.nix` → builds `optimum-onnx` (from GitHub) + `tablassert` (17 Python deps + 3 system: chromium, gawk, jq)
- `makeWrapper` sets env vars: `CHROMIUM_PATH`, `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD`, `AWK_PATH`, `JQ_PATH`
- `doCheck = false` — no test suite
- Four install patterns: dev shell, `nix run`, `nix profile install`, flake overlay

## NOTES

- **No tests** — no pytest, no test directory, no test CI
- **No linter beyond flake8** — no ruff, mypy, pyright config
- **ARCHITECTURE.md** is the canonical architecture doc — consult it for pipeline details, data model, external deps
- **Intermediate parquet** in `storessert/` — enables resumable processing
- **Disk cache** in `cachessert/` — ~100MB LRU for expensive QC operations
- **Environment variables required** at runtime: `CHROMIUM_PATH`, `AWK_PATH`, `JQ_PATH` (set by Nix wrapper)

- **No tests** — no pytest, no test directory, no test CI
- **No linter beyond flake8** — no ruff, mypy, pyright config
- **CLAUDE.md** is the canonical architecture doc — consult it for pipeline details, data model, external deps
- **Intermediate parquet** in `storessert/` — enables resumable processing
- **Disk cache** in `cachessert/` — ~100MB LRU for expensive QC operations
- **Environment variables required** at runtime: `CHROMIUM_PATH`, `AWK_PATH`, `JQ_PATH` (set by Nix wrapper)


## DATA FLOW

```
YAML Config (GC2/TC3)
    ↓
Pydantic Validation
    ↓
Tcode Generation (transform list)
    ↓
compile_subgraph() [reduce pattern]
    ↓
Parallel Table Processing
    ├─→ Download (Playwright)
    ├─→ Parse (CSV/Excel)
    ├─→ Transform (LazyFrame ops)
    ├─→ Entity Resolution (DuckDB)
    ├─→ QC (exact → fuzzy → BERT)
    └─→ Provenance (SQLite: MeSH, captions)
    ↓
storessert/{hash}.parquet (checkpoint)
    ↓
compile_graph() [aggregate all subgraphs]
    ↓
NDJSON Export (KGX format)
    ↓
Post-Processing (awk + jq)
```

## KEY ARCHITECTURAL PATTERNS

**Declarative Configuration**: YAML configs drive everything via Pydantic models
**Functional Composition**: Tcode pattern as [(function, args)] list composed via reduce
**Lazy Evaluation**: Polars LazyFrames enable efficient pipelining
**Resumable Processing**: Parquet checkpoints enable incremental execution
**Entity Resolution**: DuckDB with taxonomic filtering and priority CASE
**Quality Control**: Three-stage cascade (exact → fuzzy → BERT) with caching
**Strict Validation**: Pydantic extra='forbid' prevents config errors

## SEE ALSO

- `ARCHITECTURE.md` — Detailed pipeline architecture and design patterns
- `lib/tablassert/AGENTS.md` — Complete coding style guide and module catalog
- `nix/AGENTS.md` — Nix packaging and environment setup
- `docs/AGENTS.md` — MkDocs documentation structure and patterns