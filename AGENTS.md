# TABLASSERT KNOWLEDGE BASE

## OVERVIEW

Declarative biomedical knowledge graph backend that extracts assertions from tabular data and exports KGX NDJSON. Python 3.13+, Nix-first workflow, minimal CI, no automated test suite.

## STRUCTURE

```
./
|- AGENTS.md                  # Root navigation and project-wide constraints
|- ARCHITECTURE.md            # Canonical pipeline architecture
|- README.md                  # Public quick start and install patterns
|- pyproject.toml             # Package metadata, CLI entrypoint
|- flake.nix                  # Flake entry, imports nix modules
|- mkdocs.yml                 # Docs nav and theme
|- .github/workflows/docs.yml # Docs + Docker publish pipeline
|- lib/tablassert/            # Python implementation (core runtime)
|- docs/                      # MkDocs content
|- nix/                       # Nix packaging and dev shell
`- storessert/, cachessert/   # Runtime artifacts (gitignored)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Core transforms and graph compile | `lib/tablassert/lib.py` | CLI commands + LazyFrame pipeline |
| Data models and schema | `lib/tablassert/models.py` | Pydantic models with strict validation |
| Entity and predicate enums | `lib/tablassert/enums.py` | Large `str, Enum` catalogs |
| Entity resolution | `lib/tablassert/fullmap.py` | DuckDB term matching pipeline |
| Quality control cascade | `lib/tablassert/qc.py` | exact -> fuzzy -> BERT |
| Config ingestion | `lib/tablassert/ingests.py` | YAML load + section merge |
| Downloader behavior | `lib/tablassert/downloader.py` | Playwright + `CHROMIUM_PATH` |
| Packaging and runtime env | `nix/overlay.nix` | Python app build + wrapper |
| Dev shell tooling | `nix/shell.nix` | `nix develop` toolchain |
| Documentation updates | `docs/` + `mkdocs.yml` | Content + nav wiring |

## CONVENTIONS

- Source-of-truth coding style is `lib/tablassert/AGENTS.md`.
- Canonical architecture reference is `ARCHITECTURE.md`.
- Keep AGENTS hierarchy non-redundant: parent for global rules, child for local deltas.

## ANTI-PATTERNS

- Do not add generic guidance that applies to every repository.
- Do not duplicate parent AGENTS content in child AGENTS files.
- Do not claim runtime env vars that are not actually wrapped in `nix/overlay.nix`.

## COMMANDS

```bash
# Enter development shell
nix develop -L .

# Run CLI
tablassert-cli -i /path/to/config.yaml

# Build package
nix build

# Lint (project standard)
flake8 lib/

# Serve docs locally
nix develop -L . -c mkdocs serve
```

## NOTES
| `tablassert-cli build-knowledge-graph <config>` | New 6.2.0 syntax |
- `pyproject.toml` defines `tablassert-cli = tablassert.lib:CLI`.
- No automated tests are configured in this repository today.

## SEE ALSO

- `lib/tablassert/AGENTS.md` - Source-level coding rules and module map
- `nix/AGENTS.md` - Packaging details and Nix-specific conventions
- `docs/AGENTS.md` - Documentation authoring map
- `ARCHITECTURE.md` - Detailed data-flow and system internals
