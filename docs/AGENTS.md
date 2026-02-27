# Documentation Guide

## OVERVIEW

MkDocs content for installation, CLI usage, configuration, tutorial flow, and API notes for Tablassert.

## STRUCTURE

```
docs/
|- index.md
|- installation.md
|- cli.md
|- tutorial.md
|- configuration/
|  |- graph.md
|  |- table.md
|  `- advanced-example.md
|- api/
|  |- fullmap.md
|  |- qc.md
|  `- utils.md
`- examples/
   |- tutorial-data.csv
   |- tutorial-graph.yaml
   `- tutorial-table.yaml
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/adjust nav items | `mkdocs.yml` | Keep labels aligned with file names |
| Update install instructions | `docs/installation.md` | Match current Nix workflow |
| Update CLI docs | `docs/cli.md` | Keep flags/examples aligned with `tablassert-cli --help` |
| Update config reference | `docs/configuration/*.md` | Keep examples consistent with `models.py` |
| Update API docs | `docs/api/*.md` | Reflect real behavior in `lib/tablassert/*.py` |
| Update tutorial assets | `docs/examples/*` | Keep tutorial steps executable |

## CONVENTIONS

- Prefer concise, task-first sections over long narrative text.
- Use fenced code blocks with language identifiers for commands/configs.
- When documenting runtime env vars, verify against `nix/overlay.nix` wrapper state.
- Keep docs scoped to this repository; avoid generic Python/MkDocs tutorials.

## COMMANDS

```bash
# Preview docs locally
nix develop -L . -c mkdocs serve

# Build docs
nix develop -L . -c mkdocs build
```

## ANTI-PATTERNS

- Do not document options that are not implemented in code.
- Do not duplicate full architecture content from `ARCHITECTURE.md`.
- Do not leave stale references after CLI/config changes.
