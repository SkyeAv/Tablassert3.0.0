# Nix Packaging Map

## OVERVIEW

Nix packaging lives in this directory. The flake is Linux-only (`x86_64-linux`) and produces three main outputs: Python package overlay, dev shell, and Docker image.

## STRUCTURE

```
nix/
|- overlay.nix  # Python package builds: optimum-onnx + tablassert
|- shell.nix    # Dev shell tools
`- docker.nix   # Docker image output used by CI
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add/remove Python runtime deps | `nix/overlay.nix` | `propagatedBuildInputs` in `tablassert` |
| Change wrapped runtime env vars | `nix/overlay.nix` | `makeWrapperArgs` |
| Update dev shell tools | `nix/shell.nix` | `devShells.default.packages` |
| Update Docker output | `nix/docker.nix` | Consumed by workflow docker job |
| Change systems/inputs | `flake.nix` | flake-parts entry in repo root |

## PROJECT-SPECIFIC CONVENTIONS

- Use `format = "pyproject"` for Python builds in this repo.
- Keep `doCheck = false` unless a real automated test command is introduced.
- Keep wrapper/runtime claims aligned with code: only document env vars actually set in `makeWrapperArgs`.

## CURRENT RUNTIME WRAPPER FACTS

- `nix/overlay.nix` currently wraps `CHROMIUM_PATH`.
- `AWK_PATH`/`JQ_PATH` are referenced in architecture/docs, but are not currently wrapped in `makeWrapperArgs`.

## COMMANDS

```bash
# Build default package
nix build

# Open dev shell
nix develop -L .

# Build docker package output
nix build .#packages.x86_64-linux.docker -L
```

## ANTI-PATTERNS

- Do not document stale dependency lists by hand; prefer reading `propagatedBuildInputs` directly.
- Do not claim cross-platform support without adding systems in `flake.nix`.
- Do not describe env vars as wrapped when `makeWrapperArgs` does not set them.
