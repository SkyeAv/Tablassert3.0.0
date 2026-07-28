# Contributing to Tablassert

Thank you for helping make Tablassert reliable. Reliability here includes the developer experience: a new contributor should be able to build, test, and understand the project without guesswork.

## Prerequisites

- Python 3.11 or newer.
- A stable Rust toolchain with `cargo`, `rustfmt`, and `clippy`.
- [uv](https://docs.astral.sh/uv/) for Python dependency and command management.
- GNU Make for the local task runner. `just` is useful in many projects, but this repo uses a minimal `Makefile` so the default loop works on this tree without extra tooling.

## One-time setup

```bash
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
uv sync --group dev --extra qc
uv run maturin develop --manifest-path rust/Cargo.toml
uv run tablassert --help
```

`uv sync --group dev --extra qc` installs the development tools plus the optional QC runtime. `maturin develop` builds the PyO3 extension from `rust/` and installs it into the uv-managed environment.

Shortcut:

```bash
make setup
```

## Daily edit → check loop

```bash
# after changing Rust code, or when the extension may be stale
make dev

# fastest focused checks
uv run pytest tests/test_lib.py::test_idxname_single_letter
cargo test --manifest-path rust/Cargo.toml fullmap::tests::clean_strips_matching_and_duplicate_quotes

# before commit
make check
```

Use `make build` when you need a release-mode extension for local performance checks. Normal development should use `make dev`; a release build can hide debug-only behavior and a later debug build replaces the editable extension in the same environment.

## Task runner

The `Makefile` is intentionally small and mirrors the underlying commands:

| Target | Runs |
|---|---|
| `make setup` | `uv sync --group dev --extra qc` and debug `maturin develop` |
| `make dev` | Debug editable extension build |
| `make build` | Release editable extension build |
| `make test` | Python tests with coverage |
| `make test-rust` | Rust tests |
| `make lint` | Ruff lint |
| `make fmt` | Ruff format and `cargo fmt` |
| `make fmt-check` | Ruff format check and `cargo fmt --check` |
| `make typecheck` | Pyright |
| `make check` | Lint, format check, typecheck, Python tests, Rust tests, and clippy with `-D warnings` |
| `make clean` | Local build/test/doc artifacts |

## Project layout

```text
src/tablassert/       Python package and CLI
rust/src/             PyO3 Rust extension exposed as tablassert.rs
tests/                Python tests and fixtures
docs/                 MkDocs site
mkdocs.yml            Documentation navigation and theme settings
pyproject.toml        Python metadata, dependencies, pytest/ruff settings
rust/Cargo.toml       Rust crate metadata and dependencies
.pre-commit-config.yaml  Local pre-commit quality hooks
```

The layout is deliberately flat. Prefer small, direct changes over new framework layers.

## Quality gates

Run the full local gate before opening a PR:

```bash
make check
```

The stable commands are:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
cargo fmt --check --manifest-path rust/Cargo.toml
cargo test --manifest-path rust/Cargo.toml
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
```

What the gates cover:

- **Ruff linting and formatting.** The current tree enforces core pycodestyle/pyflakes safety checks plus stale-suppression detection. The lint gate is being expanded to cover common bug patterns, simplifications, Python-version upgrades, pytest style, import order, and comprehensions. Treat `uv run ruff check .` and `uv run ruff format --check .` as the stable interface rather than relying on individual rule codes.
- **Pyright.** Type checking runs through `uv run pyright`; the project is tightening this as a strict-inference ratchet over time.
- **Python tests.** The suite is offline and currently runs 294 tests in roughly 30-40 seconds on this development tree, reporting coverage around 88%.
- **Rust tests.** `cargo test --manifest-path rust/Cargo.toml` currently runs 40 Rust unit tests for the extension.
- **Rust style and lints.** `cargo fmt --check` enforces formatting; clippy runs all targets with warnings denied.

## Pre-commit hooks

Install hooks after setup if you want the same checks to run automatically:

```bash
uv run pre-commit install
```

Configured hooks:

- `ruff`: fixes lint issues in `src/` and `tests/` when possible.
- `ruff-format`: formats Python files in `src/` and `tests/`.
- `pyright`: runs `uv run pyright` once per commit attempt.
- `pytest`: rebuilds the extension with `uv run maturin develop --manifest-path rust/Cargo.toml`, then runs `uv run pytest`.
- `cargo-fmt`: runs `cargo fmt --check --manifest-path rust/Cargo.toml`.
- `cargo-clippy`: runs `cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings`.
- `cargo-test`: runs `cargo test --manifest-path rust/Cargo.toml`.

## Running subsets

```bash
# one Python test
uv run pytest tests/test_lib.py::test_idxname_single_letter

# Python tests by keyword
uv run pytest -k "encoding"

# Rust-only tests
cargo test --manifest-path rust/Cargo.toml

# one Rust test by name
cargo test --manifest-path rust/Cargo.toml fullmap::tests::clean_strips_matching_and_duplicate_quotes
```

## Fullmap development notes

Fullmap builds are the heaviest local workflow. The build tunables are documented in [Fullmap: Build Tunables](docs/fullmap.md#build-tunables-environment); keep that page as the source of truth for `TABLASSERT_FULLMAP_SHARDS`, spill settings, producer counts, cache sizing, and related environment variables.

If a full BABEL build fails with `EMFILE`, `Too many open files`, or another NOFILE-limit error, raise the shell limit before rerunning:

```bash
ulimit -n 65535
```

If your OS hard limit is lower, raise the system/user NOFILE limit first, then open a new shell and rerun the build.

## Pull requests

- Use conventional commits (`docs:`, `fix:`, `feat:`, `test:`, `chore:`, etc.).
- Keep PRs focused on one concern.
- Include tests or explain why none are needed.
- Run `make check` and include the result in the PR description.
- PRs are squash-merged to `main`; write commits and PR titles so the squashed history stays clear.

## Reporting issues

Open bug reports and feature requests at [github.com/SkyeAv/Tablassert/issues](https://github.com/SkyeAv/Tablassert/issues). Include reproduction steps, the command you ran, and relevant environment details.

## License

By contributing to Tablassert, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
