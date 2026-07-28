# Development

This page is the docs-site entry point for contributor setup. The canonical contributor guide is [`CONTRIBUTING.md`](https://github.com/SkyeAv/Tablassert/blob/main/CONTRIBUTING.md).

## Setup from source

```bash
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
uv sync --group dev --extra qc
uv run maturin develop --manifest-path rust/Cargo.toml
uv run tablassert --help
```

Or use the task runner:

```bash
make setup
```

## Daily loop

```bash
# Rebuild the editable Rust extension after Rust changes
make dev

# Run the full local gate before committing
make check
```

Useful focused checks:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/test_lib.py::test_idxname_single_letter
cargo test --manifest-path rust/Cargo.toml
```

## Fullmap builds

Fullmap build environment variables are documented in [Fullmap: Build Tunables](fullmap.md#build-tunables-environment). If a full build hits `EMFILE` / `Too many open files`, raise the NOFILE limit before rerunning:

```bash
ulimit -n 65535
```
