# Installation

Get a working `tablassert` install, then pick the `rt` / `qc` / `agent` extras that match how you will
use it (building graphs, auditing mappings, or running the autonomous agent).

## Prerequisites

- **Python 3.11 or higher**: Tablassert requires Python 3.11+ for compatibility with modern tooling
- **UV package manager**: Recommended for fast, reliable dependency management

### Installing UV

See the [official UV installation guide](https://github.com/astral-sh/uv) for your platform:

```bash
# Linux/macOS with curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# or with pip (any platform)
pip install uv
```

## Installation Methods

### Method 1: Development Installation with UV (Recommended for contributors)

Best for development, testing, and active work on Tablassert.

```bash
# Clone the repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install development dependencies and optional QC runtime
uv sync --group dev --extra qc

# Build the editable Rust extension into the uv environment
uv run maturin develop --manifest-path rust/Cargo.toml

# Verify the CLI
uv run tablassert --help
```

This creates a virtual environment in `.venv/`, installs the development dependencies, and builds the local PyO3 extension. The `tablassert` command is available through `uv run`. See [Development](development.md) and the repository `CONTRIBUTING.md` for the daily edit/check loop.

### Method 2: Install from PyPI

Recommended for most users. The base install builds knowledge graphs from CSV/TSV sources; QC and other
extras are opt-in.

```bash
# Option A: Install from PyPI with UV
uv tool install tablassert

# Option B: Install from PyPI with pip
pip install tablassert
```

#### Optional Extras

| Extra | Description | Includes |
|---|---|---|
| `rt` | Runtime-compatible Polars build | `polars[rtcompat]` |
| `qc` | QC runtime (exact → fuzzy → BioBERT audit) | `scikit-learn`, `sentence-transformers` (`torch` + `numpy` arrive transitively; `rapidfuzz` is a core dependency) |
| `agent` | Autonomous PMC → KG agent (`tablassert agent`) | `smolagents`, `dspy`, `litellm` |

```bash
# Install with runtime-compatible Polars
# (for CPUs without the required Polars instructions)
uv tool install "tablassert[rt]"

# pip equivalents
pip install "tablassert[rt]"

# Install the QC runtime
uv tool install "tablassert[qc]"
pip install "tablassert[qc]"

# Install the autonomous agent
uv tool install "tablassert[agent]"
pip install "tablassert[agent]"
```

Excel (`.xlsx`) input is read through Polars' `calamine` engine and additionally requires `python-calamine` (`pip install python-calamine`).

### Method 3: Install from GitHub main

Use this when you want the latest main-branch build.

```bash
# Install from main branch
uv tool install git+https://github.com/SkyeAv/Tablassert.git@main
```

### Method 4: Install from local source

For contributors testing local repository changes.

```bash
# Clone the repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install Tablassert CLI tool from local source
uv tool install .
```

## Verifying Installation

Confirm the CLI is on your path (use `uv run tablassert --help` for the in-repo dev environment):

```bash
tablassert --help
```

You should see the Tablassert CLI help message with available commands.

## Development Setup

For contributing to Tablassert, use the source install above, then run the local task runner:

```bash
make setup
make check
```

The underlying stable gates are `ruff check` / `ruff format --check`, `pyright`, `pytest`, and `cargo fmt --check` / `cargo test` / `cargo clippy --all-targets -- -D warnings` (see [Development](development.md)).

Install pre-commit hooks to run the fast lint/format checks automatically before commits (the full gates run in CI):

```bash
uv run pre-commit install
```

## Upgrading Development Installation

To upgrade to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies and rebuild the editable extension
uv sync --group dev --extra qc
uv run maturin develop --manifest-path rust/Cargo.toml
```

## Troubleshooting

### Python version

Tablassert requires Python 3.11+. On version errors, check `python --version` and pin a supported
release:

```bash
uv python install 3.11
uv python pin 3.11
```

### QC runtime

If `build-kg --qc` reports a missing QC runtime, install the `qc` extra (torch / sentence-transformers
BioBERT backend for the audit stage):

```bash
pip install "tablassert[qc]"
```
