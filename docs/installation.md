# Installation

This guide covers installing Tablassert on your system.

## Prerequisites

- **Python 3.11 or higher**: Tablassert requires Python 3.11+ for compatibility with modern tooling
- **UV package manager**: Recommended for fast, reliable dependency management

### Installing UV

See the [official UV installation guide](https://github.com/astral-sh/uv) for your platform:

```bash
# On Linux/macOS with curl
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Linux/macOS with pip
pip install uv

# On Windows with PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation Methods

### Method 1: Development Installation with UV (Recommended)

Best for development, testing, and active work on Tablassert.

```bash
# Clone the repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install dependencies with UV
uv sync

# Run Tablassert
uv run tablassert --help
```

This creates a virtual environment in `.venv/` and installs all dependencies. The `tablassert` command is available through `uv run`.

### Method 2: Install from PyPI

Recommended for most users who just need the CLI.
Tablassert uses optional dependency groups so you only install what you need.

```bash
# Option A: Install from PyPI with UV (minimal install)
uv tool install tablassert

# Option B: Install from PyPI with pip (minimal install)
pip install tablassert
```

#### Optional Extras

Tablassert defines the following optional dependency groups in `pyproject.toml`:

| Extra | Description | Includes |
|---|---|---|
| `ml` | Machine learning / quality control | `sentence-transformers`, `onnxruntime`, `optimum-onnx`, `scikit-learn` |
| `web` | Web-based file downloads | `playwright` |
| `pyexcel` | Legacy Excel format support | `pyexcel` |
| `rtcompat` | Runtime-compatible Polars build | `polars[rtcompat]` |
| `rt` | Alias for `rtcompat` | Same as `rtcompat` |
| `full` | All optional extras (no runtime compat) | `ml` + `web` + `pyexcel` |
| `full-rt` | All optional extras including runtime compat | `full` + `rtcompat` |

```bash
# Install with specific extras
uv tool install "tablassert[ml]"
uv tool install "tablassert[web]"
uv tool install "tablassert[ml,web]"

# Install everything (no runtime compat)
uv tool install "tablassert[full]"

# Install with runtime-compatible Polars
# (for CPUs without the required Polars instructions)
uv tool install "tablassert[rtcompat]"
# or use the shorter alias
uv tool install "tablassert[rt]"

# Install everything including runtime-compatible Polars
uv tool install "tablassert[full-rt]"

# pip equivalents
pip install "tablassert[full]"
pip install "tablassert[full-rt]"
pip install "tablassert[rtcompat]"
```

# Tablassert CLI is now available
tablassert --help
```

### Method 3: Docker

Pre-built Docker images are available from GitHub Container Registry for containerized usage without a local Python installation. The image includes all optional extras (`tablassert[full]`).

```bash
docker pull ghcr.io/skyeav/tablassert:latest

# Run CLI
docker run --rm ghcr.io/skyeav/tablassert:latest --help
```

See the [Docker documentation](docker.md) for full usage details including volume mounts and CI/CD integration.

### Method 4: Install from GitHub main

Use this when you want the latest main-branch build.

```bash
# Install from main branch
uv tool install git+https://github.com/SkyeAv/Tablassert.git@main

# Tablassert CLI is now available
tablassert --help
```

### Method 5: Install from local source

For contributors testing local repository changes.

```bash
# Clone the repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Install Tablassert CLI tool from local source
uv tool install .

# Tablassert CLI is now available
tablassert --help
```

## Verifying Installation

After installation, verify that Tablassert is working correctly:

```bash
# If using UV
uv run tablassert --help

# If installed as a UV tool
tablassert --help
```

You should see the Tablassert CLI help message with available commands.

## Development Setup

For contributing to Tablassert or running tests, follow these additional steps:

```bash
# Install development dependencies (includes pre-commit hooks)
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests
uv run pytest

# Run type checking
uv run pyright

# Run linting
uv run ruff check .
```

## Upgrading Development Installation

To upgrade to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
uv sync
```

## Troubleshooting

### Python Version Issues

Tablassert requires Python 3.11 or higher. If you encounter version errors:

```bash
# Check your Python version
python --version

# Use UV to manage Python versions
uv python install 3.11
uv python pin 3.11
```

### Dependency Installation Issues

If you encounter dependency installation issues, try:

```bash
# Clear UV cache and reinstall
uv cache clean
uv sync --reinstall
```

### Polars CPU Instruction Issues

If your machine does not support the CPU instructions required by default Polars
builds, install Tablassert with the runtime-compat extra from `pyproject.toml`:

```bash
uv tool install "tablassert[rtcompat]"
# or use the shorter alias
uv tool install "tablassert[rt]"
# or
pip install "tablassert[rtcompat]"
```
