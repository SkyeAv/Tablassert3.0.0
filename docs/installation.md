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

This creates a virtual environment in `.venv/` and installs the base dependencies. The `tablassert` command is available through `uv run`.

### Method 2: Install from PyPI

Recommended for most users who just need the CLI.
Base install includes web and Excel support. QC runtime support is opt-in.

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
| `qc` | CPU QC runtime | `onnxruntime` |
| `qc-cuda` | CUDA QC runtime | `onnxruntime-gpu` |

```bash
# Install with runtime-compatible Polars
# (for CPUs without the required Polars instructions)
uv tool install "tablassert[rt]"

# pip equivalents
pip install "tablassert[rt]"

# Install CPU QC runtime
uv tool install "tablassert[qc]"
pip install "tablassert[qc]"

# Install CUDA QC runtime
uv tool install "tablassert[qc-cuda]"
pip install "tablassert[qc-cuda]"
```

The `qc` and `qc-cuda` extras are intended as separate install choices. `qc-cuda` targets a single NVIDIA GPU on `device_id=0` and hard-fails if `CUDAExecutionProvider` is unavailable at runtime.

Tablassert CLI is now available:

```bash
tablassert --help
```

### Method 3: Docker

Pre-built Docker images are available from GitHub Container Registry for containerized usage without a local Python installation.

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

# Add CPU QC runtime for QC tests
uv sync --dev --extra qc

# Or add CUDA QC runtime for GPU-backed QC tests
uv sync --dev --extra qc-cuda

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
builds, install Tablassert with the `rt` extra from `pyproject.toml`:

```bash
uv tool install "tablassert[rt]"
# or
pip install "tablassert[rt]"
```

### QC Runtime Issues

If you enable `qc: true` in a graph configuration without a QC runtime installed, install one of:

```bash
pip install "tablassert[qc]"
pip install "tablassert[qc-cuda]"
```

Use `qc-cuda` only on systems with a working NVIDIA CUDA/cuDNN environment. Tablassert will not silently fall back to CPU from the CUDA path.
