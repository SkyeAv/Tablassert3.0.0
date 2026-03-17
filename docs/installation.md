# Installation

This guide covers installing Tablassert on your system.

## Prerequisites

- **Python 3.13 or higher**: Tablassert requires Python 3.13+ for compatibility with modern tooling
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
uv run tablassert-cli --help
```

This creates a virtual environment in `.venv/` and installs all dependencies. The `tablassert-cli` command is available through `uv run`.

### Method 2: Install to Virtual Environment

For a traditional Python development environment without UV's managed virtual environment.

```bash
# Clone the repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Tablassert with UV
uv pip install -e .

# Tablassert CLI is now available
tablassert-cli --help
```

## Verifying Installation

After installation, verify that Tablassert is working correctly:

```bash
# If using UV
uv run tablassert-cli --help

# If installed to virtual environment (with venv activated)
tablassert-cli --help
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

## Upgrading

To upgrade to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
uv sync
```

## Troubleshooting

### Python Version Issues

Tablassert requires Python 3.13 or higher. If you encounter version errors:

```bash
# Check your Python version
python --version

# Use UV to manage Python versions
uv python install 3.13
uv python pin 3.13
```

### Dependency Installation Issues

If you encounter dependency installation issues, try:

```bash
# Clear UV cache and reinstall
uv cache clean
uv sync --reinstall
```
