# Installation

Tablassert uses Nix flakes for reproducible development environments. Below are all supported usage patterns.

## Prerequisites

- **Nix with flakes enabled** - [Install Nix](https://nixos.org/download.html)
- **Databases** (required at runtime):
  - `dbssert.duckdb` - Entity resolution database (DuckDB)
  - `PubMed.db` - PubMed metadata (SQLite)
  - `PMCSuppCaptions.db` - PMC figure captions (SQLite)

## Method 1: Development Shell (Recommended)

Best for exploring Tablassert or active development.

```bash
# Clone repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Enter development shell
nix develop -L .

# CLI is now available
tablassert-cli --help
```

The development shell provides:
- `tablassert-cli` command
- `mkdocs` for documentation
- All Python dependencies
- Chromium binary (auto-configured)

## Method 2: Direct Run from Flake

Run Tablassert without cloning or installing.

```bash
nix run github:SkyeAv/Tablassert#default -- /path/to/config.yaml
```

Useful for:
- One-off graph builds
- CI/CD pipelines
- Testing latest version

## Method 3: User Profile Installation

Install Tablassert persistently to your user environment.

```bash
# Install
nix profile install github:SkyeAv/Tablassert#default

# Use anywhere
tablassert-cli /path/to/config.yaml

# Upgrade
nix profile upgrade tablassert

# Remove
nix profile remove tablassert
```

## Method 4: Use as Overlay

Integrate Tablassert into your own Nix flake or NixOS configuration.

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    tablassert.url = "github:SkyeAv/Tablassert";
  };

  outputs = { self, nixpkgs, tablassert }: {
    # Add overlay to nixpkgs
    pkgs = import nixpkgs {
      system = "x86_64-linux";
      overlays = [ tablassert.overlays.default ];
    };

    # Now tablassert is available as pkgs.python313Packages.tablassert
    devShells.default = pkgs.mkShell {
      packages = [ pkgs.python313Packages.tablassert ];
    };
  };
}
```
## Method 5: Docker

Use prebuilt images from GitHub Container Registry when Nix is not available, on non-x86 systems, or in CI environments.

```bash
# x86_64 / amd64
docker run --rm -v $(pwd):/workdir ghcr.io/skyeav/tablassert-cli-amd64:latest tablassert-cli build-knowledge-graph /path/to/config.yaml
```

```bash
# aarch64 / arm64
docker run --rm -v $(pwd):/workdir ghcr.io/skyeav/tablassert-cli-arm64:latest tablassert-cli build-knowledge-graph /path/to/config.yaml
```

### Environment variables in the container

These are auto-configured in the image and do not need manual setup:

- `CHROMIUM_PATH`
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`

## Environment Variables

Tablassert requires these environment variables (automatically set by Nix wrapper):

- `CHROMIUM_PATH` - Path to Chromium browser for Playwright downloads
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` - Use system Chromium

**Note:** When using the Nix-provided package, these are configured automatically. Manual installation would require setting these.

## Python Requirements

If installing outside Nix (not recommended):

- Python 3.13+
- See `nix/overlay.nix` for complete dependency list

## Verifying Installation

```bash
# Check CLI is available
tablassert-cli --help

# Should output:
# Usage: tablassert-cli [OPTIONS] COMMAND [ARGS]...
#
# Tablassert Builds Knowledge Graphs From Declarative Configuration
#
# Commands:
#   build-knowledge-graph     Build knowledge graph from configuration
#   verify-table-configuration Verify table configuration syntax
#   --help                    Show this message and exit.
```

## Next Steps

- **[Tutorial](tutorial.md)** - Build your first knowledge graph
- **[Configuration](configuration/graph.md)** - Learn configuration syntax
