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
- Chromium, AWK, JQ binaries (auto-configured)

## Method 2: Direct Run from Flake

Run Tablassert without cloning or installing.

```bash
nix run github:SkyeAv/Tablassert#default -- -i /path/to/config.yaml
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
tablassert-cli -i /path/to/config.yaml

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


Use prebuilt Docker images from GitHub Container Registry. Useful for:

- Systems not running NixOS
- Non-x86_64 architectures (ARM64)
- CI/CD environments with Docker
- Isolated container environments

### x86_64 / amd64

```bash
# Pull and run latest image
docker run --rm -v $(pwd):/workdir -w /workdir \
  ghcr.io/SkyeAv/Tablassert:latest \
  tablassert-cli -i /path/to/config.yaml

# Pull specific commit version
docker run --rm -v $(pwd):/workdir -w /workdir \
  ghcr.io/SkyeAv/Tablassert:sha-<commit> \
  tablassert-cli -i /path/to/config.yaml
```

### aarch64 / arm64

```bash
# Pull and run latest image (available after Task 7 CI update)
docker run --rm -v $(pwd):/workdir -w /workdir \
  ghcr.io/SkyeAv/Tablassert:latest \
  tablassert-cli -i /path/to/config.yaml

# Pull specific commit version (available after Task 7 CI update)
docker run --rm -v $(pwd):/workdir -w /workdir \
  ghcr.io/SkyeAv/Tablassert:sha-<commit> \
  tablassert-cli -i /path/to/config.yaml
```

### Environment Variables

The Docker image includes the same environment variables as the Nix wrapper:

- `CHROMIUM_PATH` - Path to Chromium browser for Playwright downloads
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` - Use system Chromium

These are automatically configured in the container and do not require manual setup.



## Environment Variables

Tablassert requires these environment variables (automatically set by Nix wrapper):

- `CHROMIUM_PATH` - Path to Chromium browser for Playwright downloads
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` - Use system Chromium
- `AWK_PATH` - Path to GNU AWK for NDJSON processing
- `JQ_PATH` - Path to JQ for JSON cleanup

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
# Usage: tablassert-cli [OPTIONS]
#
# Tablassert Builds Knowledge Graphs From Declarative Configuration
#
# Options:
#   -i, --ingest PATH  Knowledge Graph Configuration -- See Docs
#                      [required]
#   --help             Show this message and exit.
```

## Next Steps

- **[Tutorial](tutorial.md)** - Build your first knowledge graph
- **[Configuration](configuration/graph.md)** - Learn configuration syntax
