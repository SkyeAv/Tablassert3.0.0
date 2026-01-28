# Tablassert

## Version 6.0.0 (Beta)

### By Skye Lane Goetz, Gwênlyn Glusman, and Jared C. Roach

Tablassert is a highly performant declarative knowledge graph backend designed to extract knowledge assertions from tabular data while exporting NCATS Translator-compliant Knowledge Graph Exchange (KGX) NDJSON.

## Documentation

📚 **[Full Documentation](https://skyeav.github.io/Tablassert/)**

Complete guides covering installation, configuration, tutorials, and API reference.

## Quick Start

```bash
# Clone repository
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert

# Enter development shell
nix develop -L .

# Run CLI
tablassert-cli --help
```

## Usage (With Nix)

### Method 1: Development Shell (Recommended)

Best for exploring Tablassert or active development.

```bash
# Clone and enter development shell
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
nix develop -L .

# CLI is now available
tablassert-cli -i /path/to/graph-config.yaml
```

### Method 2: Direct Run from Flake

Run without cloning or installing.

```bash
nix run github:SkyeAv/Tablassert#default -- -i /path/to/config.yaml
```

### Method 3: User Profile Installation

Install persistently to your user environment.

```bash
# Install
nix profile install github:SkyeAv/Tablassert#default

# Use anywhere
tablassert-cli -i /path/to/config.yaml
```

### Method 4: Use as Overlay

Integrate into your own Nix flake or NixOS configuration.

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

    # Tablassert available as pkgs.python313Packages.tablassert
    devShells.default = pkgs.mkShell {
      packages = [ pkgs.python313Packages.tablassert ];
    };
  };
}
```

## Key Features

- **Declarative Configuration:** YAML-based, no code required
- **Entity Resolution:** Maps text to biological entities (genes, diseases, chemicals)
- **Quality Control:** Three-stage validation (exact → fuzzy → BERT embeddings)
- **KGX Compliance:** NCATS Translator-compatible NDJSON output
- **Performance:** Parallel processing with disk caching

## Contributors

[Skye Lane Goetz](mailto:sgoetz@isbscience.org) - Institute for Systems Biology, CalPoly SLO

[Gwênlyn Glusman](mailto:gglusman@isbscience.org) - Institute for Systems Biology

Jared C. Roach - Institute for Systems Biology
