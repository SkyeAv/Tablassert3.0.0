# Nix Packaging and Environment Setup

## OVERVIEW

Flake-based Nix packaging for the Tablassert declarative biomedical knowledge graph backend. Python 3.13+ environment with 17 PyPI dependencies and 3 system packages (chromium, gawk, jq).

## STRUCTURE

```
nix/
├── overlay.nix        # Python package build (optimum-onnx + tablassert)
├── shell.nix          # Dev shell definition (tablassert + mkdocs)
└── docker.nix         # Docker image build
```

## FLAKE ARCHITECTURE

### flake.nix

**Purpose**: Entry point using flake-parts for modular configuration

**Key Components**:
- **Systems**: `["x86_64-linux"]` — Linux-only
- **Inputs**: nixpkgs (nixos-unstable), flake-parts
- **Overlay**: `./nix/overlay.nix` provides tablassert package
- **Imports**: `./nix/docker.nix`, `./nix/shell.nix`
- **Per-system config**: Imports nixpkgs with project overlay

**Pattern**: Standard flake-parts setup with system-specific configuration

## OVERLAY PACKAGING

### overlay.nix

**Purpose**: Builds both optimum-onnx (external) and tablassert (internal) packages

**optimum-onnx Build**:
- Source: GitHub repository
- Format: `pyproject` build system
- No tests: `doCheck = false`
- Build isolation disabled: `format = "pyproject"`

**tablassert Build**:
- Source: `. ` (current directory)
- Format: `pyproject` build system
- Dependencies: 17 Python packages
  - Core: pyyaml, pydantic, polars
  - NLP: rapidfuzz-fuzzy, onnxruntime
  - Processing: duckdb, pyarrow
  - CLI: click, rich
  - Validation: pyright, packaging
  - Web: playwright, beautifulsoup4
  - Config: jsonschema
- System dependencies: chromium, gawk, jq
- Entrypoint: `tablassert-cli → tablassert.lib:CLI`
- Tests: `doCheck = false` (no test suite)

**Environment Variables** (via makeWrapper):
- `CHROMIUM_PATH` — Path to Chromium binary (for Playwright)
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD` — Skip browser auto-download
- `AWK_PATH` — Path to gawk (for NDJSON post-processing)
- `JQ_PATH` — Path to jq (for JSON transformation)

## DEV SHELL

### shell.nix

**Purpose**: Development shell with tablassert and mkdocs tools

**Includes**:
- `self.packages.${system}.default` (tablassert package)
- `pkgs.mkdocs` (documentation build)
- `pkgs.python3` (Python interpreter)

**Usage**:
```bash
nix develop -L .
```

## DOCKER IMAGE

### docker.nix

**Purpose**: Container image definition for reproducible deployments

**Components**:
- Base: NixOS container with tablassert package
- Environment: All required system packages (chromium, gawk, jq)
- Runtime: Full tablassert CLI with all Python dependencies

## INSTALLATION PATTERNS

### 1. Dev Shell (Recommended for development)
```bash
nix develop -L .
```

### 2. One-off Command Execution
```bash
nix run .# -- -i config.yaml
```

### 3. Profile Installation
```bash
nix profile install .
# Then run tablassert-cli from anywhere
```

### 4. Flake Overlay Integration
```nix
# In another flake.nix
inputs.tablassert.url = "path:/path/to/tablassert";

outputs = { self, nixpkgs, tablassert }: {
  overlays.default = final: prev: {
    inherit (tablassert.packages.${prev.system}) tablassert;
  };
};
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add Python dependency | `nix/overlay.nix` | Add to propagatedBuildInputs |
| Add system dependency | `nix/overlay.nix` | Add to nativeBuildInputs and buildInputs |
| Change Python version | `flake.nix` | nixpkgs.url input |
| Modify env vars | `nix/overlay.nix` | makeWrapper section |
| Add shell tool | `nix/shell.nix` | Add to buildInputs |
| Modify Docker base | `nix/docker.nix` | Change pkgs reference |

## NIX PATTERNS

### Dependency Management
- Python packages go in `propagatedBuildInputs` (runtime + build time)
- System packages go in both `nativeBuildInputs` (build) and `buildInputs` (runtime)

### Environment Variables
- Use `makeWrapper` to set runtime environment variables
- Pattern: `--set VAR_NAME value`

### Build System
- Always use `format = "pyproject"` for Python packages
- Disable isolation for local packages: `doCheck = false`
- External packages may need build isolation modifications

### Flake-parts
- Use `perSystem` callback for system-specific configuration
- Import overlay in `_module.args.pkgs`
- Standard pattern: `overlays = [self.overlays.default]`

## TROUBLESHOOTING

**Issue**: Chromedriver not found
- **Fix**: Ensure `CHROMIUM_PATH` is set in overlay.nix makeWrapper

**Issue**: mkdocs not available in dev shell
- **Fix**: Verify mkdocs is added to shell.nix buildInputs

**Issue**: Failing tests
- **Expected**: `doCheck = false` is intentional (no test suite)

**Issue**: Permission errors with storessert/ or cachessert/
- **Fix**: These are gitignored runtime directories, ensure they exist or are created on first run

## NOTES

- **No test suite** — `doCheck = false` is intentional
- **Python 3.13+** — nixos-unstable channel required
- **System packages** — chromium, gawk, jq are required at runtime
- **Environment variables** — MUST be set for tablassert to function (handled by makeWrapper)
- **Platform** — x86_64-linux only (no Darwin support in current config)

## SEE ALSO

- `../AGENTS.md` — Root-level project overview
- `../ARCHITECTURE.md` — Detailed architecture and data flow
- `../pyproject.toml` — Python package metadata and dependencies
- `../flake.nix` — Flake entry point
