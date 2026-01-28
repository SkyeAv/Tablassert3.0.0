# MkDocs Documentation & GitHub Pages Deployment Design

**Date:** 2026-01-27
**Author:** Skye Lane Goetz with Claude Code
**Status:** Approved

## Overview

This design establishes comprehensive MkDocs documentation for Tablassert 6.0.0, covering installation, CLI usage, YAML configuration, and core API functions. It includes automated GitHub Pages deployment via GitHub Actions.

## Goals

1. Create essential documentation (8-10 pages) covering installation through advanced usage
2. Document all Nix usage patterns in README
3. Automate MkDocs deployment to GitHub Pages on main branch updates
4. Provide clear examples: simple tutorial + advanced real-world reference

## Documentation Structure

### Site Organization

```
docs/
├── index.md                    # Landing page with overview & quick start
├── installation.md             # Nix installation methods
├── cli.md                      # CLI usage and options
├── tutorial.md                 # Simple tutorial with synthetic example
├── configuration/
│   ├── graph.md               # Graph configuration reference
│   ├── table.md               # Table/Section configuration reference
│   └── advanced-example.md    # ALAM example as advanced reference
└── api/
    ├── fullmap.md             # Entity resolution (version4 function)
    ├── qc.md                  # Quality control utilities
    └── utils.md               # namespace_uuid function only
```

### MkDocs Configuration (mkdocs.yml)

```yaml
site_name: Tablassert
theme: readthedocs
nav:
  - Home: index.md
  - Installation: installation.md
  - CLI Reference: cli.md
  - Tutorial: tutorial.md
  - Configuration:
    - Graph Config: configuration/graph.md
    - Table Config: configuration/table.md
    - Advanced Example: configuration/advanced-example.md
  - API Reference:
    - Entity Resolution: api/fullmap.md
    - Quality Control: api/qc.md
    - Utilities: api/utils.md
```

## Content Specifications

### Landing Page (index.md)

**Purpose:** Project introduction and quick navigation

**Content:**
- Project overview: Declarative knowledge graph backend for biomedical data
- Key features: NCATS Translator KGX compliance, entity resolution, multi-stage QC
- Quick start: 3-4 commands to run with Nix
- Navigation links to major sections

**Length:** ~200-250 words

### Installation (installation.md)

**Purpose:** Complete guide to all Nix usage patterns

**Content:**
1. **Development Shell** (recommended for exploration)
   - `nix develop -L .`
   - Provides `tablassert-cli` command

2. **Direct Run from Flake**
   - `nix run github:SkyeAv/Tablassert#default -- -i config.yaml`
   - No installation needed

3. **Profile Installation**
   - `nix profile install github:SkyeAv/Tablassert#default`
   - Persists in user environment

4. **Overlay Usage**
   - How to add tablassert to your own flake
   - Code snippet showing overlay import

**Additional Info:**
- Environment variables (CHROMIUM_PATH, AWK_PATH, JQ_PATH) - auto-set by Nix wrapper
- Database requirements: dbssert, pubmed_db, pmc_db paths
- Python 3.13+ requirement (handled by Nix)

**Length:** ~300-400 words

### CLI Reference (cli.md)

**Purpose:** Document the single CLI command

**Content:**
- Command: `tablassert-cli -i <config.yaml>`
- Parameters:
  - `-i/--ingest`: Path to graph configuration (required)
- Output files: `{name}_{version}.nodes.ndjson` and `{name}_{version}.edges.ndjson`
- Example invocations with different configurations
- **Note:** Skip detailed error handling/exit codes (currently half-baked)

**Length:** ~200-250 words

### Tutorial (tutorial.md)

**Purpose:** End-to-end walkthrough with simple synthetic example

**Content:**
- Scenario: Gene-Disease associations from CSV
- Creates minimal graph config + table config
- Shows transformation pipeline: raw CSV → entity resolution → KGX output
- Step-by-step with command outputs
- Designed for 3-5 minute completion

**Example Structure:**
1. Create sample CSV file
2. Write table configuration
3. Write graph configuration
4. Run `tablassert-cli -i graph.yaml`
5. Inspect output NDJSON

**Length:** ~400-500 words with code blocks

### Graph Configuration (configuration/graph.md)

**Purpose:** Reference for top-level graph config

**Content:**
- Purpose: Orchestrates table processing into single knowledge graph
- Required fields with types:
  - `syntax`: "GC2"
  - `name`: String (output filename prefix)
  - `version`: String (output filename suffix)
  - `tables`: List of table config paths
  - `dbssert`: Path to DuckDB entity resolution database
  - `pubmed_db`: Path to SQLite PubMed metadata
  - `pmc_db`: Path to SQLite PMC figure captions
- Path resolution notes (absolute vs relative)
- Minimal example with 1 table
- Reference to MOKGV6.yaml structure

**Length:** ~250-300 words

### Table Configuration (configuration/table.md)

**Purpose:** Comprehensive reference for table/section configs

**Content:**

**Section 1: Template vs Sections Concept**
- Template: Shared/default configuration
- Sections: List of variations that inherit from template
- `fastmerge()` behavior explanation:
  - Dictionaries: Recursive merge, section overrides template
  - Lists: Concatenation (extends)
  - Scalars: Section value replaces template
- Use cases:
  - Template only: Single table, one output
  - Template + sections: Single table, multiple predicates/columns with shared provenance
- Example showing inheritance pattern

**Section 2: Configuration Schema**
Five main configuration sections:

1. **Template Metadata**
   - `syntax`: "TC3"
   - `status`: "alpha" | "beta" | "stable"

2. **Source**
   - Excel: `kind`, `local`, `url`, `row_slice`, `sheet`, `reindex`
   - Text: `kind`, `local`, `url`, `row_slice`, `delimiter`, `reindex`
   - Field reference table

3. **Statement** (Subject-Predicate-Object)
   - NodeEncoding fields: `method`, `encoding`, `taxon`, `prioritize`, `avoid`, `regex`, `remove`, `prefix`, `suffix`, `explode_by`, `transformations`
   - `predicate`: Biolink predicate enum
   - `qualifiers`: Optional list of Qualifier objects
   - Field reference table with types

4. **Provenance**
   - `repo`: "PMC" | "PUBMED" | "DOI"
   - `publication`: Identifier string
   - `contributors`: List with `kind`, `name`, `date`, `organizations`, `comment`

5. **Annotations**
   - Optional edge attributes
   - Same encoding options as Statement fields
   - Common examples: p-value, sample size, correction method

**Section 3: NodeEncoding Deep Dive**
- `method`: "value" (literal) vs "column" (reference to column name)
- `encoding`: The value or column reference
- `taxon`: NCBI Taxon ID for filtering
- `prioritize`: List of Biolink categories to prefer
- `avoid`: List of Biolink categories to exclude
- `regex`: List of pattern/replacement transformations
- `remove`: List of strings to filter out
- `explode_by`: Delimiter to split multi-value cells
- Examples for each option

**Length:** ~600-800 words total

### Advanced Example (configuration/advanced-example.md)

**Purpose:** Real-world configuration with annotations

**Content:**
- Full ALAM configuration (ALAMV6.yaml) with inline explanations
- Highlights:
  - Excel source with row slicing and sheet selection
  - Complex regex transformations for taxonomic names
  - Taxonomic filtering (prioritize OrganismTaxon, avoid Gene)
  - Multiple annotations (p-value, sample size, correction method, etc.)
  - Value vs column method examples
- Optional: Multi-section example demonstrating template inheritance

**Length:** ~400-500 words + annotated YAML

### API Reference: Entity Resolution (api/fullmap.md)

**Purpose:** Document the `version4()` entity resolution function

**Content:**
- Function signature with parameter types
- Parameters explained:
  - `p`: Path to parquet DataFrame
  - `col`: Column name to resolve
  - `dbssert`: Path to DuckDB database
  - `taxon`: Optional NCBI Taxon ID filter
  - `prioritize`: Optional list of preferred categories
  - `avoid`: Optional list of excluded categories
  - `tag`: NLP processing level suffix (default " one")
- Return type: `pl.DataFrame` with resolved entities
- Output columns: CURIE, name, category, taxon, source, version, NLP level, synonym
- DuckDB query explanation: Case-dependent NER with provenance
- Example usage

**Length:** ~250-300 words

### API Reference: Quality Control (api/qc.md)

**Purpose:** Document the `fullmap_audit()` QC function

**Content:**
- Function signature with parameter types
- Purpose: Multi-stage validation of entity mappings
- Three-stage process:
  1. **Exact Match**: String equality (fast path)
  2. **Fuzzy Match**: RapidFuzz ratio and partial token sort (medium confidence)
  3. **BERT Semantic**: BioBERT embeddings + cosine similarity (high confidence, expensive)
- Disk caching behavior via `diskcache` (~100MB LRU)
- Parameters:
  - `df`: DataFrame with entity mappings
  - `col`: Column name to audit
  - `out`: Output column name for pass/fail (default "passed")
- Returns: Filtered DataFrame with only validated mappings
- Performance notes

**Length:** ~250-300 words

### API Reference: Utilities (api/utils.md)

**Purpose:** Document `namespace_uuid()` function only

**Content:**
- Function signature: `namespace_uuid(domain: Any, *values: list[Any]) -> str`
- Purpose: Deterministic UUID generation for KGX compliance
- How it works:
  - Creates namespace UUID from domain string
  - Hashes tab-joined values within that namespace
  - Returns UUID v3 string
- Use case: Generates consistent IDs for edges across runs
- Example usage with different domains

**Length:** ~150-200 words

## README.md Updates

### Current Content
The README currently has minimal Nix usage:
```bash
nix develop -L .
tablassert-cli --help
```

### Proposed Updates

**Section: "Usage (With Nix)"**

Expand to cover all patterns:

```markdown
## Usage (With Nix)

### Development Shell (Recommended)
```bash
git clone https://github.com/SkyeAv/Tablassert.git
cd Tablassert
nix develop -L .
tablassert-cli --help
```

### Direct Run from Flake
```bash
nix run github:SkyeAv/Tablassert#default -- -i /path/to/config.yaml
```

### Install to User Profile
```bash
nix profile install github:SkyeAv/Tablassert#default
tablassert-cli -i /path/to/config.yaml
```

### Use as Overlay in Your Flake
```nix
{
  inputs.tablassert.url = "github:SkyeAv/Tablassert";

  outputs = { self, nixpkgs, tablassert }: {
    # Add overlay to your nixpkgs
    nixpkgsConfig = {
      overlays = [ tablassert.overlays.default ];
    };
  };
}
```
```

**New Section: "Documentation"**

Add after usage section:

```markdown
## Documentation

📚 **[Full Documentation](https://skyeav.github.io/Tablassert/)**

Covers installation, configuration, tutorials, and API reference.
```

## GitHub Actions Workflow

### File: `.github/workflows/docs.yml`

**Trigger:** Push to main branch (any changes)

**Strategy:** Use Nix development environment to build and deploy docs

**Steps:**
1. Checkout repository
2. Install Nix (using `cachix/install-nix-action@v27`)
3. Build docs: `nix develop -L . -c mkdocs build`
4. Deploy `site/` folder to `gh-pages` branch (using `peaceiris/actions-gh-pages@v3`)

**Rationale:**
- Leverages existing Nix setup (mkdocs already in devShell)
- No manual Python/pip configuration needed
- Consistent with local development environment
- Deploys built static site, not source files

### Workflow YAML Structure

```yaml
name: Deploy MkDocs to GitHub Pages

on:
  push:
    branches:
      - main

permissions:
  contents: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: cachix/install-nix-action@v27
        with:
          nix_path: nixpkgs=channel:nixos-unstable

      - name: Build documentation
        run: nix develop -L . -c mkdocs build

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./site
```

## Implementation Notes

### Content Style
- Follow previous documentation style from main branch (seen in graph_config.md, table_config.md)
- Technical but accessible
- Use tables for field references
- Include code examples for each major concept
- Keep explanations concise (YAGNI principle)

### Tutorial Example
Create synthetic example with:
- Simple CSV: 3-4 rows of gene-disease associations
- Minimal table config (no complex regex/filtering)
- Basic graph config (1 table, 3 database paths)
- Focus on understanding the pipeline, not edge cases

### Advanced Example
- Use ALAM config as-is with inline annotations
- Optionally create a second example showing template + sections pattern
- Reference column letters (A, B, C) as they appear in Excel

### API Documentation
- Extract function signatures directly from code
- Include type hints
- Focus on practical usage, not implementation details
- Link concepts back to configuration docs where relevant

## Success Criteria

1. ✅ MkDocs builds without errors
2. ✅ All navigation links work
3. ✅ Tutorial example runs successfully
4. ✅ GitHub Actions deploys to gh-pages on main push
5. ✅ README includes all four Nix usage patterns
6. ✅ Documentation site accessible at https://skyeav.github.io/Tablassert/

## Future Enhancements (Out of Scope)

- Search functionality (requires mkdocs-material theme)
- API documentation for all modules
- Video tutorials
- Interactive configuration builder
- Troubleshooting guide with common errors (after error handling is mature)
