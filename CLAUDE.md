# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tablassert is a highly performant declarative knowledge graph backend that extracts knowledge assertions from tabular data and exports NCATS Translator-compliant KGX (Knowledge Graph Exchange) NDJSON. Version 6.1.0 is a complete Python rewrite designed for biomedical knowledge graph construction.

**Key Authors**: Skye Lane Goetz, Gwênlyn Glusman, Jared C. Roach
**License**: Apache License 2.0

## Development Environment

### Nix-Based Development

This project uses Nix flakes for reproducible development environments:

```bash
# Enter development shell
nix develop -L .

# Run the CLI
tablassert-cli --help

# Build the package
nix build

# Four Nix usage patterns:
# 1. Development shell (above)
# 2. Direct run: nix run github:SkyeAv/Tablassert#default -- -i config.yaml
# 3. User profile: nix profile install github:SkyeAv/Tablassert#default
# 4. Overlay: Integrate into own flake (see nix/overlay.nix)
```

### Documentation Preview

```bash
# Build documentation site
nix develop -L . -c mkdocs build

# Live preview with hot reload
nix develop -L . -c mkdocs serve
```

### Python Requirements

- Python 3.13+
- Build system: setuptools + wheel
- Entry point: `tablassert-cli` (maps to `tablassert.lib:CLI`)

## Core Architecture

### Processing Pipeline

Tablassert transforms tabular data into knowledge graphs through a declarative configuration pipeline:

1. **Configuration Ingestion** (`ingests.py`): YAML configurations define how to process tables
   - `from_yaml()`: Loads YAML configuration files
   - `to_sections()`: Converts configurations into processing sections using template merging via `fastmerge()`
   - **Template + Sections Pattern**: Configs can have a shared `template` with multiple `sections` for extracting different predicates from the same source
     - Dicts: Recursive merge, section overrides template
     - Lists: Concatenation
     - Scalars: Section replaces template

2. **Data Acquisition** (`downloader.py`): Downloads files using Playwright with Chromium
   - Handles Excel/CSV/TSV files from URLs
   - Caches downloaded files locally
   - Converts legacy XLS to XLSX format

3. **Transformation** (`lib.py`): Core data transformation using Polars LazyFrames with lazy-first hybrid execution
   - **"Tcode"**: A list of `(function, arguments)` tuples executed sequentially via `reduce()`
   - Text normalization levels: `zero()` (lowercase, strip), `one()` (remove non-word chars)
   - Column operations: value assignment, column copying, regex replacement, prefix/suffix
   - **Math transformations**: `copysign`, `pow` - use `"values"` token to reference column values
   - Row operations: slicing, picking, filtering via `reindex()`
     - Comparison operators: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
   - Special operations: exploding delimited values, filling nulls, significance testing
   - **Collection points**: Explicit eager evaluation at file I/O, custom Python functions (QC), database queries

4. **Entity Resolution** (`fullmap.py`): Maps text to biological entities (CURIEs)
   - `version4()`: Case-dependent, provenance-rich named entity recognition
   - Uses DuckDB queries against the `dbssert` database
   - Supports taxonomic filtering, category prioritization/avoidance
   - Outputs: CURIE ID, preferred name, biolink category, taxon, source, version

5. **Quality Control** (`qc.py`): Multi-stage validation of entity mappings
   - Stage 1: Exact string matching (fast path)
   - Stage 2: Fuzzy matching via RapidFuzz (medium confidence)
   - Stage 3: BioBERT embeddings + cosine similarity (semantic matching)
   - Uses disk caching (`diskcache`) for expensive operations

6. **Normalization** (`lib.py`): Converts wide format to KGX-compliant nodes and edges
   - `normalize()`: Extracts node columns (subject/object/qualifiers) into separate node tables
   - `publications()`: Creates publication nodes from provenance metadata
   - Adds MeSH annotations from PubMed database
   - Adds figure captions from PMC database

7. **Export** (`lib.py`): Aggregates subgraphs into NDJSON files
   - `compile_graph()`: Merges parquet subgraphs, deduplicates nodes
   - Uses AWK and JQ for final cleanup (removes nulls, deduplicates)
   - `label_edges()`: Assigns namespace UUIDs to edges
   - Output: `{name}_{version}.nodes.ndjson` and `{name}_{version}.edges.ndjson`

### Data Model (models.py)

All models inherit from `TablaBase` (Pydantic with strict validation):

- **Graph**: Top-level configuration specifying tables, databases, output name/version
- **Section**: Defines a single table transformation (source → statement → provenance)
- **Statement**: Knowledge assertion (subject-predicate-object with optional qualifiers)
- **NodeEncoding**: How to extract and resolve entities from columns
- **Encoding**: Generic column transformation (value/column method, regex, fill, explode)
- **Source**: Excel or Text file specification (URL, local path, row slicing, reindexing)
- **Provenance**: Publication metadata and contributor information
- **Status levels**: `"alpha"` (initial), `"beta"` (validation), `"primetime"` (production-ready)

### Key Enums (enums.py)

- **Categories**: 200+ Biolink entity types (Gene, Disease, ChemicalEntity, etc.)
- **Predicates**: 240+ Biolink relationship types (treats, affects, related_to, etc.)
- **Qualifiers**: Biolink qualifiers (anatomical_context, species_context, etc.)
- **Files**: TEXT (CSV/TSV) vs EXCEL
- **EncodingMethods**: VALUE (literal) vs COLUMN (reference to another column)

### Database Dependencies

- **dbssert**: DuckDB database for entity resolution (synonyms, CURIEs, sources, categories)
- **pubmed_db**: SQLite database with PubMed metadata (MeSH terms, authors, journals)
- **pmc_db**: SQLite database with PubMed Central figure captions

### Storage

- **storessert/**: Directory for intermediate parquet files (one per section)
- **cachessert/**: Disk cache for QC operations (~100MB LRU)
- Subgraph files named by MD5 hash of section configuration

## Common Commands

### Running Tablassert

```bash
# Process a knowledge graph configuration
tablassert-cli -i /path/to/config.yaml
```

### Development Tools

```bash
# Linting (flake8 is available in dev shell)
flake8 lib/

# Interactive data exploration (available in dev shell)
duckdb path/to/dbssert.db
datafusion-cli
```

## Architecture Patterns

### Functional Composition via Tcode

The core design pattern is "Tcode" - a list of transformation operations:

```python
tcode = [
    (from_url, (url, local_path)),
    (csv, (delimiter,)),
    (zero, (column_name,)),
    (version4, (column, dbssert_path, taxon, prioritize, avoid)),
    (to_store, (output_path,))
]
result = reduce(lambda acc, op: op[0](acc, *op[1]), tcode, None)
```

This enables declarative, composable transformations that can be cached and resumed.

### Multiprocessing

The main function uses `multiprocessing.Pool` to parallelize:
- YAML configuration loading
- Section extraction
- Subgraph compilation (each section processes independently)

### Pydantic Validation

All configuration is validated through Pydantic models with:
- `extra="forbid"`: Reject unknown fields
- `validate_assignment=True`: Validate on attribute changes
- `use_enum_values=True`: Convert enums to strings automatically

### Column Naming Conventions

- Excel columns: `column_1`, `column_2`, etc. (converted from A, B, C...)
- Original values: Prefixed with `"original "` (e.g., `"original subject"`)
- NLP processing: Suffixed with `" one"` for level-one text processing
- Node attributes: Suffixed with column role (`" name"`, `" category"`, `" taxon"`, etc.)

## External Dependencies

- **Polars**: LazyFrame operations with lazy-first hybrid execution model
- **DuckDB**: Entity resolution queries
- **Playwright + Chromium**: File downloads
- **BioBERT**: Semantic similarity for QC (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`)
- **RapidFuzz**: Fuzzy string matching
- **ONNX Runtime**: Optimized BERT inference
- **AWK + JQ**: Final JSON cleanup (set via env vars `AWK_PATH`, `JQ_PATH`)

## TODOs (from lib.py)

These are future improvements tracked in the codebase:

- Make MeSH nodes instead of just annotations
- Add more database access patterns
- Add more QC access patterns
- Change database architecture
- Add dbssert-cli as a micro repo
- Convert Perl download script to Python with zstd
- Add Loguru logging
- Add pytest tests
- Explore best model for QC (BioBERT evaluation)

## Important Notes

- **No tests**: Project currently has no test suite
- **Environment variables required**: `CHROMIUM_PATH`, `AWK_PATH`, `JQ_PATH` (set by Nix wrapper)
- **Disk cache**: QC operations are expensive; the disk cache is critical for performance
- **Parquet intermediate format**: Enables resumable processing and efficient parallel aggregation
- **UUID generation**: Uses namespace UUIDs (UUID v3) with domain-specific namespaces for deterministic IDs
