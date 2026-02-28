# Tablassert Architecture

## OVERVIEW

Tablassert is a declarative biomedical knowledge graph (KG) extraction system that transforms tabular data into KGX NDJSON format through a declarative YAML-driven pipeline. The system emphasizes reproducibility, resumability, and quality control through multi-stage entity resolution and semantic matching.

## CORE PRINCIPLES

**Declarative Configuration**: All behavior is driven by YAML configs validated by Pydantic models
**Functional Composition**: Transform pipelines built via reduce() pattern on callable+args tuples
**Lazy Evaluation**: Polars LazyFrames enable efficient pipelining with deferred computation
**Resumable Processing**: Parquet checkpoints in storessert/ enable incremental execution
**Strict Validation**: Pydantic extra='forbid' ensures config correctness at load time
**Quality First**: Three-stage QC cascade (exact → fuzzy → BERT) with LRU caching

## DATA FLOW ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                     YAML Configuration                          │
│  ┌─────────────────┐         ┌─────────────────┐                │
│  │  Graph Config   │         │  Table Config   │                │
│  │  (GC2/TC3)      │         │  (Multiple)     │                │
│  └─────────────────┘         └─────────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pydantic Validation                          │
│  - Schema validation                                            │
│  - Type checking                                                │
│  - extra='forbid' enforcement                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Tcode Generation                              │
│  Transform functions compiled into callable list:               │
│  [(func1, args1), (func2, args2), ...]                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               compile_subgraph() [reduce pattern]               │
│  reduce(lambda acc, (f, a): f(acc, *a), tcode, lf)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Parallel Table Processing (Pool.map)               │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Download   │  │     Parse    │  │  Transform   │           │
│  │  (Playwright)│  │  (CSV/Excel) │  │ (LazyFrame)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │        Entity Resolution (DuckDB)            │               │
│  │  • Distinct term extraction                  │               │
│  │  • Taxonomic filtering                       │               │
│  │  • Priority CASE for disambiguation          │               │
│  │  • Union-based query optimization            │               │
│  └──────────────────────────────────────────────┘               │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │      Quality Control Cascade                 │               │
│  │  1. Exact string matching                    │               │
│  │  2. Fuzzy matching (RapidFuzz ≥80)           │               │
│  │  3. BERT embedding similarity (≥0.80)        │               │
│  │  • diskcache LRU ~100MB                      │               │
│  └──────────────────────────────────────────────┘               │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │       Provenance Tracking                    │               │
│  │  • SQLite: MeSH terms                        │               │
│  │  • SQLite: Figure captions                   │               │
│  │  • Contributor attribution                   │               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           storessert/{hash}.parquet (Checkpoint)                │
│  • Resumable processing state                                   │
│  • Enables partial re-runs                                      │
│  • Parquet format for efficiency                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              compile_graph() [Aggregation]                      │
│  • Combines all subgraphs                                       │
│  • Deduplicates nodes/edges                                     │
│  • Applies graph-level provenance                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  NDJSON Export (KGX Format)                     │
│  • node.ndjson, edge.ndjson                                     │
│  • KGX standard format                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
```

## CORE ABSTRACTIONS
```

## CORE ABSTRACTIONS

### Tcode Pattern

**Purpose**: Declarative transform pipeline specification

**Structure**: List of (callable, args) tuples
```python
tcode: list[tuple[Callable, tuple]] = [
  (value, ("col_name", "literal")),
  (zero, ("text_col",)),
  (regex, ("col", r"\s+", "")),
  # ...
]
```

**Execution**: Functional composition via reduce()
```python
result: LazyFrame = reduce(
  lambda acc, (f, a): f(acc, *a),
  tcode,
  initial_frame
)
```

**Benefits**:
- Declarative: Transform list is config, not imperative code
- Resumable: Cached at Path checkpoints
- Composable: Transforms are pure LazyFrame→LazyFrame functions
- Testable: Each transform is independently testable

### LazyFrame Operations

**Principle**: All data transforms are LazyFrame→LazyFrame functions

**Pattern**:
```python
def transform(lf: LazyFrame, col: str, x: Any) -> LazyFrame:
  return lf.with_columns(pl.col(col).operation(x).alias(col))
```

**Why LazyFrames**:
- Query optimization: Polars plans execution
- Memory efficiency: Streaming where possible
- Checkpointing: Easy to write/read parquet

**Eager Collection Points** (requires .collect()):
- DuckDB queries (SQL engine)
- map_elements with custom Python functions
- BioBERT embeddings (sentence-transformers)
- RapidFuzz fuzzy matching
- File I/O operations

### Entity Resolution Strategy

**Input**: LazyFrame with raw entity names (e.g., gene names, disease names)

**Process** (fullmap.py version4):
1. **Distinct Extraction** (distinct()):
   - Extract unique terms from two normalization levels (original + " one" suffix)
   - Mark terms with NLP level (0 = original, 1 = processed)

2. **Temp File** (to_temp()):
   - Write terms to temp parquet
   - Required for DuckDB read_parquet() optimization

3. **DuckDB Query** (query_distinct()):
   - UNION-based query for better index utilization
   - Joins with SYNONYMS, SOURCES, CURIES, CATEGORIES tables (dbssert)
   - Priority CASE: prioritized categories → avoid list → default priority 50
   - Taxonomic filtering: optional TAXON_ID constraint
   - Result: term → CURIE mapping with metadata

4. **Join Back** (version4):
   - Join original LazyFrame with level 0 matches on original column
   - Join with level 1 matches on " one" suffix column
   - Coalesce: level 0 > level 1 for CURIE, PREFERRED_NAME, CATEGORY_NAME, etc.
   - Add provenance: source name, version, taxon, NLP level

**Output**: LazyFrame with resolved entities and metadata columns

**Key Design Decisions**:
- **UNION over OR**: DuckDB UNION enables better index utilization than OR conditions
- **Two-level matching**: Original + processed (" one") terms for robustness
- **Priority-based disambiguation**: Categories can be prioritized/avoided
- **Provenance tracking**: Every resolution has source and NLP level metadata
- **Taxonomic filtering**: Enables species-specific entity resolution

### Quality Control Cascade

**Input**: LazyFrame with resolved entities (CURIEs)

**Process** (qc.py fullmap_audit):
1. **Stage 1: Exact Matching** (can stay lazy):
   - Filter pairs where original == preferred_name
   - Fast, no NLP needed

2. **Stage 2: Fuzzy Matching** (requires eager):
   - Use RapidFuzz on remaining pairs
   - Thresholds: ratio ≥80 or partial_token_sort_ratio ≥80
   - Cached via diskcache LRU

3. **Stage 3: BERT Embeddings** (requires eager):
   - Use BioBERT sentence embeddings on remaining pairs
   - Compute cosine similarity between original and preferred_name
   - Threshold: cosine_similarity ≥0.80
   - Cached via diskcache LRU

**Output**: Filtered LazyFrame with only high-confidence mappings

**Key Design Decisions**:
- **Three-stage cascade**: Fast → medium → slow, only proceeding when needed
- **LRU caching**: diskcache ~100MB for expensive fuzzy/BERT operations
- **Lazy loading**: BioBERT loaded once on first use, saved to onnx/
- **Deletable errors**: Suspected errors are removed, not flagged

## MODULE ARCHITECTURE

### lib.py (Core Pipeline, 528 LOC)

**Responsibilities**:
- Transform operation implementations (value, column, math_op, zero, one, prefix, suffix, regex, fill, explode, sig, idx, csv, excel, crop)
- Tcode composition via compile_subgraph()
- Parallel YAML loading (Pool.map)
- Subgraph compilation (compile_subgraph)
- Graph compilation (compile_graph)
- CLI entry point (typer.app)

**Key Functions**:
- `compile_subgraph()`: Applies Tcode to LazyFrame via reduce
- `compile_graph()`: Aggregates all subgraphs, deduplicates nodes/edges
- `csv()`, `excel()`: Read sources as LazyFrames
- `from_url()`: Download files via Playwright

**Design Pattern**:
- Transform functions are pure LazyFrame→LazyFrame
- Tcode is list of (func, args) tuples
- compile_subgraph uses reduce for composition

### enums.py (Enumerations, 506 LOC)

**Responsibilities**:
- Categories (~170): Biolink categories for entities
- Predicates (~240): Biolink predicates for edges
- Qualifiers (~37): Edge qualifiers
- Tokens: Special tokens for transform configs (e.g., Tokens.AUTO, Tokens.VALUES)
- Files: File type literals (Excel, Text)
- Comparisons: Comparison operators
- EncodingMethods: Encoding strategy choices

**Pattern**:
- All are `class Name(str, Enum)`
- Values are lowercase/PascalCase (consistent with KGX standard)
- Used as literal types in Pydantic models

### models.py (Pydantic Models, 120 LOC)

**Responsibilities**:
- TablaBase: Base class with ConfigDict (extra='forbid', strict validation)
- Reindex: Reindexing specification
- BaseSource → Excel, Text: Source file specifications
- Regex, Math: Transform operation specifications
- Encoding → NodeEncoding → Qualifier: Encoding specifications
- Statement: Knowledge graph statement (edge)
- Contributor, Provenance: Attribution and provenance
- Annotation: Annotation encoding

**Pattern**:
- Strict Pydantic validation (extra='forbid')
- Field() for required fields, Field(None) for optional
- Literal types for kind fields
- Union types for flexible values

### fullmap.py (Entity Resolution, 156 LOC)

**Responsibilities**:
- distinct(): Extract unique terms from two normalization levels
- to_temp(): Write LazyFrame to temp parquet
- query_builder(): Build UNION-based DuckDB query with taxonomic filtering
- query_distinct(): Execute query against dbssert database
- version4(): Main entity resolution function with provenance

**Pattern**:
- Input: LazyFrame with raw entity names
- Output: LazyFrame with resolved entities (CURIEs) and metadata
- Uses DuckDB for SQL-based entity resolution
- Tracks provenance (source, version, taxon, NLP level)

### qc.py (Quality Control, 89 LOC)

**Responsibilities**:
- get_biobert(): Lazy-load BioBERT model, cache globally
- fuzz_audit(): Fuzzy matching decision (cached)
- BERT_audit(): BERT embedding similarity decision (cached)
- fullmap_audit(): Three-stage QC cascade

**Pattern**:
- Three-stage cascade: exact → fuzzy → BERT
- diskcache LRU (~100MB) for expensive operations
- BioBERT lazy-loaded, saved to onnx/
- Removes suspected errors (filters out)

### ingests.py (YAML Config, 41 LOC)

**Responsibilities**:
- CLoader: YAML loader for configs
- to_yaml(): Save objects to YAML
- from_yaml(): Load and validate Pydantic models from YAML
- to_sections(): Extract sections from YAML with fastmerge

**Pattern**:
- Template-based YAML parsing
- fastmerge for merging sections
- Returns validated Pydantic models

### downloader.py (File Downloads, 41 LOC)

**Responsibilities**:
- from_url(): Download files via Playwright headless browser
- XLS→XLSX conversion for Excel files

**Pattern**:
- Playwright with Chromium for headless downloads
- Handles authentication and complex web scraping
- Converts legacy XLS to XLSX for Polars compatibility

### utils.py (Utilities, 39 LOC)

**Responsibilities**:
- STORE: Path to storessert/ (intermediate parquet)
- DISKCACHE: Path to cachessert/ (LRU cache)
- hash_md5(): MD5 hashing
- samphash(): Sample-based MD5 hashing
- mkhash(): Multiple key hash
- namespace_uuid(): Namespace UUID generation (v5)

**Pattern**:
- Centralized path management
- Hashing utilities for checkpointing
- UUID generation for entity namespaces

## DATA MODEL HIERARCHY

```
TablaBase (BaseModel, extra='forbid')
├── Reindex (reindex specification)
├── BaseSource (source file)
│   ├── Excel (Excel source)
│   └── Text (CSV/Text source)
├── Regex (regex transform)
├── Math (math operation)
├── Encoding (encoding strategy)
│   └── NodeEncoding (node-specific)
│       └── Qualifier (node qualifier)
├── Statement (knowledge graph edge)
├── Contributor (attribution)
├── Provenance (publication info)
├── Annotation (annotation encoding)
├── Section (YAML section)
└── Graph (top-level config)
    └── provenance: Provenance
    └── contributors: list[Contributor]
```

## EXTERNAL DEPENDENCIES

### Python Packages (17 total)

**Core Framework**:
- polars: LazyFrame data processing
- pydantic: Config validation and data models
- pyyaml: YAML parsing

**Entity Resolution**:
- duckdb: SQL queries for entity resolution
- pyarrow: Arrow format (DuckDB interop)

**NLP / QC**:
- rapidfuzz-fuzzy: Fuzzy string matching
- sentence-transformers: BioBERT embeddings
- onnxruntime: ONNX model inference

**CLI / UX**:
- typer: CLI framework
- rich: Terminal output formatting

**Validation / Build**:
- pyright: Type checking
- packaging: Version parsing

**Web / Downloads**:
- playwright: Headless browser automation
- beautifulsoup4: HTML parsing

**Config**:
- jsonschema: JSON schema validation

### System Packages (1)

- chromium: Playwright browser (CHROMIUM_PATH env var)
### External Data

- dbssert: DuckDB database with SYNONYMS, SOURCES, CURIES, CATEGORIES tables
- BioBERT model: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb (cached in onnx/)

## DESIGN PATTERNS

### 1. Functional Composition Pattern

**Pattern**: Transform pipelines as reduce over callable+args tuples

**Example**:
```python
tcode = [(zero, ("text",)), (regex, ("text", r"\s+", "")), (upper, ("text",))]
result = reduce(lambda acc, (f, a): f(acc, *a), tcode, lf)
```

**Benefits**:
- Declarative config
- Easy to reserialize/deserialize
- Testable in isolation
- Composable

### 2. Lazy Evaluation Pattern

**Pattern**: Stay lazy as long as possible, collect only when necessary

**Example**:
```python
# Can stay lazy
lf = lf.with_columns(pl.col("x").alias("y"))

# Must collect (DuckDB)
df = lf.collect()
matches = conn.execute(query).pl()

# Back to lazy
result = df.lazy()
```

**Benefits**:
- Query optimization by Polars
- Streaming where possible
- Memory efficiency
- Easy checkpointing (write_parquet)

### 3. Strict Validation Pattern

**Pattern**: Pydantic extra='forbid' prevents config errors

**Example**:
```python
class TablaBase(BaseModel):
  ConfigDict(extra='forbid', validate_assignment=True)
```

**Benefits**:
- Fail fast on config errors
- Catch typos before execution
- Enforce schema at load time

### 4. Three-Stage Cascade Pattern

**Pattern**: Fast → medium → slow, only proceeding when needed

**Example**:
```python
# Stage 1: Fast (can stay lazy)
exact = filter(eq(original, preferred))

# Stage 2: Medium (eager)
fuzzy = map_elements(fuzz_audit, exact_failures)

# Stage 3: Slow (eager)
bert = map_elements(BERT_audit, fuzzy_failures)
```

**Benefits**:
- Optimize for common case (exact match)
- Only pay cost when needed
- Caching reduces repeated work

### 5. Union-Based Query Pattern

**Pattern**: DuckDB UNION over OR for better index utilization

**Example**:
```sql
-- BAD: OR doesn't use index efficiently
SELECT * FROM synonyms WHERE synonym = 'term1' OR synonym = 'term2'

-- GOOD: UNION uses index twice
SELECT * FROM synonyms WHERE synonym = 'term1'
UNION
SELECT * FROM synonyms WHERE synonym = 'term2'
```

**Benefits**:
- Better index utilization
- Parallelizable execution
- More predictable performance

### 6. Resumable Processing Pattern

**Pattern**: Parquet checkpoints enable incremental execution

**Example**:
```python
checkpoint_path = STORE / mkhash(config_hash)
if checkpoint_path.exists():
  return pl.read_parquet(checkpoint_path)
result = process(...)
result.sink_parquet(checkpoint_path)
```

**Benefits**:
- Resume from failures
- Skip completed work
- Debug intermediate states

## DIRECTORY LAYOUT

```
storessert/      # Intermediate parquet (gitignored)
├── {hash}.parquet # Checkpoints for resumable processing

cachessert/      # Disk cache (gitignored)
└── *           # LRU cache ~100MB for QC operations

onnx/            # BioBERT model cache (gitignored)
└── *           # Saved BioBERT ONNX model

dbssert         # Entity resolution database (external)
└── *.duckdb   # Synonyms, sources, curies, categories
```

## KEY DESIGN DECISIONS

### Why LazyFrames?
- Query optimization: Polars plans execution
- Memory efficiency: Streaming where possible
- Checkpointing: Easy to write/read parquet
- Composition: Transform functions are pure

### Why DuckDB for Entity Resolution?
- SQL for complex joins and filtering
- UNION pattern for index utilization
- read_parquet() for temp file integration
- Python interop via .pl() method

### Why Three-Stage QC Cascade?
- Optimize for common case (exact match)
- Only pay cost when needed (fuzzy/BERT)
- Caching reduces repeated work
- Deletable errors (filter out)

### Why Pydantic extra='forbid'?
- Fail fast on config errors
- Catch typos before execution
- Enforce schema at load time
- Prevent silent config bugs

### Why Playwright for Downloads?
- Handles authentication
- JavaScript rendering
- Complex web scraping
- Headless automation

### Why ONNX for BioBERT?
- Faster inference than PyTorch
- No PyTorch dependency
- Portable across platforms
- Sentence-transformers backend support

## PERFORMANCE CHARACTERISTICS

### Memory Usage
- LazyFrames: Streaming where possible
- Parquet checkpoints: Memory-mapped I/O
- LRU cache: ~100MB disk cache
- DuckDB: In-memory caching of hot tables

### CPU Usage
- Polars: Multi-threaded LazyFrame operations
- DuckDB: Parallel query execution
- Multiprocessing: Pool.map for parallel YAML loading
- BioBERT: ONNX runtime (CPU-optimized)

### I/O Patterns
- Parquet: Efficient columnar storage
- Temp files: DuckDB read_parquet optimization
- Checkpoints: Resumable processing
- LRU cache: Disk-based caching

## SEE ALSO

- `AGENTS.md` — Root-level project overview and quick reference
- `lib/tablassert/AGENTS.md` — Coding style guide and module catalog
- `nix/AGENTS.md` — Nix packaging and environment setup
- `docs/AGENTS.md` — MkDocs documentation structure
