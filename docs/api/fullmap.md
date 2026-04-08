# Entity Resolution (fullmap)

The `fullmap` module provides entity resolution functionality, mapping text strings to standardized biological entities (CURIEs).

## resolve()

Primary entity resolution function using DuckDB queries against the datassert database.

### Function Signature

```python
def resolve(
  lf: pl.LazyFrame,
  col: str,
  conns: list[object],
  taxon: Optional[str] = None,
  prioritize: Optional[list[Categories]] = None,
  avoid: Optional[list[Categories]] = None,
  log: bool = True,
  section_hash: Optional[str] = None,
  config_file: Optional[str] = None,
  column_context: bool = True,
  tag: str = " two"
) -> pl.LazyFrame
```

### Parameters

**`lf: pl.LazyFrame`**

Input LazyFrame containing the data to process. Internally collected at explicit collection points for DuckDB queries and joins.

**`col: str`**

Column name containing text strings to resolve.

**`conns: list[object]`**

List of 12 DuckDB shard connections to the datassert database.

Each shard contains:
- Synonym mappings (text → CURIE)
- Preferred entity names
- Biolink categories
- NCBI Taxon IDs
- Source databases and versions

**`taxon: Optional[str]`**

Optional NCBI Taxon ID for filtering results.

Example: `"9606"` filters to human-specific entities.

**`prioritize: Optional[list[Categories]]`**

Optional list of Biolink categories to prefer when multiple matches exist.

Example: `[Categories.Gene, Categories.Protein]` prefers gene/protein mappings.

**`avoid: Optional[list[Categories]]`**

Optional list of Biolink categories to exclude from results.

Example: `[Categories.Gene]` prevents gene mappings.

**`log: bool` (default: `True`)**

Controls unmatched-value logging. When enabled, unresolved terms are logged with section/config/column context.

**`section_hash: Optional[str]` / `config_file: Optional[str]`**

Optional context fields used for operational logging when unmatched values are encountered.

**`column_context: bool` (default: `True`)**

Controls category-frequency tie-breaking when multiple matches exist for a term. When `True`, the query result adds a category frequency score and prefers more frequent category hits.

**`tag: str` (default: `" two"`)**

Suffix appended to `col` to locate the `level_two` output column.

`resolve()` expects the LazyFrame to already have two NLP columns applied upstream:
- `col` — the `level_one` output (whitespace stripped, lowercased)
- `col + tag` — the `level_two` output (non-word characters removed via `\W+`)

The default `" two"` matches `level_two`'s default tag.

### Return Value

Returns a Polars LazyFrame with these columns added:

| Column | Description | Example |
|--------|-------------|---------|
| `{col}` | CURIE identifier | `"HGNC:11998"` |
| `{col} name` | Preferred entity name | `"TP53"` |
| `{col} category` | Biolink category | `"biolink:Gene"` |
| `{col} taxon` | NCBI Taxon ID | `"NCBITaxon:9606"` |
| `{col} source` | Source database | `"HGNC"` |
| `{col} source version` | Database version | `"2025-01"` |
| `{col} nlp level` | NLP processing level | `0` or `1` |

### DuckDB Query

The function executes a SQL query that:

1. **Builds an in-memory term table** by collecting terms from both NLP levels, deduplicating by keeping first occurrences for deterministic ordering, and registering them in DuckDB as `PARQUET` via `conn.register("PARQUET", df.to_arrow())`.

2. **Ranks matches** by:
   - Category priority (if `prioritize` specified)
   - Preferred-name exactness (case-insensitive exact match of normalized term to preferred name)
   - NLP level (exact case match preferred over normalized)
   - Category frequency (if `column_context=True`)

3. **Filters by:**
   - Taxon ID (if specified)
   - Category avoidance (if specified)

4. **Deduplicates** to one CURIE per input string

### Example Usage

```python
from tablassert.fullmap import resolve
from tablassert.enums import Categories
import duckdb
import polars as pl

# Open all 12 shard connections
datassert_dir = "/path/to/datassert"
conns = [
    duckdb.connect(f"{datassert_dir}/data/{i}.duckdb", read_only=True)
    for i in range(12)
]

# LazyFrame with data to resolve
lf = pl.scan_parquet("data.parquet")

# Resolve gene symbols to CURIEs
result = resolve(
  lf=lf,
  col="gene_symbol",
  conns=conns,
  taxon="9606",  # Human only
  prioritize=[Categories.Gene],
  avoid=[Categories.Protein],
  log=True,
  section_hash="tutorial-section",
  config_file="tutorial-table.yaml",
  column_context=True,
)

# Result LazyFrame includes:
# - gene_symbol: "HGNC:11998"
# - gene_symbol name: "TP53"
# - gene_symbol category: "biolink:Gene"
# - etc.
```

### Mapping a Python List

A common entry point for programmatic use is resolving a plain Python list of terms:

```python
import duckdb
import polars as pl
from tablassert.fullmap import resolve
from tablassert.nlp import level_one, level_two
from tablassert.enums import Categories

# Open all 12 shard connections
datassert_dir = "/path/to/datassert"
conns = [
    duckdb.connect(f"{datassert_dir}/data/{i}.duckdb", read_only=True)
    for i in range(12)
]

# Map a list of gene symbols to CURIEs
genes = ["TP53", "BRCA1", "EGFR", "KRAS"]
lf = pl.LazyFrame({"gene": genes})

# Apply NLP normalization (required before resolve)
lf = level_one(lf, "gene")   # lowercase + strip
lf = level_two(lf, "gene")   # remove non-word chars → "gene two" column

result = resolve(
    lf=lf,
    col="gene",
    conns=conns,
    taxon="9606",               # Human only
    prioritize=[Categories.Gene],
    log=False,
).collect()

print(result.select(["gene", "gene name", "gene category"]))
```

### NLP Processing Levels

`resolve()` requires that `level_one` and `level_two` have been applied to the LazyFrame before calling it:

**`level_one` output** (column: `col`):
- Whitespace stripped, lowercased
- Queried first; preferred for acronyms and gene symbols

**`level_two` output** (column: `col + " two"`):
- All non-word characters removed (`\W+` → `""`) from the `level_one` result
- Used as fallback when `level_one` produces no match
- Preferred for disease names and free text

Rows without a valid CURIE are filtered from the returned frame.

### Case-Dependent Behavior

`"TP53"` vs `"tp53"`:
- After `level_one`: both become `"tp53"` — matches any case variant
- `level_two` further strips punctuation, helping with hyphenated or slash-delimited names

This preserves specificity for case-sensitive identifiers while allowing looser matching for general terms.

### Provenance Tracking

Every resolved entity includes:
- **Source database** (HGNC, MONDO, ChEBI, etc.)
- **Source version** (database snapshot date)
- **Matched synonym** (which text triggered the match)

This enables auditing and quality control.

## Integration with QC

Entity resolution output is validated by `fullmap_audit()` from the `qc` module before being included in the knowledge graph.

See [Quality Control](qc.md) for details.

## Next Steps

- **[Quality Control](qc.md)** - Multi-stage validation
- **[Configuration](../configuration/table.md)** - How to specify prioritize/avoid in YAML
