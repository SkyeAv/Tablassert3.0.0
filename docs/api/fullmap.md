# Entity Resolution (fullmap)

The `fullmap` module provides entity resolution functionality, mapping text strings to standardized biological entities (CURIEs).

## version4()

Primary entity resolution function using DuckDB queries against the dbssert database.

### Function Signature

```python
def version4(
  lf: pl.LazyFrame,
  col: str,
  conn: object,
  taxon: Optional[str] = None,
  prioritize: Optional[list[Categories]] = None,
  avoid: Optional[list[Categories]] = None,
  log: bool = True,
  section_hash: Optional[str] = None,
  config_file: Optional[str] = None,
  column_context: bool = True,
  tag: str = " one"
) -> pl.LazyFrame
```

### Parameters

**`lf: pl.LazyFrame`**

Input LazyFrame containing the data to process. Internally collected at explicit collection points for DuckDB queries and joins.

**`col: str`**

Column name containing text strings to resolve.

**`conn: object`**

DuckDB connection to the entity resolution database.

This database contains:
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

**`tag: str` (default: `" one"`)**

Suffix for NLP processing level column.

The function looks for both:
- `col` (original text, case-preserved)
- `col + tag` (normalized text, typically lowercase)

Default `" one"` means it uses level-one text processing (lowercase, stripped).

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

1. **Builds an in-memory term table** by collecting distinct terms from both NLP levels and registering them in DuckDB as `PARQUET` via `conn.register("PARQUET", df.to_arrow())`.

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
from tablassert.fullmap import version4
from tablassert.enums import Categories
import duckdb
import polars as pl

# Open DuckDB connection
conn = duckdb.connect("/data/dbssert.duckdb", read_only=True)

# LazyFrame with data to resolve
lf = pl.scan_parquet("data.parquet")

# Resolve gene symbols to CURIEs
result = version4(
  lf=lf,
  col="gene_symbol",
  conn=conn,
  taxon="9606",  # Human only
  prioritize=[Categories.Gene],
  avoid=[Categories.Protein],
  log=True,
  section_hash="tutorial-section",
  config_file="tutorial-table.yaml",
  column_context=True,
  tag=" one"
)

# Result LazyFrame includes:
# - gene_symbol: "HGNC:11998"
# - gene_symbol name: "TP53"
# - gene_symbol category: "biolink:Gene"
# - etc.
```

### NLP Processing Levels

**Level 0** (exact case):
- Column: `col`
- Matches preserve case
- Preferred for acronyms, gene symbols

**Level 1** (normalized):
- Column: `col + " one"`
- Lowercased, whitespace stripped
- Preferred for disease names, free text

The function tries level 0 first, then level 1.

Rows without a valid CURIE are filtered from the returned frame.

### Case-Dependent Behavior

"TP53" vs "tp53":
- Level 0: Only matches "TP53" synonym
- Level 1: Matches any case variant

This preserves specificity for case-sensitive identifiers while allowing fuzzy matching for general terms.

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
