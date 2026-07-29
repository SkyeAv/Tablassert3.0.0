# Batch Resolution (lib)

The `lib` module exposes `resolve_many()`, a high-level convenience function for resolving an iterable of entity strings to CURIEs without requiring manual LazyFrame construction or NLP preprocessing.

It wraps the lower-level [`resolve()`](fullmap.md) pipeline — preserving the original input text, applying `level_one` and `level_two` normalization, querying the embedded fullmap redb database, executing entity resolution, optionally running the QC audit (when `qc=True`), and returning results as a plain Python list of row dictionaries.

## resolve_many()

Standalone batch entity resolution function. Accepts a column name, an iterable of text strings, and a path to the fullmap database, then returns resolved CURIEs and metadata as a list of row dictionaries.

### Function Signature

```python
def resolve_many(
    col: str,
    entities: Iterable[str],
    fullmap: Path,
    taxon: Optional[str] = None,
    prioritize: Optional[list[Categories]] = None,
    avoid: Optional[list[Categories]] = None,
    qc: bool = False,
    column_context: bool = True,
) -> list[dict[str, Any]]
```

### Parameters

**`col: str`**

Column name used internally to label the Polars Series and DataFrame columns during resolution. This name propagates through the NLP and resolution pipeline and determines the keys in each returned row dictionary.

For example, if `col="gene"`, each returned row dictionary will contain keys like `"gene"`, `"gene_name"`, `"gene_category"`, etc.

**`entities: Iterable[str]`**

An iterable of text strings to resolve. Each string is treated as a candidate entity name that will be normalized and matched against the fullmap synonym database. Accepts any iterable — lists, tuples, generators, sets, etc.

Examples: `["TP53", "BRCA1", "EGFR"]`, `("aspirin", "ibuprofen")`, or a generator expression.

**`fullmap: Path`**

Filesystem path to the fullmap redb file, or a base directory containing it (resolved via `fullmap_db_path()` — see [Fullmap](../fullmap.md)).

The database contains:
- Synonym mappings (text → CURIE)
- Preferred entity names
- Biolink categories
- NCBI Taxon IDs
- Source databases and versions

**`taxon: Optional[str]` (default: `None`)**

Optional NCBI Taxon ID for filtering results to a specific organism.

Example: `"9606"` restricts **gene** matches to human-specific entries; non-gene categories (e.g., diseases, chemicals) are returned regardless of taxon. When `None`, no taxon filtering is applied and matches from all organisms are returned.

**`prioritize: Optional[list[Categories]]` (default: `None`)**

Optional list of Biolink categories to prefer when multiple matches exist for the same input term. Categories listed here receive higher ranking scores during resolution.

Example: `[Categories.GENE, Categories.PROTEIN]` prefers gene and protein mappings over other categories like diseases or chemicals.

**`avoid: Optional[list[Categories]]` (default: `None`)**

Optional list of Biolink categories to exclude from results entirely. Any match belonging to an avoided category is filtered out before ranking.

Example: `[Categories.GENE]` prevents gene mappings from appearing in the output, even if they would otherwise be the best match.

**`column_context: bool` (default: `True`)**

Controls category-frequency tie-breaking when multiple matches exist for a term. When `True`, the deduplication stage adds a category-frequency score (computed in Polars after the SQL query) and prefers the category that appears most frequently across all matched terms in the batch. When `False`, frequency-based tie-breaking is disabled.

This is useful when resolving a column of related entities (e.g., all genes) — the shared context helps disambiguate terms that map to multiple categories.

**`qc: bool` (default: `False`)**

When `True`, runs the QC audit stage after entity resolution. The QC pipeline validates mappings through a three-stage audit: exact match, fuzzy matching via rapidfuzz, and BioBERT sentence embeddings with cosine similarity. Mappings that fail all three stages are dropped from the returned list (in addition to the unresolved-entity filtering performed by `resolve()`). Requires the QC runtime to be installed (`tablassert[qc]`); see [Quality Control](qc.md) for the stage thresholds and backend.

### Return Value

Returns a `list[dict[str, Any]]` — one dictionary per resolved entity. The list is produced by calling `polars.DataFrame.to_dicts()` on the collected resolution output.

Each dictionary contains the following keys (where `{col}` is the value of the `col` parameter):

| Key | Description | Example Value |
|-----|-------------|---------------|
| `original_{col}` | Original input text before normalization | `"TP53"` |
| `{col}` | CURIE identifier | `"HGNC:11998"` |
| `{col}_name` | Preferred entity name | `"TP53"` |
| `{col}_category` | Biolink category (prefixed) | `"biolink:Gene"` |
| `{col}_taxon` | NCBI Taxon ID (prefixed) | `"NCBITaxon:9606"` |
| `{col}_source` | Source database | `"HGNC"` |
| `{col}_source_version` | Database version | `"2025-01"` |
| `{col}_nlp_level` | NLP processing level used for match | `1` or `2` |

**Important:** Only entities that successfully resolve to a CURIE are included in the output. Unresolved entities are filtered out by `resolve()`. The returned list may therefore be shorter than the input iterable.

### Pipeline Internals

`resolve_many()` executes the following steps internally:

1. **Series construction** — Wraps the input iterable in a `pl.Series` with the given column name, then converts to a single-column `pl.LazyFrame`.

2. **Original column capture** — Copies the raw input column into `original_{col}` (pristine source value) via `column(lf, add("original_", col), col)`, and into `{col}_pre_resolution` (the value fed to resolution) via `column(lf, add(col, "_pre_resolution"), col)`. The `original_{col}` column is returned; `{col}_pre_resolution` is used internally for QC and dropped from the result to mirror edge output.

3. **NLP normalization** — Applies `level_one()` (whitespace stripping + lowercasing) and `level_two()` (non-word character removal via `\W+`) to produce the two normalized columns required by `resolve()`.

4. **Path resolution** — Resolves the `fullmap` argument to the actual redb file via `fullmap_db_path()`.

5. **Entity resolution** — Delegates to `fullmap.resolve()` which queries the embedded redb database, ranks matches by category priority, preferred-name exactness, NLP level, and category frequency, then deduplicates to one CURIE per input string.

6. **QC audit (optional)** — When `qc=True`, runs `fullmap_audit()` on the resolved LazyFrame. Rows that fail all three audit stages are dropped from the result.

7. **Collection and conversion** — Collects the lazy result into an eager `pl.DataFrame` and converts to a list of row dictionaries via `to_dicts()`.

### Example Usage

#### Basic Gene Resolution

```python
from pathlib import Path
from typing import Any
from tablassert.lib import resolve_many
from tablassert.biolink import Categories

fullmap: Path = Path("/path/to/fullmap")

result: list[dict[str, Any]] = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1", "EGFR", "KRAS"],
    fullmap=fullmap,
    taxon="9606",
    prioritize=[Categories.GENE],
)

# result[0] → {"original_gene": "TP53", "gene": "HGNC:11998", "gene_name": "TP53", ...}
# result[1] → {"original_gene": "BRCA1", "gene": "HGNC:1100", "gene_name": "BRCA1", ...}
```

#### Disease Resolution With Category Avoidance

```python
from pathlib import Path
from typing import Any
from tablassert.lib import resolve_many
from tablassert.biolink import Categories

fullmap: Path = Path("/path/to/fullmap")

result: list[dict[str, Any]] = resolve_many(
    col="disease",
    entities=["diabetes mellitus", "breast cancer", "alzheimer disease"],
    fullmap=fullmap,
    avoid=[Categories.GENE, Categories.PROTEIN],
)

# result[0] → {"original_disease": "diabetes mellitus", "disease": "MONDO:0005015", ...}
# result[1] → {"original_disease": "breast cancer", "disease_name": "breast cancer", ...}
```

#### Chemical Resolution Without Column Context

```python
from pathlib import Path
from typing import Any
from tablassert.lib import resolve_many

fullmap: Path = Path("/path/to/fullmap")

result: list[dict[str, Any]] = resolve_many(
    col="chemical",
    entities=["aspirin", "metformin", "ibuprofen"],
    fullmap=fullmap,
    column_context=False,
)
```

#### Consuming Results

```python
import polars as pl
from pathlib import Path
from typing import Any
from tablassert.lib import resolve_many

fullmap: Path = Path("/path/to/fullmap")

result: list[dict[str, Any]] = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1"],
    fullmap=fullmap,
    taxon="9606",
)

# Convert back to a Polars DataFrame
df: pl.DataFrame = pl.DataFrame(result)

# Or iterate over resolved rows
for row in result:
    print(f"{row['gene_name']} → {row['gene']}")
```

### Comparison With resolve()

| Aspect | `resolve_many()` | `resolve()` |
|--------|-------------------|-------------|
| **Module** | `tablassert.lib` | `tablassert.fullmap` |
| **Input** | Plain iterable of strings | Pre-normalized `pl.LazyFrame` |
| **NLP** | Applied automatically | Must be applied upstream |
| **Path resolution** | Resolved internally via `fullmap_db_path()` | Caller must pass the resolved redb path |
| **Output** | `list[dict[str, Any]]` | `pl.LazyFrame` |
| **Logging** | Uses default (`log=True`) | Configurable |
| **Context params** | `column_context` exposed; `section_hash`, `config_file`, `tag` not exposed | Fully configurable |
| **Use case** | Standalone batch lookups, scripting, notebooks | Internal pipeline integration |

`resolve_many()` is designed for ad-hoc and programmatic use — scripts, notebooks, and one-off lookups. For pipeline integration where you need full control over logging, context metadata, and lazy evaluation, use `resolve()` directly.

### NLP Processing

`resolve_many()` applies both NLP normalization levels before resolution:

**Level one** — `level_one(lf, col)`:
- Strips leading/trailing whitespace
- Converts to lowercase
- Output column: `{col}` (overwrites the original)

**Level two** — `level_two(lf, col)`:
- Removes all non-word characters (`\W+` → `""`) from the level-one result
- Output column: `{col}_two`

Both levels are queried during resolution. Level one (exact case-insensitive match) is preferred; level two is used as a fallback for terms with punctuation or special characters.

### Error Handling

- If the `fullmap` path does not resolve to a valid redb file, or the file's schema tag doesn't match the expected version, the Rust extension raises a `RuntimeError`.
- If `entities` is empty, the function returns `[]`.
- Unresolved entities are silently filtered from the output (logged at INFO level by default via `resolve()`).

## Integration

`resolve_many()` is a self-contained entry point. It does not require any prior setup beyond having a fullmap database available. For full pipeline builds, use the CLI (`tablassert build-kg`) which orchestrates resolution through the `Tcode` class.

## Next Steps

- **[Entity Resolution](fullmap.md)** — Lower-level `resolve()` function details
- **[Quality Control](qc.md)** — Multi-stage validation of resolved entities
- **[Configuration](../configuration/table.md)** — YAML-driven entity resolution settings
