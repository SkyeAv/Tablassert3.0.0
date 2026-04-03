# Batch Resolution (lib)

The `lib` module exposes `resolve_many()`, a high-level convenience function for resolving an iterable of entity strings to CURIEs without requiring manual LazyFrame construction, NLP preprocessing, or DuckDB shard management.

It wraps the lower-level [`resolve()`](fullmap.md) pipeline — applying `level_one` and `level_two` normalization, opening all 16 DuckDB shard connections, executing entity resolution, and returning results as a plain Python dictionary.

## resolve_many()

Standalone batch entity resolution function. Accepts a column name, an iterable of text strings, and a path to the datassert database, then returns resolved CURIEs and metadata as a dictionary of lists.

### Function Signature

```python
def resolve_many(
    col: str,
    entities: Iterable[str],
    datassert: Path,
    taxon: Optional[str] = None,
    prioritize: Optional[list[Categories]] = None,
    avoid: Optional[list[Categories]] = None,
    column_context: bool = True,
) -> dict[str, list[str]]
```

### Parameters

**`col: str`**

Column name used internally to label the Polars Series and DataFrame columns during resolution. This name propagates through the NLP and resolution pipeline and determines the keys in the returned dictionary.

For example, if `col="gene"`, the returned dictionary will contain keys like `"gene"`, `"gene name"`, `"gene category"`, etc.

**`entities: Iterable[str]`**

An iterable of text strings to resolve. Each string is treated as a candidate entity name that will be normalized and matched against the datassert synonym database. Accepts any iterable — lists, tuples, generators, sets, etc.

Examples: `["TP53", "BRCA1", "EGFR"]`, `("aspirin", "ibuprofen")`, or a generator expression.

**`datassert: Path`**

Filesystem path to the root of the datassert database directory. The function expects a `data/` subdirectory containing 16 DuckDB shard files (`0.duckdb` through `15.duckdb`).

Each shard contains:
- Synonym mappings (text → CURIE)
- Preferred entity names
- Biolink categories
- NCBI Taxon IDs
- Source databases and versions

**`taxon: Optional[str]` (default: `None`)**

Optional NCBI Taxon ID for filtering results to a specific organism.

Example: `"9606"` restricts matches to human-specific entities. When `None`, no taxon filtering is applied and matches from all organisms are returned.

**`prioritize: Optional[list[Categories]]` (default: `None`)**

Optional list of Biolink categories to prefer when multiple matches exist for the same input term. Categories listed here receive higher ranking scores during resolution.

Example: `[Categories.Gene, Categories.Protein]` prefers gene and protein mappings over other categories like diseases or chemicals.

**`avoid: Optional[list[Categories]]` (default: `None`)**

Optional list of Biolink categories to exclude from results entirely. Any match belonging to an avoided category is filtered out before ranking.

Example: `[Categories.Gene]` prevents gene mappings from appearing in the output, even if they would otherwise be the best match.

**`column_context: bool` (default: `True`)**

Controls category-frequency tie-breaking when multiple matches exist for a term. When `True`, the resolution query adds a category frequency score and prefers the category that appears most frequently across all terms in the batch. When `False`, frequency-based tie-breaking is disabled.

This is useful when resolving a column of related entities (e.g., all genes) — the shared context helps disambiguate terms that map to multiple categories.

### Return Value

Returns a `dict[str, list[str]]` where each key is a column name and each value is a list of resolved values. The dictionary is produced by calling `polars.DataFrame.to_dict(as_series=False)` on the collected resolution output.

The returned dictionary contains the following keys (where `{col}` is the value of the `col` parameter):

| Key | Description | Example Value |
|-----|-------------|---------------|
| `{col}` | CURIE identifier | `"HGNC:11998"` |
| `{col} name` | Preferred entity name | `"TP53"` |
| `{col} category` | Biolink category (prefixed) | `"biolink:Gene"` |
| `{col} taxon` | NCBI Taxon ID (prefixed) | `"NCBITaxon:9606"` |
| `{col} source` | Source database | `"HGNC"` |
| `{col} source version` | Database version | `"2025-01"` |
| `{col} nlp level` | NLP processing level used for match | `0` or `1` |

**Important:** Only entities that successfully resolve to a CURIE are included in the output. Unresolved entities are filtered out by `resolve()`. The returned lists may therefore be shorter than the input iterable.

### Pipeline Internals

`resolve_many()` executes the following steps internally:

1. **Series construction** — Wraps the input iterable in a `pl.Series` with the given column name, then converts to a single-column `pl.LazyFrame`.

2. **NLP normalization** — Applies `level_one()` (whitespace stripping + lowercasing) and `level_two()` (non-word character removal via `\W+`) to produce the two normalized columns required by `resolve()`.

3. **DuckDB connection management** — Opens all 16 shard connections inside a `contextlib.ExitStack`, ensuring every connection is properly closed when resolution completes or if an error occurs.

4. **Entity resolution** — Delegates to `fullmap.resolve()` which queries the sharded DuckDB database, ranks matches by category priority, preferred-name exactness, NLP level, and category frequency, then deduplicates to one CURIE per input string.

5. **Collection and conversion** — Collects the lazy result into an eager `pl.DataFrame` and converts to a Python dictionary via `to_dict(as_series=False)`.

### Example Usage

#### Basic Gene Resolution

```python
from pathlib import Path
from tablassert.lib import resolve_many
from tablassert.enums import Categories

datassert: Path = Path("/path/to/datassert")

result: dict[str, list[str]] = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1", "EGFR", "KRAS"],
    datassert=datassert,
    taxon="9606",
    prioritize=[Categories.Gene],
)

# result["gene"]          → ["HGNC:11998", "HGNC:1100", ...]
# result["gene name"]     → ["TP53", "BRCA1", ...]
# result["gene category"] → ["biolink:Gene", "biolink:Gene", ...]
```

#### Disease Resolution With Category Avoidance

```python
from pathlib import Path
from tablassert.lib import resolve_many
from tablassert.enums import Categories

datassert: Path = Path("/path/to/datassert")

result: dict[str, list[str]] = resolve_many(
    col="disease",
    entities=["diabetes mellitus", "breast cancer", "alzheimer disease"],
    datassert=datassert,
    avoid=[Categories.Gene, Categories.Protein],
)

# result["disease"]          → ["MONDO:0005015", ...]
# result["disease name"]     → ["diabetes mellitus", ...]
# result["disease category"] → ["biolink:Disease", ...]
```

#### Chemical Resolution Without Column Context

```python
from pathlib import Path
from tablassert.lib import resolve_many

datassert: Path = Path("/path/to/datassert")

result: dict[str, list[str]] = resolve_many(
    col="chemical",
    entities=["aspirin", "metformin", "ibuprofen"],
    datassert=datassert,
    column_context=False,
)
```

#### Consuming Results

```python
import polars as pl
from pathlib import Path
from tablassert.lib import resolve_many

datassert: Path = Path("/path/to/datassert")

result: dict[str, list[str]] = resolve_many(
    col="gene",
    entities=["TP53", "BRCA1"],
    datassert=datassert,
    taxon="9606",
)

# Convert back to a Polars DataFrame
df: pl.DataFrame = pl.DataFrame(result)

# Or iterate over resolved pairs
for curie, name in zip(result["gene"], result["gene name"]):
    print(f"{name} → {curie}")
```

### Comparison With resolve()

| Aspect | `resolve_many()` | `resolve()` |
|--------|-------------------|-------------|
| **Module** | `tablassert.lib` | `tablassert.fullmap` |
| **Input** | Plain iterable of strings | Pre-normalized `pl.LazyFrame` |
| **NLP** | Applied automatically | Must be applied upstream |
| **Connections** | Managed internally via `ExitStack` | Must be opened externally |
| **Output** | `dict[str, list[str]]` | `pl.LazyFrame` |
| **Logging** | Uses default (`log=True`) | Configurable |
| **Context params** | Not exposed (`section_hash`, `config_file`, `tag`) | Fully configurable |
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
- Output column: `{col} two`

Both levels are queried during resolution. Level one (exact case-insensitive match) is preferred; level two is used as a fallback for terms with punctuation or special characters.

### Error Handling

- If the `datassert` path does not contain the expected shard files, `duckdb.connect()` will raise an `IOException`.
- If `entities` is empty, the function returns a dictionary with empty lists for all output columns.
- The `ExitStack` ensures all 16 DuckDB connections are closed even if resolution raises an exception.
- Unresolved entities are silently filtered from the output (logged at INFO level by default via `resolve()`).

## Integration

`resolve_many()` is a self-contained entry point. It does not require any prior setup beyond having a datassert database available. For full pipeline builds, use the CLI (`tablassert build-knowledge-graph`) which orchestrates resolution through the `Tcode` class.

## Next Steps

- **[Entity Resolution](fullmap.md)** — Lower-level `resolve()` function details
- **[Quality Control](qc.md)** — Multi-stage validation of resolved entities
- **[Configuration](../configuration/table.md)** — YAML-driven entity resolution settings
