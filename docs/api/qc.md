# Quality Control (qc)

The `qc` module validates entity resolution mappings through a multi-stage pipeline: exact matching, fuzzy matching, and BERT semantic similarity.

## fullmap_audit()

Primary quality control function that filters entity mappings based on confidence criteria.

### Function Signature

```python
def fullmap_audit(
  lf: pl.LazyFrame,
  col: str,
  section_hash: str,
  config_file: str,
  out: str = "passed"
) -> pl.LazyFrame
```

### Parameters

**`lf: pl.LazyFrame`**

Input LazyFrame containing entity resolution results.

Expected columns:
- `original {col}` - Original text string
- `{col}` - Resolved CURIE
- `{col} name` - Preferred entity name

**`col: str`**

Base column name for entity resolution.

Example: If `col="subject"`, looks for:
- `"original subject"`
- `"subject"`
- `"subject name"`

**`out: str` (default: `"passed"`)**

Name of the boolean column indicating validation status.

Rows with `out=True` passed QC, `out=False` failed.

**`section_hash: str` / `config_file: str`**

Context fields used in QC failure logs for traceability.

### Return Value

Returns a Polars LazyFrame with only validated rows (where `out=True`). Failed pairs are logged with section/config/column context.

Removes the `out` column before returning.

### Three-Stage Pipeline

The function applies three validation stages in sequence:

#### Stage 1: Exact String Match

**Fast path for high-confidence mappings.**

```python
original == preferred_name
```

**Example passes:**
- Original: `"TP53"` → Preferred: `"TP53"` ✓
- Original: `"diabetes"` → Preferred: `"diabetes mellitus"` ✗ (goes to Stage 2)

**Performance:** O(1) string comparison

Before fuzzy matching, the function also applies rule-based pass-through checks for known safe patterns (for example CHEBI/PR/UniProtKB CURIE families and selected exception prefixes).

#### Stage 2: Fuzzy Matching

**Medium confidence using RapidFuzz.**

Two fuzzy matching algorithms:
1. **Ratio:** Overall string similarity
2. **Partial token sort ratio:** Combined token/subsequence matching

**Threshold:** Default 20% similarity (configurable)

```python
fuzz.ratio(original, preferred) >= 20
or fuzz.partial_token_sort_ratio(original, preferred) >= 20
```

**Example passes:**
- Original: `"breast ca"` → Preferred: `"breast cancer"` ✓
- Original: `"T53"` → Preferred: `"tumor protein p53"` ✗ (goes to Stage 3)

**Performance:** O(n) string operations, cached via `@DISKCACHE.memoize()`

#### Stage 3: BERT Semantic Similarity

**High confidence using BioBERT embeddings.**

1. **Encode** original and preferred name with BioBERT
2. **Compute** cosine similarity between embeddings
3. **Accept** if similarity >= 0.2 (20%)

```python
embeddings = BIOBERT.encode([original, preferred])
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
return similarity >= 0.2
```

**Example passes:**
- Original: `"lung carcinoma"` → Preferred: `"lung cancer"` ✓ (high semantic similarity)
- Original: `"random text"` → Preferred: `"diabetes"` ✗ (rejected, low similarity)

**Performance:** Expensive (ONNX inference), heavily cached

### BioBERT Model

**Model:** `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`

**Backend:** ONNX Runtime (CPU)

**Optimizations:**
- Graph optimization level: ALL
- ONNX session caching
- Disk cache for embeddings (~100MB LRU)

Lazy-loaded on first `BERT_audit()` call, then reused for subsequent calls.

### Disk Caching

All expensive operations are cached to disk:

```python
@DISKCACHE.memoize()
def fuzz_audit(...): ...

@DISKCACHE.memoize()
def BERT_audit(...): ...
```

**Cache location:** `./.cachassert` directory

**Cache strategy:** LRU eviction when size exceeds limit

**Why caching matters:**
- Fuzzy matching: 100-1000x speedup on repeated strings
- BERT inference: 10,000x speedup on repeated strings
- Enables iterative development without recomputing

### Example Usage

```python
from tablassert.qc import fullmap_audit
import polars as pl

# LazyFrame with entity resolution results
lf = pl.scan_parquet("resolved.parquet")

# Expected columns:
# - original subject
# - subject (CURIE)
# - subject name

# Run QC
validated = fullmap_audit(
  lf,
  col="subject",
  section_hash="tutorial-section",
  config_file="tutorial-table.yaml"
)

# Only rows that passed QC remain
# Rows with low-confidence mappings removed
```

### Pipeline Flow

```
Input: 1000 rows with entity mappings

Stage 1 (Exact): 700 pass → 300 pending
Stage 2 (Fuzzy): 250 pass → 50 pending
Stage 3 (BERT): 40 pass → 10 rejected

Output: 990 rows (700 + 250 + 40)
```

### Confidence Levels

| Stage | Method | Confidence | Use Case |
|-------|--------|-----------|----------|
| 1 | Exact match | Highest | Standardized IDs, acronyms |
| 2 | Fuzzy | Medium | Abbreviations, typos |
| 3 | BERT | High | Synonyms, paraphrases |

### Performance Characteristics

**Best case** (all exact matches):
- 1M rows: ~1 second

**Worst case** (all go to BERT):
- 1M rows: ~30 minutes (first run)
- 1M rows: ~10 seconds (cached)

**Typical case** (70% exact, 25% fuzzy, 5% BERT):
- 1M rows: ~2 minutes (first run)
- 1M rows: ~5 seconds (cached)

### Integration with Pipeline

QC is applied after entity resolution:

1. **Entity resolution** (`version4()`) - Maps text to CURIEs
2. **Quality control** (`fullmap_audit()`) - Validates mappings
3. **Export** - Only validated mappings in final output

This ensures knowledge graphs contain only high-confidence assertions.

## Next Steps

- **[Entity Resolution](fullmap.md)** - How mappings are generated
- **[Configuration](../configuration/table.md)** - Prioritize/avoid to improve resolution
