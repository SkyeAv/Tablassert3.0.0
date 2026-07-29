# Quality Control (qc)

The `qc` module validates entity resolution mappings through a multi-stage pipeline: exact matching, fuzzy matching, and BioBERT semantic similarity.

QC runtime support is optional. Install `tablassert[qc]` to enable it — the extra pulls `scikit-learn` and `sentence-transformers` (`torch` and `numpy` arrive transitively); `rapidfuzz` is a core dependency and is always available. If the audit stage runs without `sentence-transformers` installed, `fullmap_audit()` raises `QcRuntimeMissingError`.

## fullmap_audit()

Primary quality control function that filters entity mappings based on confidence criteria.

### Function Signature

```python
def fullmap_audit(
  lf: pl.LazyFrame,
  col: str,
  section_hash: str,
  config_file: str,
  out: str = "passed",
  log: bool = True,
) -> pl.LazyFrame
```

### Parameters

**`lf: pl.LazyFrame`**

Input LazyFrame containing entity resolution results.

Expected columns:
- `{col}_pre_resolution` - Original (pre-resolution) text string
- `{col}` - Resolved CURIE
- `{col}_name` - Preferred entity name

**`col: str`**

Base column name for entity resolution.

Example: If `col="subject"`, looks for:
- `"subject_pre_resolution"`
- `"subject"`
- `"subject_name"`

**`out: str` (default: `"passed"`)**

Name of the boolean column indicating validation status (used internally).

Rows with `out=True` passed QC, `out=False` failed.

**`log: bool` (default: `True`)**

Controls whether failed QC rows are logged.

**`section_hash: str` / `config_file: str`**

Context fields used in QC failure logs for traceability.

### Return Value

Returns a Polars LazyFrame containing only rows whose `col` value passed QC (via a semi-join on the surviving CURIEs). Failed pairs are logged with section/config/column context and their fuzzy/BioBERT scores.

### Three-Stage Pipeline

The function applies three validation stages in sequence. Each stage progressively filters out correct resolutions and leaves suspected errors for the next stage.

#### Stage 1: Exact Match & Rule-Based Pass-Through

**Fast path for high-confidence mappings.** A row passes if any of these hold:

```python
original == preferred_name
```

- The pre-resolution text exactly equals the resolved preferred name.
- The resolved CURIE matches an exempt prefix (`CHEBI`, `PR`, `UniProtKB`, `NCBIGene`, `UMLS`, `UNII`, `PUBCHEM`, `MONDO`).
- The original text contains `:` (already looks like a CURIE).
- The preferred name matches an exception prefix (`^LOC` or `^si:`).

**Example passes:**
- Original: `"TP53"` → Preferred: `"TP53"` ✓
- Original: `"diabetes"` → Preferred: `"diabetes mellitus"` ✗ (goes to Stage 2)

**Performance:** O(1) string comparison per row.

#### Stage 2: Fuzzy Matching

**Medium confidence using RapidFuzz (batched via `rapidfuzz.process.cpdist`).**

Two fuzzy matching algorithms:
1. **Ratio:** Overall string similarity
2. **Partial token sort ratio:** Combined token/subsequence matching

**Thresholds:** `fuzz.ratio` >= 70 OR `fuzz.partial_token_sort_ratio` >= 80

```python
fuzz.ratio(original, preferred) >= 70
or fuzz.partial_token_sort_ratio(original, preferred) >= 80
```

**Example passes:**
- Original: `"breast ca"` → Preferred: `"breast cancer"` ✓
- Original: `"T53"` → Preferred: `"tumor protein p53"` ✗ (goes to Stage 3)

**Performance:** O(n) string operations, batched.

#### Stage 3: BioBERT Semantic Similarity

**High confidence using BioBERT embeddings.**

1. **Encode** original and preferred name with BioBERT (sentence-transformers)
2. **Compute** cosine similarity between embeddings (scikit-learn)
3. **Accept** if similarity >= 0.5

```python
embeddings = get_biobert().encode(originals + preferreds)
similarity = cosine_similarity(embeddings[:n], embeddings[n:]).diagonal()
return similarity >= 0.5
```

**Example passes:**
- Original: `"lung carcinoma"` → Preferred: `"lung cancer"` ✓ (high semantic similarity)
- Original: `"random text"` → Preferred: `"diabetes"` ✗ (rejected, low similarity)

**Performance:** Expensive (transformer inference); the model is loaded once and cached.

### BioBERT Model

**Model:** `pritamdeka/BioBERT-mnli-snli-scitail-mednli-stsb`

**Backend:** [sentence-transformers](https://www.sbert.net/) (PyTorch). Embeddings are compared with scikit-learn's `cosine_similarity`.

**Lazy-loaded** on the first `fullmap_audit()` call that reaches the embedding stage via `get_biobert()`, then cached globally for the lifetime of the process.

### Model Caching

`get_biobert()` loads the model from the local cache when present; otherwise it downloads `pritamdeka/BioBERT-mnli-snli-scitail-mednli-stsb` and saves it for future runs.

**Cache location:** `.tablassert/biobert/` on disk (`qc.MODEL`); the loaded model object is also cached in memory for the lifetime of the process.

**Why caching matters:**
- Fuzzy matching: large speedup on repeated strings (batched)
- BioBERT inference: avoids re-encoding repeated strings and re-downloading the model
- Enables iterative development without recomputing

### Example Usage

```python
from tablassert.qc import fullmap_audit
import polars as pl

# LazyFrame with entity resolution results
lf = pl.scan_parquet("resolved.parquet")

# Expected columns:
# - subject_pre_resolution
# - subject (CURIE)
# - subject_name

# Run QC
validated = fullmap_audit(
  lf,
  col="subject",
  section_hash="tutorial-section",
  config_file="tutorial-table.yaml",
)

# Only rows that passed QC remain
# Rows with low-confidence mappings removed
```

### Pipeline Flow

```
Input: 1000 rows with entity mappings

Stage 1 (Exact): 700 pass → 300 pending
Stage 2 (Fuzzy): 250 pass → 50 pending
Stage 3 (BioBERT): 40 pass → 10 rejected

Output: 990 rows (700 + 250 + 40)
```

### Confidence Levels

| Stage | Method | Confidence | Use Case |
|-------|--------|-----------|----------|
| 1 | Exact match / rule-based | Highest | Standardized IDs, acronyms, CURIE-like inputs |
| 2 | Fuzzy | Medium | Abbreviations, typos |
| 3 | BioBERT | High | Synonyms, paraphrases |

### Rejection Logging

When `log=True`, each rejected CURIE is logged at INFO level with its context and the scores that caused the rejection: `curie`, `original`, `preferred`, `col`, `fuzz` (partial token sort ratio), `config`, `hash`, and — when the BioBERT stage ran — `bert` (cosine similarity).

### Integration with Pipeline

QC is applied after entity resolution when QC is enabled (the `build-kg --qc` flag, or `resolve_many(..., qc=True)`):

1. **Entity resolution** (`resolve()`) - Maps text to CURIEs
2. **Quality control** (`fullmap_audit()`) - Validates mappings
3. **Export** - Only validated mappings in final output

This ensures knowledge graphs contain only high-confidence assertions.

## Next Steps

- **[Entity Resolution](fullmap.md)** - How mappings are generated
- **[Configuration](../configuration/table.md)** - Prioritize/avoid to improve resolution
