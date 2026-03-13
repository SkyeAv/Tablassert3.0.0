from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tablassert.utils import DISKCACHE
from tablassert.log import logger
from typing import Optional
from rapidfuzz import fuzz
import onnxruntime as ort
from pathlib import Path
from operator import add
from operator import ge
from operator import eq
import polars as pl

SESSION_OPTS: object = ort.SessionOptions()
SESSION_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # pyright: ignore

MODEL: Path = Path("./.onnxassert/")
MODEL_BACKEND: str = "onnx"
MODEL_KWARGS: dict[str, object] = {
  "provider": "CPUExecutionProvider",
  "session_options": SESSION_OPTS,
}

# TODO: Explore Best Model For QC
BIOBERT: Optional[object] = None
def get_biobert() -> object:
  # ? Lazy-loads BioBERT once on first BERT_audit call, then caches globally
  global BIOBERT
  if BIOBERT:
    return BIOBERT
  elif not BIOBERT and MODEL.exists():
    BIOBERT = SentenceTransformer(str(MODEL), backend=MODEL_BACKEND, model_kwargs=MODEL_KWARGS) # pyright: ignore
  else:
    BIOBERT = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", backend=MODEL_BACKEND, model_kwargs=MODEL_KWARGS) # pyright: ignore
    MODEL.mkdir(parents=True, exist_ok=True)
    BIOBERT.save(MODEL) # pyright: ignore
  return BIOBERT

@DISKCACHE.memoize() # pyright: ignore
def fuzz_audit(
  x: object, original: str, preferred: str, min_fuzz: float = 20
) -> bool:
  # ? Decides Whether To Remove A Suspected Fullmap Error Based On Fuzzy Matching
  o: str = x[original]  # pyright: ignore
  p: str = x[preferred]  # pyright: ignore
  return bool(ge(fuzz.ratio(o, p), min_fuzz) or ge(fuzz.partial_token_sort_ratio(o, p), min_fuzz))

@DISKCACHE.memoize()  # pyright: ignore
def BERT_audit(x: object, original: str, preferred: str, min_cos: float = 0.2) -> bool:
  # ? Decides Whether To Remove A Suspected Fullmap Error Based On BERT EMBEDDINGS
  o: str = x[original]  # pyright: ignore
  p: str = x[preferred]  # pyright: ignore

  embeddings: object = get_biobert().encode([o, p])  # pyright: ignore
  similarity: float = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # pyright: ignore
  return bool(ge(similarity, min_cos))

def fullmap_audit(lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed") -> pl.LazyFrame:
  # ? Ensures Fullmap Correct Processes Strings To CURIES
  # * Deletes Suspected Errors
  # ! Collection Points: map_elements With Custom Functions Require Eager
  original: str = add("original ", col)
  preferred: str = add(col, " name")
  cols: list[str] = [col, original, preferred]

  # * Stage 1: Exact String Matching Or Is Curie (Can Stay Lazy Until Filter)
  df: pl.DataFrame = lf.collect()
  pairs: pl.DataFrame = df.select(cols).unique()
  pairs = pairs.with_columns(eq(pl.col(cols[1]), pl.col(cols[2])).alias(out))

  passed: pl.DataFrame = pairs.filter(pl.col(out))
  pending: pl.DataFrame = pairs.filter(~pl.col(out))

  exempt_curies: str = r"^CHEBI|^PR|^UniProtKB"
  is_exempt: pl.DataFrame = pending.with_columns(pl.col(cols[0]).str.contains(exempt_curies).alias(out))
  pairs = pl.concat((passed, is_exempt))

  passed = pairs.filter(pl.col(out))
  pending = pairs.filter(~pl.col(out))

  is_curie: pl.DataFrame = pending.with_columns(pl.col(cols[1]).str.contains(":").alias(out))
  pairs = pl.concat((passed, is_curie))

  passed = pairs.filter(pl.col(out))
  pending = pairs.filter(~pl.col(out))

  exceptions: str = r"^LOC|^si:"
  is_exception: pl.DataFrame = pending.with_columns(pl.col(cols[2]).str.contains(exceptions).alias(out))
  pairs = pl.concat((passed, is_exception))

  passed = pairs.filter(pl.col(out))
  pending = pairs.filter(~pl.col(out))

  # * Stage 2: Fuzzy Matching Via RapidFuzz (Requires Eager)
  masked_fuzz: pl.DataFrame = pending.with_columns(pl.struct(cols[1:]).map_elements(lambda x: fuzz_audit(x, original, preferred), return_dtype=pl.Boolean).alias(out))
  pairs = pl.concat((passed, masked_fuzz))

  passed = pairs.filter(pl.col(out))
  pending = pairs.filter(~pl.col(out))

  # * Stage 3: BioBERT Embeddings (Requires Eager)
  BERT_fuzz: pl.DataFrame = pending.with_columns(pl.struct(cols[1:]).map_elements(lambda x: BERT_audit(x, original, preferred), return_dtype=pl.Boolean).alias(out))
  pairs = pl.concat((passed, BERT_fuzz))

  passed = pairs.filter(pl.col(out))
  pending = pairs.filter(~pl.col(out))

  # * Add Logging For Failed CURIES
  if pending.height > 0:
    for c, o, p in zip(pending.get_column(col).to_list(), pending.get_column(original).to_list(), pending.get_column(preferred).to_list()):
      logger.info(f"FAILED QC | STORE: {section_hash} | CONFIG: {config_file} | COL: {col} | ORIGINAL: {o!r} | PREFERRED: {p!r} | CURIE: {c!r}")

  return df.join(passed.select(col), on=col, how="semi").lazy()
