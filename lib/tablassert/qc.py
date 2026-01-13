from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tablassert.utils import DISKCACHE
from rapidfuzz import fuzz
import onnxruntime as ort
from operator import add
from operator import ge
from operator import eq
import polars as pl

SESSION_OPTS: object = ort.SessionOptions()
SESSION_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# TODO: Explore Best Model For QC
BIOBERT: object = SentenceTransformer(
  "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
  backend="onnx",
  model_kwargs={
      "provider": "CPUExecutionProvider",
      "session_options": SESSION_OPTS
    }
)

@DISKCACHE.memoize()
def fuzz_audit(x: object, min_fuzz: float = 30) -> bool:
  # ? Decides Whether To Remove A Suspected Fullmap Error Based On Fuzzy Matching
  o: str = x[0]
  p: str = x[1]

  return (ge(fuzz.ratio(o, p), min_fuzz) or ge(fuzz.partial_token_sort_ratio(o, p), min_fuzz))

@DISKCACHE.memoize()
def BERT_audit(x: object, min_cos: float = 0.3) -> bool:
  # ? Decides Whether To Remove A Suspected Fullmap Error Based On BERT EMBEDDINGS
  o: str = x[0]
  p: str = x[1]

  embeddings: object = BIOBERT.encode([o, p])
  similarity: float = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
  return ge(similarity, min_cos)

def fullmap_audit(df: pl.DataFrame, col: str, out: str = "passed") -> pl.DataFrame:
  # ? Ensures Fullmap Correct Processes Strings To CURIES
  # * Deletes Suspected Errors
  original: str = add("original_", col)
  preferred: str = add(col, "_name")
  cols: list[str] = [original, preferred]

  pairs: pl.DataFrame = df.select(cols).unique()
  pairs = pairs.with_columns(eq(pl.col(cols[0]), pl.col(cols[1])).alias(out))

  passed: pl.DataFrame = pairs.filter(pl.col(out))
  pending: pl.DataFrame = pairs.filter(~pl.col(out))

  masked_fuzz: pl.DataFrame = pending.with_columns(pl.struct(cols).map_elements(fuzz_audit, return_dtype=pl.Boolean).alias(out))
  pairs = pl.concat((passed, masked_fuzz))

  passed = pairs.filter(pl.col(out))
  pending = pairs.filter(~pl.col(out))

  BERT_fuzz: pl.DataFrame = pending.with_columns(pl.struct(cols).map_elements(BERT_audit, return_dtype=pl.Boolean).alias(out))
  pairs = pl.concat((passed, BERT_fuzz))

  return df.join(pairs, on=cols, how="left").filter(pl.col(out)).drop(out)
