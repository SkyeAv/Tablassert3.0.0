from __future__ import annotations

from operator import add, eq
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import lazy_loader as Lazy
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    import onnxruntime as ort
    import polars as pl
    import sentence_transformers
else:
    ort = Lazy.load("onnxruntime")
    sentence_transformers = Lazy.load("sentence_transformers")
    pl = Lazy.load("polars")

from tablassert.log import logger

SESSION_OPTS: object = ort.SessionOptions()
SESSION_OPTS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # pyright: ignore

MODEL: Path = Path("./.onnxassert/")
MODEL_BACKEND: Literal["onnx"] = "onnx"
MODEL_KWARGS: dict[str, object] = {"provider": "CPUExecutionProvider", "session_options": SESSION_OPTS}

# TODO: Explore Best Model For QC
BIOBERT: Optional[object] = None


def get_biobert() -> object:
    # ? Lazy-loads BioBERT once on first batch audit call, then caches globally
    global BIOBERT
    if BIOBERT:
        return BIOBERT
    elif not BIOBERT and MODEL.exists():
        BIOBERT = sentence_transformers.SentenceTransformer(
            str(MODEL), backend=MODEL_BACKEND, model_kwargs=MODEL_KWARGS
        )  # pyright: ignore
    else:
        BIOBERT = sentence_transformers.SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", backend=MODEL_BACKEND, model_kwargs=MODEL_KWARGS
        )  # pyright: ignore
        MODEL.mkdir(parents=True, exist_ok=True)
        BIOBERT.save(MODEL)  # pyright: ignore
    return BIOBERT


def fullmap_audit(lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed") -> pl.LazyFrame:
    # ? Ensures Fullmap Correct Processes Strings To CURIES
    # * Deletes Suspected Errors
    # ! Collection Point: Pending Pairs Require Eager
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

    # * Stage 2: Fuzzy Matching Via RapidFuzz (Batched)
    originals: list[str] = pending.get_column(cols[1]).to_list()
    preferreds: list[str] = pending.get_column(cols[2]).to_list()

    ratio_scores: object = cpdist(originals, preferreds, scorer=fuzz.ratio)
    partial_scores: object = cpdist(originals, preferreds, scorer=fuzz.partial_token_sort_ratio)

    fuzz_mask: pl.Series = pl.Series(out, (ratio_scores >= 20) | (partial_scores >= 20), dtype=pl.Boolean)
    masked_fuzz: pl.DataFrame = pending.with_columns(fuzz_mask)
    pairs = pl.concat((passed, masked_fuzz))

    passed = pairs.filter(pl.col(out))
    pending = pairs.filter(~pl.col(out))

    # * Stage 3: BioBERT Embeddings (Batched)
    originals = pending.get_column(cols[1]).to_list()
    preferreds = pending.get_column(cols[2]).to_list()

    embeddings: object = get_biobert().encode(originals + preferreds)  # pyright: ignore
    n: int = len(originals)
    similarity: object = cosine_similarity(embeddings[:n], embeddings[n:]).diagonal()  # pyright: ignore

    bert_mask: pl.Series = pl.Series(out, similarity >= 0.2, dtype=pl.Boolean)  # pyright: ignore
    BERT_fuzz: pl.DataFrame = pending.with_columns(bert_mask)
    pairs = pl.concat((passed, BERT_fuzz))

    passed = pairs.filter(pl.col(out))
    pending = pairs.filter(~pl.col(out))

    # * Add Logging For Failed CURIES
    if pending.height > 0:
        for c, o, p in zip(
            pending.get_column(col).to_list(),
            pending.get_column(original).to_list(),
            pending.get_column(preferred).to_list(),
        ):
            logger.info(
                f"FAILED QC | STORE: {section_hash} | CONFIG: {config_file} | COL: {col} | ORIGINAL: {o!r} | PREFERRED: {p!r} | CURIE: {c!r}"
            )

    return df.join(passed.select(col), on=col, how="semi").lazy()
