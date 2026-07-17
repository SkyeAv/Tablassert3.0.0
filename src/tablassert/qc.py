from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from operator import add, eq
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import lazy_loader as Lazy

if TYPE_CHECKING:
    import onnxruntime as ort
    import polars as pl
    import sentence_transformers
else:
    ort = Lazy.load("onnxruntime")
    sentence_transformers = Lazy.load("sentence_transformers")
    pl = Lazy.load("polars")

from tablassert.log import cat

logger = cat("QC")

MODEL: Path = Path("./.onnxassert/")
MODEL_BACKEND: Literal["onnx"] = "onnx"
CPU_PROVIDER: str = "CPUExecutionProvider"
CUDA_PROVIDER: str = "CUDAExecutionProvider"

# TODO: Explore Best Model For QC
BIOBERT: dict[str, object] = {}


def has_qc_runtime(name: str) -> bool:
    try:
        get_version(name)
        return True
    except PackageNotFoundError:
        return False


def get_qc_provider(provider: Optional[Literal["cpu", "cuda"]] = None) -> tuple[str, Optional[dict[str, object]]]:
    has_cpu: bool = has_qc_runtime("onnxruntime")
    has_cuda: bool = has_qc_runtime("onnxruntime-gpu")

    if provider == "cpu":
        if has_cpu or has_cuda:
            return CPU_PROVIDER, None
        raise RuntimeError("03 | QC requires optional runtime dependencies. Install tablassert[qc] or tablassert[qc-cuda].")

    if provider == "cuda":
        if not has_cuda:
            raise RuntimeError("04 | QC requested CUDA runtime but onnxruntime-gpu is not installed. Install tablassert[qc-cuda].")
        available: list[str] = ort.get_available_providers()  # pyright: ignore
        if CUDA_PROVIDER not in available:
            raise RuntimeError(
                "05 | QC requested CUDA runtime but CUDAExecutionProvider is unavailable. Verify the CUDA/cuDNN environment for tablassert[qc-cuda]."
            )
        return CUDA_PROVIDER, {"device_id": 0}

    if has_cuda:
        available = ort.get_available_providers()  # pyright: ignore
        if CUDA_PROVIDER not in available:
            raise RuntimeError(
                "06 | Detected onnxruntime-gpu but CUDAExecutionProvider is unavailable. Tablassert will not fall back to CPU from qc-cuda. Install tablassert[qc] or fix the CUDA/cuDNN environment."
            )
        return CUDA_PROVIDER, {"device_id": 0}

    if has_cpu:
        return CPU_PROVIDER, None

    raise RuntimeError("07 | QC requires optional runtime dependencies. Install tablassert[qc] or tablassert[qc-cuda].")


def get_biobert(provider: Optional[Literal["cpu", "cuda"]] = None) -> object:
    # ? Lazy-loads BioBERT once on first batch audit call, then caches globally
    provider_name: str
    provider_options: Optional[dict[str, object]]
    provider_name, provider_options = get_qc_provider(provider)
    cache_key: str = add(provider_name, str(provider_options))
    if cache_key in BIOBERT:
        return BIOBERT[cache_key]

    session_opts: object = ort.SessionOptions()
    session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # pyright: ignore
    model_kwargs: dict[str, object] = {"provider": provider_name, "session_options": session_opts}
    if provider_options:
        model_kwargs["provider_options"] = provider_options

    if MODEL.exists():
        model: object = sentence_transformers.SentenceTransformer(str(MODEL), backend=MODEL_BACKEND, model_kwargs=model_kwargs)  # pyright: ignore
    else:
        model = sentence_transformers.SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", backend=MODEL_BACKEND, model_kwargs=model_kwargs
        )  # pyright: ignore
        MODEL.mkdir(parents=True, exist_ok=True)
        model.save(MODEL)  # pyright: ignore

    BIOBERT[cache_key] = model
    return model


def fullmap_audit(
    lf: pl.LazyFrame,
    col: str,
    section_hash: str,
    config_file: str,
    out: str = "passed",
    log: bool = True,
    provider: Optional[Literal["cpu", "cuda"]] = None,
) -> pl.LazyFrame:
    # ? Ensures Fullmap Correct Processes Strings To CURIES
    # * Deletes Suspected Errors
    # ! Collection Point: Pending Pairs Require Eager
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    from sklearn.metrics.pairwise import cosine_similarity

    original: str = add(col, "_pre_resolution")
    preferred: str = add(col, "_name")
    cols: list[str] = [col, original, preferred]

    # * Stage 1: Exact String Matching Or Is Curie (Can Stay Lazy Until Filter)
    df: pl.DataFrame = lf.collect()
    pairs: pl.DataFrame = df.select(cols).unique()
    pairs = pairs.with_columns(eq(pl.col(cols[1]), pl.col(cols[2])).alias(out))

    passed: pl.DataFrame = pairs.filter(pl.col(out))
    pending: pl.DataFrame = pairs.filter(~pl.col(out))

    exempt_curies: str = r"^CHEBI|^PR|^UniProtKB|^NCBIGene|^UMLS|^UNII|^PUBCHEM|^MONDO"
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

    pending = pending.with_columns([pl.Series("fuzz_ratio", ratio_scores), pl.Series("fuzz_partial", partial_scores)])

    fuzz_mask: pl.Series = pl.Series(out, (ratio_scores >= 70) | (partial_scores >= 80), dtype=pl.Boolean)
    masked_fuzz: pl.DataFrame = pending.with_columns(fuzz_mask)
    pairs = pl.concat((passed, masked_fuzz), how="diagonal")

    passed = pairs.filter(pl.col(out))
    pending = pairs.filter(~pl.col(out))

    # * Quick Exit If No Rows Need BioBERT QC
    if pending.height == 0:
        return df.join(passed.select(col), on=col, how="semi").lazy()

    # * Stage 3: BioBERT Embeddings (Batched)
    originals = pending.get_column(cols[1]).to_list()
    preferreds = pending.get_column(cols[2]).to_list()

    embeddings: object = get_biobert(provider).encode(originals + preferreds)  # pyright: ignore
    n: int = len(originals)
    similarity: object = cosine_similarity(embeddings[:n], embeddings[n:]).diagonal()  # pyright: ignore

    pending = pending.with_columns(pl.Series("bert_similarity", similarity))  # pyright: ignore

    bert_mask: pl.Series = pl.Series(out, similarity >= 0.5, dtype=pl.Boolean)  # pyright: ignore
    BERT_fuzz: pl.DataFrame = pending.with_columns(bert_mask)
    pairs = pl.concat((passed, BERT_fuzz), how="diagonal")

    passed = pairs.filter(pl.col(out))
    pending = pairs.filter(~pl.col(out))

    # * Add Logging For Failed CURIES
    if log and pending.height > 0:
        has_bert: bool = "bert_similarity" in pending.columns
        curies: list[str] = pending.get_column(col).to_list()
        originals_list: list[str] = pending.get_column(original).to_list()
        preferreds_list: list[str] = pending.get_column(preferred).to_list()
        fuzz_partials: list[object] = pending.get_column("fuzz_partial").to_list()
        bert_sims: list[object] = pending.get_column("bert_similarity").to_list() if has_bert else []

        for i, c in enumerate(curies):
            msg: str = f"FAILED | HASH: {section_hash} | CONFIG: {config_file} | COL: {col} | ORIGINAL: {originals_list[i]!r} | PREFERRED: {preferreds_list[i]!r} | CURIE: {c!r} | FUZZ: {fuzz_partials[i]}"
            if has_bert:
                msg = f"{msg} | BERT: {bert_sims[i]}"
            logger.info(msg)

    return df.join(passed.select(col), on=col, how="semi").lazy()
