from __future__ import annotations

from operator import add, eq
from pathlib import Path
from typing import TYPE_CHECKING

import lazy_loader as Lazy

if TYPE_CHECKING:
    import polars as pl
    import sentence_transformers
else:
    sentence_transformers = Lazy.load("sentence_transformers")
    pl = Lazy.load("polars")

from tablassert.errors import QcRuntimeMissingError
from tablassert.log import cat
from tablassert.utils import BASE

logger = cat("QC")

MODEL: Path = BASE / "biobert"

# TODO: Explore Best Model For QC
BIOBERT: dict[str, object] = {}


def get_biobert() -> object:
    # ? Lazy-loads BioBERT once on first batch audit call, then caches globally
    if "model" in BIOBERT:
        return BIOBERT["model"]
    try:
        if MODEL.exists():
            model: object = sentence_transformers.SentenceTransformer(str(MODEL))  # pyright: ignore
        else:
            model = sentence_transformers.SentenceTransformer(  # pyright: ignore
                "pritamdeka/BioBERT-mnli-snli-scitail-mednli-stsb"
            )
            MODEL.mkdir(parents=True, exist_ok=True)
            model.save(MODEL)  # pyright: ignore
    except ImportError as exc:
        raise QcRuntimeMissingError() from exc
    BIOBERT["model"] = model
    return model


def fullmap_audit(lf: pl.LazyFrame, col: str, section_hash: str, config_file: str, out: str = "passed", log: bool = True) -> pl.LazyFrame:
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

    embeddings: object = get_biobert().encode(originals + preferreds)  # pyright: ignore
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
            fields: dict[str, object] = {
                "curie": c,
                "original": originals_list[i],
                "preferred": preferreds_list[i],
                "col": col,
                "fuzz": fuzz_partials[i],
                "config": config_file,
                "hash": section_hash,
            }
            template: str = "QC rejected {curie!r}: original={original!r} preferred={preferred!r} col={col} fuzz={fuzz} config={config} hash={hash}"
            if has_bert:
                fields["bert"] = bert_sims[i]
                template += " bert={bert}"
            logger.info(template, **fields)  # pyright: ignore

    return df.join(passed.select(col), on=col, how="semi").lazy()
