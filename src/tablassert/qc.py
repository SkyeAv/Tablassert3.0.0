from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from tablassert._lazy import LazyModule

if TYPE_CHECKING:
    import polars as pl
    import sentence_transformers
else:
    sentence_transformers = LazyModule("sentence_transformers")
    pl = LazyModule("polars")

from tablassert.errors import QcRuntimeMissingError
from tablassert.log import cat
from tablassert.utils import BASE

logger = cat("QC")

MODEL: Path = BASE / "biobert"


def _cascade(passed: pl.DataFrame, scored: pl.DataFrame, out: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Concat already-passed rows with a newly-scored frame and split on ``out``.

    Args:
        passed: Rows accumulated as passing from prior scoring steps.
        scored: Newly scored rows carrying the boolean ``out`` column.
        out: Name of the boolean pass/fail column.

    Returns:
        Tuple of ``(passed, pending)`` split on the ``out`` column.
    """
    pairs: pl.DataFrame = pl.concat((passed, scored))
    return pairs.filter(pl.col(out)), pairs.filter(~pl.col(out))


@cache
def get_biobert() -> object:
    """Lazy-load and memoize the BioBERT sentence-transformer (``functools.cache``).

    Loads from the local cache at ``MODEL`` when present; otherwise downloads
    ``pritamdeka/BioBERT-mnli-snli-scitail-mednli-stsb`` and saves it for
    future runs.

    Returns:
        The memoized ``SentenceTransformer`` instance.

    Raises:
        QcRuntimeMissingError: If ``sentence_transformers`` is not installed.
    """
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
    return model


def fullmap_audit(
    lf: pl.LazyFrame,
    col: str,
    section_hash: str,
    config_file: str,
    out: str = "passed",
    log: bool = True,
    on_phase: Callable[[str], None] | None = None,
) -> pl.LazyFrame:
    """Audit that fullmap correctly processed source strings into CURIEs.

    Runs a three-stage cascade that progressively filters out correct
    resolutions and leaves suspected errors behind:

    1. Exact match between pre-resolution text and the resolved preferred
       name, or the source value already looks like a CURIE, or the resolved
       preferred name matches a small allow-list of exception prefixes.
    2. Fuzzy string similarity (RapidFuzz ratio / partial token sort).
    3. BioBERT embedding cosine similarity for the remaining rows.

    Rows that fail every stage are treated as QC rejects and dropped from the
    output LazyFrame; when ``log`` is set, each reject is logged with the
    CURIE, original text, preferred name, and similarity scores.

    Args:
        lf: Source LazyFrame. Must expose ``col``, ``{col}_pre_resolution`` and
            ``{col}_name`` columns.
        col: Node column being audited.
        section_hash: Short section hash (for log context).
        config_file: Originating config file (for log context).
        out: Name of the boolean pass/fail column produced internally.
        log: When ``True``, log rejected CURIEs at INFO level.
        on_phase: Optional callback fired with ``"qc:exact"``, ``"qc:fuzzy"``
            and ``"qc:bert"`` at the start of each cascade stage (``qc:bert``
            only fires when Stage 3 actually runs), used to drive fine-grained
            progress UX.

    Returns:
        LazyFrame containing only rows whose ``col`` value passed QC.

    Notes:
        Collection point: pending pairs are handled eagerly because each stage
        needs the full set of survivors to batch-similarity-score them.
    """
    # Stage 0: deletes suspected errors.
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    from sklearn.metrics.pairwise import cosine_similarity

    original: str = f"{col}_pre_resolution"
    preferred: str = f"{col}_name"
    cols: list[str] = [col, original, preferred]

    # Stage 1: exact string matching or is CURIE (can stay lazy until filter).
    if on_phase is not None:
        on_phase("qc:exact")
    # Collection point: pending pairs require eager.
    df: pl.DataFrame = lf.collect()
    pairs: pl.DataFrame = df.select(cols).unique()
    pairs = pairs.with_columns((pl.col(cols[1]) == pl.col(cols[2])).alias(out))

    passed: pl.DataFrame
    pending: pl.DataFrame
    passed, pending = _cascade(pairs.clear(), pairs, out)

    exempt_curies: str = r"^CHEBI|^PR|^UniProtKB|^NCBIGene|^UMLS|^UNII|^PUBCHEM|^MONDO"
    is_exempt: pl.DataFrame = pending.with_columns(pl.col(cols[0]).str.contains(exempt_curies).alias(out))
    passed, pending = _cascade(passed, is_exempt, out)

    is_curie: pl.DataFrame = pending.with_columns(pl.col(cols[1]).str.contains(":").alias(out))
    passed, pending = _cascade(passed, is_curie, out)

    exceptions: str = r"^LOC|^si:"
    is_exception: pl.DataFrame = pending.with_columns(pl.col(cols[2]).str.contains(exceptions).alias(out))
    passed, pending = _cascade(passed, is_exception, out)

    # Stage 2: fuzzy matching via RapidFuzz (batched).
    if on_phase is not None:
        on_phase("qc:fuzzy")
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

    # Quick exit if no rows need BioBERT QC.
    if pending.height == 0:
        return df.join(passed.select(col), on=col, how="semi").lazy()

    # Stage 3: BioBERT embeddings (batched).
    if on_phase is not None:
        on_phase("qc:bert")
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

    # Log rejected CURIEs.
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
