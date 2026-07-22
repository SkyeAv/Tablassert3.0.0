from __future__ import annotations

import math
import operator
import re
from collections.abc import Iterable
from functools import cache
from operator import add, eq, le
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Self, Union

import lazy_loader as Lazy
from pydantic import Field, NonNegativeInt

from tablassert import rs
from tablassert.enums import ALLOWED_EDGE_FIELDS, Categories, EdgeCategories, EncodingMethods, Files, InformationResources, Repositories, Tokens
from tablassert.fullmap import ResolveSpec, fullmap_db_path, resolve, resolve_batch
from tablassert.log import cat
from tablassert.models import DEFAULT_RIG_UI_EXPLANATION, Encoding, NodeEncoding, Section, default_rig_contributions
from tablassert.nlp import level_one, level_two
from tablassert.qc import fullmap_audit

if TYPE_CHECKING:
    import numpy as np
    import polars as pl
else:
    np = Lazy.load("numpy")
    pl = Lazy.load("polars")

logger = cat("PIPELINE")

TERMS_OF_USE_WARNING: str = (
    "Terms of use and license information for the upstream sources used to create this KGX were not declared in the "
    "Tablassert graph configuration. Translator source-ingest guidance expects source owners to assess terms of use "
    "before publication or downstream ingest; review the upstream source terms and replace this generated warning with "
    "explicit license or terms information when known."
)

CATEGORY_PARENT: dict[str, str] = {
    # Biolink is_a chain -- leaf to parent role for association name matching.
    "SmallMolecule": "MolecularEntity",
    "MolecularEntity": "ChemicalEntity",
    "Drug": "MolecularMixture",
    "MolecularMixture": "ChemicalMixture",
    "ChemicalMixture": "ChemicalEntity",
    "Protein": "Gene",
    "SequenceVariant": "Variant",
    "Haplotype": "Genotype",
}


def parse_edge_name(name: str) -> Optional[tuple[str, list[str]]]:
    """Parse ``{Subject}To{Object}Association`` into ``(subject, [object roles])``.

    Args:
        name: Edge category name (e.g. ``ChemicalEntityToGeneOrProteinAssociation``).

    Returns:
        Tuple of ``(subject_role, object_roles)`` with the trailing
        ``Association`` suffix removed and object side split on ``Or``, or
        ``None`` if ``name`` does not contain ``"To"``.
    """
    name = name.removesuffix("Association")
    if "To" not in name:
        return None
    subj: str = ""
    rest: str = ""
    subj, rest = name.split("To", 1)
    # Split on Camel-case "Or" tokens only, so embedded "Or" substrings such as
    # the one in "Organism" are not treated as object-role separators.
    return (subj, re.split(r"(?<=[a-z])Or(?=[A-Z])", rest))


@cache
def edge_tables() -> tuple[dict[str, str], dict[str, str]]:
    """Generate and cache ``CATEGORY_ROLE`` and ``EDGE_LOOKUP`` on first call.

    Returns:
        Tuple of ``(CATEGORY_ROLE, EDGE_LOOKUP)`` where ``CATEGORY_ROLE`` maps
        each ``CATEGORY_PARENT`` leaf to its root role (via the parent chain),
        and ``EDGE_LOOKUP`` maps ``"subj_role|obj_role"`` to the biolink CURIE
        of the corresponding ``EdgeCategories`` value (falling back to explicit
        overrides and ultimately to ``biolink:Association``).
    """
    # Flattened leaf -> root role (walks CATEGORY_PARENT chain to root).
    CATEGORY_ROLE: dict[str, str] = {}
    for leaf in CATEGORY_PARENT:
        role: str = leaf
        while role in CATEGORY_PARENT:
            role = CATEGORY_PARENT[role]
        CATEGORY_ROLE[leaf] = role

    # Auto-generated (subject role, object role) -> EdgeCategories.
    EDGE_MAP: dict[tuple[str, str], EdgeCategories] = {}
    for ec in EdgeCategories:
        if ec is not EdgeCategories.ASSOCIATION:
            parsed: Optional[tuple[str, list[str]]] = parse_edge_name(ec.value)
            if parsed is not None:
                subj: str = ""
                objs: list[str] = []
                subj, objs = parsed
                for obj in objs:
                    EDGE_MAP[(subj, obj)] = ec

    # Non-standard names -- explicit overrides.
    EDGE_MAP[("ChemicalEntity", "Gene")] = EdgeCategories.CHEMICAL_GENE_INTERACTION

    # Flattened "subj_role|obj_role" -> biolink CURIE (for polars replace_strict).
    EDGE_LOOKUP: dict[str, str] = {f"{s}|{o}": add("biolink:", ec.value) for (s, o), ec in EDGE_MAP.items()}

    return CATEGORY_ROLE, EDGE_LOOKUP


def edge_category(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Add the derived ``category`` column using native polars replace operations.

    Args:
        lf: Source LazyFrame with ``subject category`` and ``object category``
            columns (biolink-prefixed).

    Returns:
        LazyFrame with a new list-typed ``category`` column containing the
        resolved biolink edge-category CURIE (falling back to
        ``biolink:Association`` when no specific mapping matches).
    """
    cat_role: dict[str, str]
    edge_lookup: dict[str, str]
    cat_role, edge_lookup = edge_tables()
    names: list[str] = lf.collect_schema().names()
    subject_col: str = "subject_category" if "subject_category" in names else "subject category"
    object_col: str = "object_category" if "object_category" in names else "object category"
    sr: pl.Expr = pl.col(subject_col).str.replace("biolink:", "").replace(cat_role).fill_null("")
    or_: pl.Expr = pl.col(object_col).str.replace("biolink:", "").replace(cat_role).fill_null("")
    return lf.with_columns(
        pl.concat_list(
            pl.concat_str([sr, pl.lit("|"), or_]).replace_strict(edge_lookup, default=add("biolink:", EdgeCategories.ASSOCIATION.value))
        ).alias("category")
    )


def value(lf: pl.LazyFrame, col: str, x: str) -> pl.LazyFrame:
    """Add a new column populated with a literal string value.

    Args:
        lf: Source LazyFrame.
        col: Name of the new column.
        x: Literal string value to populate every row with.

    Returns:
        LazyFrame with the new literal column appended.
    """
    return lf.with_columns(pl.lit(x).alias(col))


def source_record_urls(lf: pl.LazyFrame, url: str) -> pl.LazyFrame:
    """Add Biolink/Translator ``source_record_urls`` as a single-element list column.

    Args:
        lf: Source LazyFrame.
        url: Source record URL to record for every row.

    Returns:
        LazyFrame with the new list column appended.
    """
    return lf.with_columns(pl.concat_list(pl.lit(url)).alias("source_record_urls"))


def publications(lf: pl.LazyFrame, curie: str) -> pl.LazyFrame:
    """Add a publication CURIE as a Biolink-compliant ``list[str]`` column.

    Args:
        lf: Source LazyFrame.
        curie: Publication CURIE (e.g. ``PMCID:PMC1234567``).

    Returns:
        LazyFrame with the new ``publications`` list column appended.
    """
    return lf.with_columns(pl.concat_list(pl.lit(curie)).alias("publications"))


def column(lf: pl.LazyFrame, col: str, x: str) -> pl.LazyFrame:
    """Add a new column copied from an existing column.

    Args:
        lf: Source LazyFrame.
        col: Name of the new column.
        x: Name of the source column to copy.

    Returns:
        LazyFrame with the new column appended.
    """
    return lf.with_columns(pl.col(x).alias(col))


def math_op(lf: pl.LazyFrame, col: str, func: str, args: list[Union[Literal[Tokens.VALUES], float, int]]) -> pl.LazyFrame:
    """Transform values in a column using a ``math`` module function.

    Args:
        lf: Source LazyFrame.
        col: Column to transform in place.
        func: Name of a callable on the stdlib ``math`` module.
        args: Argument list, where ``"values"`` is replaced by the current
            cell value.

    Returns:
        New LazyFrame (collected eagerly, then re-lazied) with the transformed
        column.

    Notes:
        Collection point: required for ``map_elements``. ``strict=False``
        tolerates residual non-numeric junk in numeric annotation columns.
    """
    # Collection point: required for map_elements.
    # strict=False tolerates residual non-numeric junk in numeric annotation columns.
    df: pl.DataFrame = lf.collect()
    expr: pl.Expr = pl.col(col).cast(pl.Float64, strict=False)
    attr: Callable[[Any], Any] = getattr(math, func)
    df = df.with_columns(expr.map_elements(lambda x: attr(*(x if eq(a, Tokens.VALUES) else a for a in args)), return_dtype=pl.Float64).alias(col))
    return df.lazy()


def numeric_columns(names: list[str]) -> list[str]:
    """Return column names that should be coerced and formatted as numbers.

    P-value columns by substring plus the exact ``relationship_strength`` and
    study-size fields.

    Args:
        names: Schema column names to filter.

    Returns:
        Subset of ``names`` destined for numeric coercion/formatting.
    """
    # P-value columns by substring plus exact strength and study-size fields.
    exact: set[str] = {"relationship_strength", "sample_size", "supporting_study_size"}
    return [c for c in names if ("p_value" in c.lower()) or (c in exact)]


def clean_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Coerce numeric annotation columns to Float64, dropping non-numeric values to null.

    Only touches p-value, relationship-strength and sample-size columns.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the matched columns cast to Float64 (no-op if none match).
    """
    # Only touches p-value, relationship-strength and sample-size columns.
    cols: list[str] = numeric_columns(lf.collect_schema().names())
    if not cols:
        return lf
    return lf.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols])


def format_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Format numeric annotation columns as strings with controlled notation.

    P-value columns use scientific notation (``{:.4e}``); all others use
    decimal general format (``{:.4g}``). Null values stay null.

    Args:
        lf: Source LazyFrame.

    Returns:
        New LazyFrame (eagerly collected then re-lazied) with matched columns
        formatted as strings.

    Notes:
        Collection point: numpy batch formatting is required for notation
        control across the full column at once.
    """
    # Collection point: numpy batch formatting required for notation control.
    # P-value columns use scientific notation; others use decimal general format.
    df: pl.DataFrame = lf.collect()
    cols: list[str] = numeric_columns(df.columns)
    for c in cols:
        df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        mask: object = df[c].is_null().to_numpy()
        arr: object = df[c].to_numpy()
        fmt: str = "{:.4e}" if "p_value" in c.lower() else "{:.4g}"
        formatted: list[Optional[str]] = [None if m else fmt.format(float(v)) for v, m in zip(arr, mask)]  # pyright: ignore
        df = df.with_columns(pl.Series(c, formatted))
    return df.lazy()


def prefix(lf: pl.LazyFrame, col: str, prefix: str) -> pl.LazyFrame:
    expr: pl.Expr = add(pl.lit(prefix), pl.col(col).cast(pl.String))
    return lf.with_columns(expr.alias(col))


def suffix(lf: pl.LazyFrame, col: str, suffix: str) -> pl.LazyFrame:
    expr: pl.Expr = add(pl.col(col).cast(pl.String), pl.lit(suffix))
    return lf.with_columns(expr.alias(col))


def regex(lf: pl.LazyFrame, col: str, pattern: str, replacement: str = "") -> pl.LazyFrame:
    expr: pl.Expr = pl.col(col).cast(pl.String).str.replace_all(pattern, replacement)
    return lf.with_columns(expr.alias(col))


def fill(lf: pl.LazyFrame, col: str, method: str) -> pl.LazyFrame:
    expr: pl.Expr = pl.col(col).fill_null(strategy=method)  # pyright: ignore
    return lf.with_columns(expr.alias(col))


def explode(lf: pl.LazyFrame, col: str, delimiter: str) -> pl.LazyFrame:
    """Explode one row with delimited items into many rows.

    Args:
        lf: Source LazyFrame.
        col: Column whose string values should be split.
        delimiter: Delimiter to split on.

    Returns:
        LazyFrame with one row per item (the split column becomes a list
        before the explode).
    """
    expr: pl.Expr = pl.col(col).cast(pl.String).str.split(delimiter)
    lf = lf.with_columns(expr.alias(col))
    return lf.explode(col)


def sig(lf: pl.LazyFrame, col: str = "p_value", out: str = "statistical_significance_qualifier") -> pl.LazyFrame:
    """Create the ``statistical_significance_qualifier`` column (Biolink PR #1766).

    Picks the closest fuzzy-matching p-value-like column when the exact name
    is missing, then buckets the value into one of five significance bands.

    Args:
        lf: Source LazyFrame.
        col: Reference column name to fuzzy-match against.
        out: Output qualifier column name.

    Returns:
        LazyFrame with the new qualifier column. No-op when no p-value-like
        column is present.

    Notes:
        Five-band cascade matches ``StatisticalSignificanceQualifierEnum``
        verbatim. Boundaries are hardcoded because the enum definitions are
        canonical.

    Warnings:
        Edges are never dropped here. No p-value column → qualifier absent;
        null p-value → null qualifier. The qualifier may only be set when
        ``p_value``/``adjusted_p_value`` is populated (Biolink class rule).
    """
    from rapidfuzz import fuzz

    names: list[str] = lf.collect_schema().names()
    candidates: list[str] = [c for c in names if col in c]
    chosen: Optional[str] = max(candidates, key=lambda c: fuzz.ratio(c, col)) if candidates else None
    if chosen is None:
        # Biolink class rule: qualifier may only be set when p_value/adjusted_p_value is populated.
        return lf
    expr: pl.Expr = pl.col(chosen).cast(pl.Float64, strict=False)
    band: pl.Expr = (
        pl.when(expr.is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(le(expr, 0.001))
        .then(pl.lit("biolink:very_strongly_significant"))
        .when(le(expr, 0.01))
        .then(pl.lit("biolink:strongly_significant"))
        .when(le(expr, 0.05))
        .then(pl.lit("biolink:significant"))
        .when(le(expr, 0.10))
        .then(pl.lit("biolink:suggestive"))
        .otherwise(pl.lit("biolink:not_significant"))
    )
    return lf.with_columns(band.alias(out))


PVALUE_TOKEN_PATTERN: re.Pattern = re.compile(r"(?i)\bp[\s_\-.]*val(?:ue)?s?\b")
QVALUE_TOKEN_PATTERN: re.Pattern = re.compile(r"(?i)\bq[\s_\-.]*val(?:ue)?s?\b")
PADJ_TOKEN_PATTERN: re.Pattern = re.compile(r"(?i)\bp[\s_\-.]*adj(?:usted)?\b")
BARE_P_TOKEN_PATTERN: re.Pattern = re.compile(r"(?i)\bp\b")
STANDALONE_ADJUSTED_PATTERN: re.Pattern = re.compile(r"(?i)\b(?:fdr|bonferroni|holm|false discovery rate)\b")
CONTEXTUAL_ADJUSTED_PATTERN: re.Pattern = re.compile(r"(?i)\b(?:adj(?:usted)?|corrected)\b")
SIGNIFICANCE_FLAG_PATTERN: re.Pattern = re.compile(r"(?i)significan")


def pvalue_target(name: str) -> Optional[str]:
    """Map a column name to its canonical Biolink-compliant target name.

    Args:
        name: Raw source column name.

    Returns:
        ``"p_value"``, ``"adjusted_p_value"``, or ``None`` if the name does
        not look like a p/q-value column.

    Notes:
        Word-boundary anchored so "Group value" style substrings are not
        falsely matched. Bare ``"P"`` and ``"padj"``/``"p.adj"`` cover common
        GWAS/DESeq2 conventions. ``"adj"``/``"adjusted"``/``"corrected"``
        only count alongside a p/q-value token, since they are generic words
        also used for adjusted hazard/odds ratios (unlike
        ``fdr``/``bonferroni``/``holm``). ``"significance"``/``"significant"``
        columns are categorical flags, not the numeric value, so they are
        excluded unless a p/q-value token is also present.
    """
    core_pvalue: bool = bool(PVALUE_TOKEN_PATTERN.search(name)) or bool(BARE_P_TOKEN_PATTERN.search(name))
    core_qvalue: bool = bool(QVALUE_TOKEN_PATTERN.search(name))
    core_padj: bool = bool(PADJ_TOKEN_PATTERN.search(name))
    has_core: bool = core_pvalue or core_qvalue or core_padj

    if SIGNIFICANCE_FLAG_PATTERN.search(name) and not has_core:
        return None

    is_adjusted: bool = (
        core_padj or core_qvalue or bool(STANDALONE_ADJUSTED_PATTERN.search(name)) or (bool(CONTEXTUAL_ADJUSTED_PATTERN.search(name)) and has_core)
    )

    if not (has_core or is_adjusted):
        return None
    return "adjusted_p_value" if is_adjusted else "p_value"


def coerce_pvalue_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename p-value-like columns to Biolink KGX-compliant ``p_value`` / ``adjusted_p_value``.

    Picks a single best fuzzy match per target when multiple candidates exist.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the chosen columns renamed (no-op if no candidates).
    """
    # Picks a single best fuzzy match per target when multiple candidates exist.
    from rapidfuzz import fuzz

    names: list[str] = lf.collect_schema().names()
    buckets: dict[str, list[str]] = {}
    for n in names:
        target: Optional[str] = pvalue_target(n)
        if target:
            buckets.setdefault(target, []).append(n)

    renames: dict[str, str] = {}
    for target, candidates in buckets.items():
        reference: str = target.replace("_", " ")
        chosen: str = max(candidates, key=lambda c: fuzz.ratio(c, reference))
        if chosen != target:
            renames[chosen] = target

    return lf.rename(renames) if renames else lf


STUDY_SIZE_EXACT_PATTERN: re.Pattern = re.compile(
    r"(?i)^(?:n|total[\s_\-.]*n|sample[\s_\-.]*size|samplesize|study[\s_\-.]*size|cohort[\s_\-.]*size|supporting[\s_\-.]*study[\s_\-.]*size)$"
)
STUDY_SIZE_COUNT_PATTERN: re.Pattern = re.compile(
    r"(?i)\b(?:samples?|participants?|subjects?|individuals?|patients?|cases?|enrollment)[\s_\-.]*(?:n|count|number|size)\b"
)
STUDY_SIZE_PREFIX_PATTERN: re.Pattern = re.compile(
    r"(?i)\b(?:n|num|number|count|total)[\s_\-.]*(?:of[\s_\-.]*)?(?:samples?|participants?|subjects?|individuals?|patients?|cases?)\b"
)
STUDY_SIZE_SUFFIX_PATTERN: re.Pattern = re.compile(r"(?i)\b(?:samples?|participants?|subjects?|individuals?|patients?|cases?)[\s_\-.]*n\b")
STUDY_SIZE_SINGLETON_PATTERN: re.Pattern = re.compile(r"(?i)^(?:participants|enrollment)$")


def study_size_target(name: str) -> Optional[str]:
    """Map study-size-like column names to the canonical ``supporting_study_size`` slot.

    Bare ``"n"`` is allowed, but other matches need explicit sample/study-size
    context.

    Args:
        name: Raw source column name.

    Returns:
        ``"supporting_study_size"`` when the name matches any of the
        study-size patterns, else ``None``.
    """
    if STUDY_SIZE_EXACT_PATTERN.search(name):
        return "supporting_study_size"
    if STUDY_SIZE_COUNT_PATTERN.search(name):
        return "supporting_study_size"
    if STUDY_SIZE_PREFIX_PATTERN.search(name):
        return "supporting_study_size"
    if STUDY_SIZE_SUFFIX_PATTERN.search(name):
        return "supporting_study_size"
    if STUDY_SIZE_SINGLETON_PATTERN.search(name):
        return "supporting_study_size"
    return None


def coerce_study_size_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename study-size-like columns to Biolink KGX-compliant ``supporting_study_size``.

    Picks a single best fuzzy match and leaves other candidate columns
    untouched.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the chosen column renamed (no-op if no match).
    """
    # Picks a single best fuzzy match and leaves other candidate columns untouched.
    from rapidfuzz import fuzz

    names: list[str] = lf.collect_schema().names()
    candidates: list[str] = [n for n in names if study_size_target(n)]
    if not candidates:
        return lf

    target: str = "supporting_study_size"
    reference: str = target.replace("_", " ")
    chosen: str = max(candidates, key=lambda c: fuzz.ratio(c, reference))
    if chosen == target:
        return lf
    return lf.rename({chosen: target})


def idx(lf: pl.LazyFrame, col: str = "extracted_from_row_number") -> pl.LazyFrame:
    """Create a 1-based index column recording the original source row.

    Args:
        lf: Source LazyFrame.
        col: Name of the new index column.

    Returns:
        LazyFrame with the index column appended.

    Notes:
        Matches pre-8.0.0 behavior; folded into ``supporting_text`` by
        ``compile_graph``.
    """
    return lf.with_row_index(col, offset=1)


def csv(p: Path, sep: str) -> pl.LazyFrame:
    """Read a CSV or TSV source as a LazyFrame.

    Args:
        p: Path to the delimited text file.
        sep: Column delimiter character.

    Returns:
        LazyFrame over the file (no header inference; all columns raw).
    """
    return pl.scan_csv(source=p, separator=sep, has_header=False, infer_schema_length=None, truncate_ragged_lines=True)


def excel(p: Path, sheet: str, engine: str = "calamine") -> pl.LazyFrame:
    """Read an Excel sheet as a LazyFrame.

    Args:
        p: Path to the workbook.
        sheet: Sheet name to read.
        engine: Polars Excel engine name.

    Returns:
        LazyFrame over the sheet contents (no header inference).
    """
    df: pl.DataFrame = pl.read_excel(
        source=p,
        sheet_name=sheet,
        engine=engine,  # pyright: ignore
        has_header=False,
        infer_schema_length=None,
    )
    return df.lazy()


def crop(lf: pl.LazyFrame, row_slice: list[Union[NonNegativeInt, Literal[Tokens.AUTO]]]) -> pl.LazyFrame:
    """Take a contiguous slice from a LazyFrame.

    Args:
        lf: Source LazyFrame.
        row_slice: Two-element ``[start, stop]`` list, where either bound may
            be the literal ``Tokens.AUTO`` sentinel.

    Returns:
        LazyFrame restricted to the requested row range.

    Notes:
        Collection point: height must be computed eagerly to resolve
        ``Tokens.AUTO`` bounds.
    """
    df: pl.DataFrame = lf.collect()
    n: int = df.select(pl.len()).item()
    start: Union[int, Literal[Tokens.AUTO]] = row_slice[0]
    stop: Union[int, Literal[Tokens.AUTO]] = row_slice[1]
    offset: int = 0 if eq(start, Tokens.AUTO) else start  # pyright: ignore
    length: int = n if eq(stop, Tokens.AUTO) else (stop - offset)  # pyright: ignore
    df = df.slice(offset=offset, length=length)
    return df.lazy()


def pick(lf: pl.LazyFrame, rows: list[int]) -> pl.LazyFrame:
    """Pick an explicit list of rows from a LazyFrame.

    Args:
        lf: Source LazyFrame.
        rows: Row indices to retain.

    Returns:
        LazyFrame containing only the requested rows, in the given order.

    Notes:
        Collection point: ``take()`` requires an eager frame, so the result
        is re-lazied afterwards.
    """
    df: pl.DataFrame = lf.collect()
    df = df.select(pl.all().take(indices=rows))  # pyright: ignore
    return df.lazy()


def reindex(df: pl.LazyFrame, col: str, op: Callable, comp: Union[str, int, float], cast: bool = True) -> pl.LazyFrame:
    """Reindex a LazyFrame by filtering rows on a column condition.

    Args:
        df: Source LazyFrame.
        col: Column to compare.
        op: Binary comparison callable (e.g. ``operator.lt``).
        comp: Right-hand comparison value.
        cast: When True, cast ``col`` to Float64 before comparison.

    Returns:
        LazyFrame containing only rows where ``op(col, comp)`` is true.
    """
    expr: pl.Expr = pl.col(col).cast(pl.Float64) if cast else pl.col(col)
    return df.filter(op(expr, comp))


def drop_not_significant(lf: pl.LazyFrame, col: str = "statistical_significance_qualifier") -> pl.LazyFrame:
    """Drop release-mode edges whose significance qualifier is ``biolink:not_significant``.

    Args:
        lf: Source LazyFrame.
        col: Significance qualifier column name.

    Returns:
        LazyFrame with non-significant edges removed.

    Notes:
        Only filters when the qualifier column exists; no-op for sections
        without a ``p_value`` column. ``ne_missing`` keeps null qualifiers
        (null p-value → null qualifier → kept, not dropped).
    """
    names: list[str] = lf.collect_schema().names()
    if col not in names:
        return lf
    return lf.filter(pl.col(col).cast(pl.String).ne_missing("biolink:not_significant"))


def idxname(col: Any) -> str:
    """Convert Excel-style column letters (e.g. ``"AA"``) to a polars-style ``column_<n>`` name.

    Args:
        col: Excel column letter (or any value whose ``str`` form is one).

    Returns:
        ``f"column_{idx}"`` where ``idx`` is the 1-based column number.
    """
    scol: str = str(col)
    idx: int = 0
    for char in scol:
        idx = idx * 26 + (ord(char) - 65 + 1)

    return f"column_{idx}"


def trim(lf: pl.LazyFrame, regex: str = r"^column_\d+$") -> pl.LazyFrame:
    """Remove columns whose names match the Excel-style ``column_<n>`` convention.

    Args:
        lf: Source LazyFrame.
        regex: Column-name pattern to drop.

    Returns:
        LazyFrame with matching columns excluded.
    """
    return lf.select(pl.exclude(regex))


def to_store(lf: pl.LazyFrame, p: Path, config_name: str) -> Path:
    """Collect and write a section LazyFrame to a parquet file.

    Args:
        lf: Section LazyFrame to materialize.
        p: Destination parquet path.
        config_name: Human-readable section name used in the empty-row warning.

    Returns:
        The parquet path written.

    Notes:
        Warns when the collected frame has zero rows.
    """
    df: pl.DataFrame = lf.collect()

    if df.height == 0:
        logger.warning("Section {config} ({hash}) produced 0 rows", config=config_name, hash=p.stem)
    df.write_parquet(p)

    return p


class Tcode(Section):
    """Extend ``Section`` with the per-section operation list that compiles a knowledge graph."""

    config: Path = Field(...)
    store: Path = Field(...)
    log: bool = Field(False)
    qc: bool = Field(False)
    release: bool = Field(False)
    name: Optional[str] = Field(None)

    def encoding(self: Self, x: Encoding, col: str, table_literal: bool = False) -> list[Any]:
        """Collect helper for Encoding classes.

        Builds the ordered list of (callable, args) tuples that implement an
        ``Encoding`` against a single column.

        Args:
            x: Encoding configuration.
            col: Target column name.
            table_literal: When True, also emit a tuple preserving the raw
                column under an ``original_`` prefix.

        Returns:
            Flat list of ``Any`` items (tuples or lists of tuples, possibly
            containing ``None`` placeholders) ready for ``clean`` to filter.
        """
        return [
            (value, (col, x.encoding)) if eq(x.method, EncodingMethods.VALUE) else None,
            (column, (col, idxname(x.encoding))) if eq(x.method, EncodingMethods.COLUMN) else None,
            (column, (add("original_", col), col)) if table_literal else None,
            (fill, (col, x.fill)) if x.fill else None,
            (explode, (col, x.explode_by)) if x.explode_by else None,
            [(regex, (col, r.pattern, r.replacement)) for r in x.regex] if x.regex else None,
            [(regex, (col, r)) for r in x.remove] if x.remove else None,
            (prefix, (col, x.prefix)) if x.prefix else None,
            (suffix, (col, x.suffix)) if x.suffix else None,
            [(math_op, (col, t.function, t.arguments)) for t in x.transformations] if x.transformations else None,
        ]

    def node_prep(self: Self, x: NodeEncoding, col: str) -> list[Any]:
        """Collect helper for NodeEncoding classes.

        Encoding plus NLP normalization, before ``resolve_batch``.

        Args:
            x: NodeEncoding configuration.
            col: Target column name.

        Returns:
            Combined list of encoding tuples plus the pre-resolution column
            copy and ``level_one``/``level_two`` normalization tuples.
        """
        encoding: list[Any] = self.encoding(x, col, table_literal=True)
        prep: list[Any] = [(column, (add(col, "_pre_resolution"), col)), (level_one, (col,)), (level_two, (col,))]
        return add(encoding, prep)

    def clean(self: Self, tcode: list[tuple[Callable, Any]]) -> list[tuple[Callable, tuple[Any]]]:
        """Clean a Tcode list so it can be used with ``reduce`` from functools.

        Args:
            tcode: Raw list possibly containing ``None`` placeholders and
                nested lists of tuples.

        Returns:
            Flat list of ``(callable, tuple[Any, ...])`` pairs with falsy
            entries removed.
        """
        result: list[tuple[Callable, tuple[Any]]] = []
        for x in tcode:
            if not x:
                continue
            elif isinstance(x, list):
                result.extend(self.clean(x))
            else:
                result.append(x)
        return result

    def collect(self: Self, db: Path) -> Union[list[tuple[Callable, tuple[Any]]], Path]:
        """Build the ordered operation list that drives section transformation.

        Args:
            db: Path to the fullmap redb used for entity resolution.

        Returns:
            Either the existing store path (quick exit when the subgraph
            parquet is already present), or a cleaned list of
            ``(callable, args)`` tuples consumed by ``compile_subgraph``.
        """
        if self.store.is_file():
            # Quick exit if subgraph already exists.
            return self.store

        else:
            # Subject/object/qualifiers share one resolve_batch call instead of one per column.
            node_columns: list[tuple[NodeEncoding, str]] = [
                (self.statement.subject, "subject"),
                (self.statement.object, "object"),
                *[(x, x.qualifier) for x in (self.statement.qualifiers or [])],
            ]
            specs: list[ResolveSpec] = [ResolveSpec(col, str(x.taxon) if x.taxon else None, x.prioritize, x.avoid) for x, col in node_columns]

            # Returns a list of: (function, (arguments)).
            tcode: Optional[list[Any]] = [
                (csv, (self.source.local, self.source.delimiter)) if eq(self.source.kind, Files.TEXT) else None,  # pyright: ignore
                (excel, (self.source.local, self.source.sheet)) if eq(self.source.kind, Files.EXCEL) else None,  # pyright: ignore
                (idx, ()),
                (crop, (self.source.row_slice,)) if self.source.row_slice else None,
                (pick, (self.source.rows,)) if self.source.rows else None,
                [
                    (reindex, (idxname(x.column), getattr(operator, x.comparison), x.comparator))
                    if x.comparison not in ["ne", "eq"]
                    else (reindex, (idxname(x.column), getattr(operator, x.comparison), x.comparator, False))
                    for x in self.source.reindex
                ]
                if self.source.reindex
                else None,
                [op for x in self.annotations for op in self.encoding(x, x.annotation.lower())] if self.annotations else None,
                (coerce_pvalue_columns, ()),
                (coerce_study_size_columns, ()),
                (clean_numeric, ()),
                # Drop insignificant rows before they ever reach the expensive fullmap resolution below.
                (sig, ()),
                (drop_not_significant, ()) if self.release else None,
                [self.node_prep(x, col) for x, col in node_columns],
                (resolve_batch, (specs, db, self.log, self.store.stem, self.config.name, True)),
                [(fullmap_audit, (col, self.store.stem, self.config.name, "passed", True)) for _, col in node_columns] if self.qc else None,
                (value, ("predicate", add("biolink:", self.statement.predicate))),
                (edge_category, ()),
                (value, ("upstream_resource_ids", upstream_resource_ids(self.provenance.repo))),
                (value, ("knowledge_level", self.provenance.knowledge_level)),
                (value, ("agent_type", self.provenance.agent_type)),
                (value, ("resource_id", infores(self.name))) if self.name else None,
                (publications, (publication_curie(self.provenance.repo, self.provenance.publication),)),
                (source_record_urls, (str(self.source.url),)),
                (value, ("sheet_name", self.source.sheet)) if eq(self.source.kind, Files.EXCEL) else None,  # pyright: ignore
                (trim, ()),
                (format_numeric, ()),
                (to_store, (self.store, self.config.name)),
            ]
            return self.clean(tcode)


PHASE_OF: dict[Callable, str] = {
    # Maps each Tcode op callable to a short lowercase phase label for progress UX.
    csv: "load",
    excel: "load",
    idx: "load",
    crop: "filter",
    pick: "filter",
    reindex: "filter",
    coerce_pvalue_columns: "clean",
    coerce_study_size_columns: "clean",
    clean_numeric: "clean",
    level_one: "resolve",
    level_two: "resolve",
    resolve: "resolve",
    resolve_batch: "resolve",
    fullmap_audit: "qc",
    column: "encode",
    edge_category: "edge",
    publications: "provenance",
    source_record_urls: "provenance",
    sig: "significance",
    drop_not_significant: "significance",
    trim: "finalize",
    format_numeric: "finalize",
    to_store: "write",
}

UNKNOWN_PHASE: str = "transform"

_VALUE_PROVENANCE_COLS: frozenset[str] = frozenset({"upstream_resource_ids", "knowledge_level", "agent_type", "resource_id", "sheet_name"})


def _phase_of(fn: Callable, args: tuple[Any, ...]) -> str:
    """Resolve the phase label for an op, handling ``value()`` specially since it spans phases.

    Args:
        fn: Op callable.
        args: Op argument tuple.

    Returns:
        Short lowercase phase label (e.g. ``"load"``, ``"resolve"``,
        ``"edge"``); falls back to ``UNKNOWN_PHASE`` when unrecognized.
    """
    if fn is value:
        col: str = str(args[0]) if args else ""
        if col == "predicate":
            return "edge"
        if col in _VALUE_PROVENANCE_COLS:
            return "provenance"
        return UNKNOWN_PHASE
    return PHASE_OF.get(fn, UNKNOWN_PHASE)


def compile_subgraph(tcode: list[tuple[Callable, tuple[Any]]], *, on_phase: Optional[Callable[[str], None]] = None) -> Path:
    """Execute a Tcode operation list to build a subgraph parquet.

    Args:
        tcode: Cleaned list of ``(callable, args)`` tuples from ``Tcode.collect``.
        on_phase: Optional callback fired when the phase label changes,
            used to drive progress UX.

    Returns:
        Path to the written subgraph parquet.
    """
    last_phase: Optional[str] = None
    acc: Union[pl.LazyFrame, Path, None] = None
    for op in tcode:
        fn: Callable = op[0]
        args: tuple[Any, ...] = op[1]
        if on_phase is not None:
            phase: str = _phase_of(fn, args)
            if phase != last_phase:
                last_phase = phase
                on_phase(phase)
        acc = fn(acc, *args) if acc is not None else fn(*args)  # pyright: ignore
    return acc  # pyright: ignore


def normalize(
    edges: pl.LazyFrame, col: str, names: list[str] = ["id", "name", "category", "taxon", "source", "source_version"]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Normalize disparate node columns into a unified format and remove them from edges.

    Args:
        edges: Source edges LazyFrame containing ``<col>``, ``<col>_name``,
            ``<col>_category``, ``<col>_taxon``, ``<col>_source``, and
            ``<col>_source_version`` columns.
        col: Base node column name (e.g. ``"subject"``).
        names: Output column names for the produced nodes frame.

    Returns:
        Tuple of ``(partial_nodes, modified_edges)`` as LazyFrames.
    """
    cols: list[str] = [col, add(col, "_name"), add(col, "_category"), add(col, "_taxon"), add(col, "_source"), add(col, "_source_version")]
    nodes: pl.LazyFrame = edges.select(cols).unique().rename({k: v for k, v in zip(cols, names)})
    # Ensures category has biolink: prefix.
    nodes = nodes.with_columns(
        pl.when(pl.col("category").str.starts_with("biolink:"))
        .then(pl.col("category"))
        .otherwise(add(pl.lit("biolink:"), pl.col("category")))
        .alias("category")
    )
    # Exports category within a list (null categories stay null for strip_nulls).
    nodes = nodes.with_columns(pl.when(pl.col("category").is_not_null()).then(pl.concat_list(pl.col("category"))).alias("category"))
    edges_out: pl.LazyFrame = edges.drop(cols[1:])
    return nodes, edges_out


def publication_curie(repo: str, publication: str) -> str:
    """Build the publication CURIE.

    Uses the ``PMCID:`` namespace for PubMed Central and ``<REPO>:`` for
    other repositories.

    Args:
        repo: Repository identifier (a ``Repositories`` enum value).
        publication: Publication identifier (e.g. PMID or PMCID).

    Returns:
        CURIE string of the form ``"<prefix>:<publication>"``.
    """
    if eq(repo, Repositories.PUBMED_CENTRAL):
        return add("PMCID:", publication)
    return add(repo, add(":", publication))


def infores(name: str) -> str:
    """Build an infores CURIE from a graph name in lower kebab case.

    Args:
        name: Graph name (typically snake_case).

    Returns:
        ``"infores:<kebab-name>"``.
    """
    return add("infores:", name.lower().replace("_", "-"))


def upstream_resource_ids(repo: Repositories) -> list[str]:
    """Map a publication repository to Translator infores upstream source IDs.

    Args:
        repo: Repository enum value.

    Returns:
        Single-element list containing the matching infores identifier.
    """
    if eq(repo, Repositories.PUBMED_CENTRAL):
        return [InformationResources.PUBMED_CENTRAL.value]
    return [InformationResources.PUBMED.value]


def strip_nulls(r: object, bad: set[str] = {"na", "nan", "null", "none", ""}) -> dict:
    """Remove null keys from an NDJSON-style record.

    Args:
        r: Object expected to be a dict (other types are not stripped).
        bad: Lowercased strings treated as null-equivalent.

    Returns:
        Dict with falsy and ``bad``-valued keys removed; recurses into
        nested dicts and lists.
    """
    return {
        k: [strip_nulls(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else strip_nulls(v) if isinstance(v, dict) else v
        for k, v in r.items()  # pyright: ignore
        if v and str(v).strip().lower() not in bad
    }


def as_list(v: object) -> list[object]:
    """Coerce scalar and list-like values into a plain list.

    Args:
        v: Any value.

    Returns:
        The original list for lists, ``[]`` for ``None``, otherwise a
        single-element list wrapping the value.
    """
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def normalize_biolink_category(v: object) -> Optional[str]:
    """Normalize category strings for RIG target summaries.

    Args:
        v: Raw category value.

    Returns:
        Category with a ``biolink:`` prefix, or ``None`` for empty/non-string
        input.
    """
    if not isinstance(v, str) or not v:
        return None
    if v.startswith("biolink:"):
        return v
    return add("biolink:", v)


def curie_prefix(v: object) -> Optional[str]:
    """Extract compact identifier prefixes for RIG node type summaries.

    Args:
        v: CURIE string (e.g. ``"CHEBI:1234"``).

    Returns:
        Prefix portion (``"CHEBI"``), or ``None`` when no prefix is present.
    """
    if not isinstance(v, str) or ":" not in v:
        return None
    prefix: str = v.split(":", 1)[0]
    return prefix or None


def clean_values(values: list[object]) -> list[str]:
    """Remove empty values and deduplicate stringified RIG summary values.

    Args:
        values: Raw values, possibly nested in lists.

    Returns:
        Sorted, deduplicated list of non-empty stringified values.
    """
    out: list[str] = []
    for value in values:
        for item in as_list(value):
            if item is None:
                continue
            text: str = str(item).strip()
            if not text or text.lower() in {"na", "nan", "null", "none"}:
                continue
            out.append(text)
    return sorted(set(out))


def rig_edge_type_info(lf: pl.LazyFrame, edges_path: Path, ui_explanation: Optional[str]) -> list[dict[str, object]]:
    """Summarize raw edge columns into RIG edge type metadata before node normalization.

    Args:
        lf: Edges LazyFrame to summarize.
        edges_path: Path of the edges file (recorded under ``source_files``).
        ui_explanation: Optional human-readable explanation; falls back to
            ``DEFAULT_RIG_UI_EXPLANATION`` when unset.

    Returns:
        List of per-edge-type dicts, or an empty list when none of the
        expected columns are present.
    """
    names: list[str] = lf.collect_schema().names()
    wanted: list[str] = [
        c
        for c in [
            "subject_category",
            "predicate",
            "object_category",
            "knowledge_level",
            "agent_type",
            "primary_knowledge_source",
            "primary_knowledge_sources",
            "resource_id",
            "upstream_resource_ids",
        ]
        if c in names
    ]
    if not wanted:
        return []

    rows: list[dict[str, Any]] = lf.select(wanted).unique().collect().to_dicts()
    info: list[dict[str, object]] = []
    for row in rows:
        primary_sources: list[str] = clean_values(
            add(
                add(as_list(row.get("primary_knowledge_source")), as_list(row.get("primary_knowledge_sources"))),
                add(as_list(row.get("resource_id")), as_list(row.get("upstream_resource_ids"))),
            )
        )
        edge_type: dict[str, object] = strip_nulls(
            {
                "subject_categories": clean_values([normalize_biolink_category(v) for v in as_list(row.get("subject_category"))]),
                "predicates": clean_values(as_list(row.get("predicate"))),
                "object_categories": clean_values([normalize_biolink_category(v) for v in as_list(row.get("object_category"))]),
                "knowledge_level": row.get("knowledge_level"),
                "agent_type": row.get("agent_type"),
                "primary_knowledge_sources": primary_sources,
                "source_files": [edges_path.name],
                "ui_explanation": ui_explanation or DEFAULT_RIG_UI_EXPLANATION,
            }
        )
        if edge_type:
            info.append(edge_type)
    return info


def rig_node_type_info(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize normalized nodes into RIG node type metadata.

    Args:
        nodes: List of node dicts with ``id`` and ``category`` keys.

    Returns:
        Sorted list of ``{"node_category", "source_identifier_types"}`` dicts.
    """
    buckets: dict[str, set[str]] = {}
    for node in nodes:
        prefixes: list[str] = clean_values([curie_prefix(node.get("id"))])
        for category in clean_values([normalize_biolink_category(v) for v in as_list(node.get("category"))]):
            buckets.setdefault(category, set()).update(prefixes)

    return [strip_nulls({"node_category": category, "source_identifier_types": sorted(prefixes)}) for category, prefixes in sorted(buckets.items())]


def unique_dicts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate small RIG summary dictionaries without adding a new dependency.

    Args:
        rows: List of dict rows.

    Returns:
        New list preserving first-occurrence order with duplicates removed.
    """
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for row in rows:
        key: str = str(row)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def compile_rig(
    name: str,
    version: str,
    description: Optional[str],
    contributions: Optional[list[str]],
    ui_explanation: Optional[str],
    tables: Optional[list[Path]],
    nodes_path: Path,
    edges_path: Path,
    node_type_info: list[dict[str, object]],
    edge_type_info: list[dict[str, object]],
) -> None:
    """Write Translator Resource Ingest Guide metadata alongside KGX outputs.

    Args:
        name: Graph name.
        version: Graph version string.
        description: Optional human-readable description.
        contributions: Optional contributor list; falls back to defaults.
        ui_explanation: Optional edge-type UI explanation (passed through).
        tables: Optional source table paths (unused; kept for API symmetry).
        nodes_path: Path of the nodes file (recorded in the RIG).
        edges_path: Path of the edges file (recorded in the RIG).
        node_type_info: Precomputed node type summaries.
        edge_type_info: Precomputed edge type summaries.
    """
    from tablassert.ingests import to_yaml

    rig_path: Path = Path(f"./{name}_{version}.RIG.yaml")
    rig: dict[str, object] = strip_nulls(
        {
            "name": f"{name} v{version}",
            "source_info": {
                "infores_id": infores(name),
                "name": name,
                "description": description or f"{name} KGX generated by Tablassert.",
                "data_provision_mechanisms": ["file_download"],
                "data_formats": ["kgx"],
                "data_access_locations": [nodes_path.name, edges_path.name],
                "source_status": "unknown",
                "terms_of_use_info": {"terms_of_use_description": TERMS_OF_USE_WARNING},
            },
            "ingest_info": {
                "ingest_categories": ["translator_knowledge_creator"],
                "utility": "Tablassert converts configured tabular source data into KGX nodes and edges for Translator ingestion.",
                "relevant_files": [nodes_path.name, edges_path.name],
                "included_content": "KGX nodes and edges generated from the graph's configured Tablassert table inputs.",
            },
            "provenance_info": {"contributions": contributions or default_rig_contributions()},
            "target_info": {"edge_type_info": edge_type_info, "node_type_info": node_type_info},
        }
    )
    to_yaml(rig_path, rig)


def dedup_stream(p_in: Path, is_edges: bool) -> None:
    """Remove null values from and deduplicate an NDJSON stream.

    Args:
        p_in: Path to the input ``.ndjson.tmp`` file.
        is_edges: When True, also add UUIDs to edges via the Rust deduper.

    Notes:
        Also adds UUIDs to edges.

    Returns:
        ``None``; writes the deduplicated stream alongside ``p_in`` with no
        ``.tmp`` suffix, then deletes ``p_in``.
    """
    p_out: Path = p_in.with_suffix("")

    if p_out.is_file():
        p_out.unlink()

    rs.dedup_ndjson(p_in, p_out, is_edges, "TABLASSERT")

    p_in.unlink()


def fold_unknown_to_supporting_text(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Fold any non-Biolink edge column into ``supporting_text`` as ``col: value`` strings.

    Args:
        lf: Edges LazyFrame possibly carrying non-allowed columns.

    Returns:
        LazyFrame with all non-allowed columns folded into ``supporting_text``
        and dropped.

    Notes:
        Stays fully lazy; null/blank values produce no entry; sorted for
        deterministic output. Existing ``list[str]`` ``supporting_text`` has
        derived entries appended (never clobbered); scalar
        ``supporting_text`` (e.g. a ``method: value`` annotation) is coerced
        to ``list[str]`` first.
    """
    schema: pl.Schema = lf.collect_schema()
    schema_names: list[str] = schema.names()
    unknown: list[str] = sorted(c for c in schema_names if c not in ALLOWED_EDGE_FIELDS)
    if not unknown:
        return lf

    parts: list[pl.Expr] = []
    for col in unknown:
        s: pl.Expr = pl.col(col).cast(pl.String).str.strip_chars()
        blank: pl.Expr = s.is_null() | (s.str.len_chars() == 0)
        entry: pl.Expr = pl.when(blank).then(pl.lit(None, dtype=pl.String)).otherwise(pl.concat_str([pl.lit(f"{col}: "), s]))
        parts.append(entry)

    derived: pl.Expr = pl.concat_list(parts).list.drop_nulls()

    if "supporting_text" in schema_names:
        # Coerce scalar supporting_text to list[str] first, then append derived entries.
        existing: pl.Expr
        if isinstance(schema["supporting_text"], pl.List):
            existing = pl.col("supporting_text")
        else:
            existing = pl.concat_list(pl.col("supporting_text"))
        combined: pl.Expr = existing.list.concat(derived).list.drop_nulls()
    else:
        combined = derived

    return lf.with_columns(combined.alias("supporting_text")).drop(unknown)


def compile_graph(
    subgraphs: list[Path],
    name: str,
    version: str,
    description: Optional[str] = None,
    contributions: Optional[list[str]] = None,
    ui_explanation: Optional[str] = None,
    tables: Optional[list[Path]] = None,
) -> None:
    """Aggregate subgraph parquets for NDJSON KGX export using a lazy scan.

    Args:
        subgraphs: List of section parquet paths to merge.
        name: Graph name (used for output file stems and the RIG).
        version: Graph version string.
        description: Optional human-readable description for the RIG.
        contributions: Optional contributor list for the RIG.
        ui_explanation: Optional UI explanation for the RIG.
        tables: Optional source table list for the RIG.

    Returns:
        ``None``; writes ``<name>_<version>.nodes.ndjson``,
        ``<name>_<version>.edges.ndjson``, and ``<name>_<version>.RIG.yaml``
        in the current working directory.
    """
    p: Path = Path(f"./{name}_{version}.tmp")

    e: Path = p.with_suffix(".edges.ndjson.tmp")  # For labeling.
    if e.exists():
        e.unlink()

    n: Path = p.with_suffix(".nodes.ndjson.tmp")
    if n.exists():
        n.unlink()

    subnodes: list[pl.LazyFrame] = []
    subedges: list[pl.LazyFrame] = []
    edge_type_info: list[dict[str, object]] = []
    for s in subgraphs:
        lf: pl.LazyFrame = pl.scan_parquet(s)
        edge_type_info.extend(rig_edge_type_info(lf, e.with_suffix(""), ui_explanation))

        # Only subject and object become nodes; qualifier columns stay as edge attributes.
        originals: list[str] = [c.removesuffix("_pre_resolution") for c in lf.collect_schema().names() if c.endswith("_pre_resolution")]
        node_cols: list[str] = [c for c in originals if c in ("subject", "object")]
        for col in node_cols:
            partial, lf = normalize(lf, col)
            subnodes.append(partial)
        # Drop internal pre-resolution snapshot columns from final edges.
        lf = lf.drop([c for c in lf.collect_schema().names() if c.endswith("_pre_resolution")])
        lf = fold_unknown_to_supporting_text(lf)
        subedges.append(lf)

    # Collection point: appending to output files.
    node_rows: list[dict[str, object]] = []
    with n.open("a") as f:
        for subnode in subnodes:
            eagernode: pl.DataFrame = subnode.collect().unique()
            node_rows.extend(eagernode.to_dicts())
            eagernode.write_ndjson(f)

    with e.open("a") as f:
        for subedge in subedges:
            eageredge: pl.DataFrame = subedge.collect().unique()
            eageredge.write_ndjson(f)

    dedup_stream(e, is_edges=True)
    dedup_stream(n, is_edges=False)
    compile_rig(
        name,
        version,
        description,
        contributions,
        ui_explanation,
        tables,
        n.with_suffix(""),
        e.with_suffix(""),
        rig_node_type_info(node_rows),
        unique_dicts(edge_type_info),
    )


def resolve_many(
    col: str,
    entities: Iterable[str],
    fullmap: Path,
    taxon: Optional[str] = None,
    prioritize: Optional[list[Categories]] = None,
    avoid: Optional[list[Categories]] = None,
    qc: bool = False,
    column_context: bool = True,
) -> list[dict[str, Any]]:
    """Resolve a batch of raw entity strings to Biolink-normalized records.

    Args:
        col: Source column name to normalize.
        entities: Raw entity strings to resolve.
        fullmap: Path to ``fullmap.redb`` (or its parent directory).
        taxon: Optional NCBITaxon constraint.
        prioritize: Optional Biolink categories preferred on ties.
        avoid: Optional Biolink categories deprioritized on ties.
        qc: When True, attach fullmap audit metadata to the result.
        column_context: Forwarded to ``resolve``.

    Returns:
        List of dicts mirroring a one-row-per-entity resolved frame, with
        internal pre-resolution snapshot columns dropped to match edge output.
    """
    series: pl.Series = pl.Series(col, entities)
    lf: pl.LazyFrame = series.to_frame().lazy()

    lf = column(lf, add("original_", col), col)
    lf = column(lf, add(col, "_pre_resolution"), col)
    lf = level_one(lf, col)
    lf = level_two(lf, col)

    lf = resolve(lf, col, fullmap_db_path(fullmap), taxon=taxon, prioritize=prioritize, avoid=avoid, column_context=column_context)
    if qc:
        lf = fullmap_audit(lf, col, "", "", log=qc)

    df: pl.DataFrame = lf.collect()
    # Drop internal pre-resolution snapshot columns to mirror edge output.
    df = df.drop([c for c in df.columns if c.endswith("_pre_resolution")])
    return df.to_dicts()
