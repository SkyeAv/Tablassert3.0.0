from __future__ import annotations

import re
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any

from tablassert._lazy import LazyModule

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")


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
    chosen: str | None = max(candidates, key=lambda c: fuzz.ratio(c, col)) if candidates else None
    if chosen is None:
        # Biolink class rule: qualifier may only be set when p_value/adjusted_p_value is populated.
        return lf
    expr: pl.Expr = pl.col(chosen).cast(pl.Float64, strict=False)
    band: pl.Expr = (
        pl.when(expr.is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(expr <= 0.001)
        .then(pl.lit("biolink:very_strongly_significant"))
        .when(expr <= 0.01)
        .then(pl.lit("biolink:strongly_significant"))
        .when(expr <= 0.05)
        .then(pl.lit("biolink:significant"))
        .when(expr <= 0.10)
        .then(pl.lit("biolink:suggestive"))
        .otherwise(pl.lit("biolink:not_significant"))
    )
    return lf.with_columns(band.alias(out))


# --- Column-name classification fragments ------------------------------------
# Real-world column labels separate tokens with spaces, underscores, hyphens, or
# dots ("p value", "p_value", "p-value", "p.value"); treat any run of these as an
# optional token separator shared by every pattern below.
_SEP: str = r"[\s_.\-]*"
# "value" spelled val / value, optionally plural (vals / values).
_VALUE: str = r"val(?:ue)?s?"
# "adjusted" spelled adj / adjusted.
_ADJUSTED: str = r"adj(?:usted)?"
# Optional trailing numeric qualifier for multi-phenotype outputs ("pvalue1",
# "p_value_2", "padj_1"): a separator-or-nothing then digits.
_NUMQUAL: str = r"(?:[\s_.\-]*\d+)?"

# A P value token: leading "p" then a "value" word across an optional separator,
# with an optional trailing numeric qualifier ("p value", "p-value", "pvalue",
# "p vals", "pvalue1", "p_value_2").
PVALUE_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b p {_SEP} {_VALUE} {_NUMQUAL} \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A Q value token (Storey's q value); same shape as the P value token.
QVALUE_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b q {_SEP} {_VALUE} {_NUMQUAL} \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A P value already marked adjusted, with an optional trailing numeric qualifier
# ("padj", "p.adj", "p adjusted", "padj_1").
PADJ_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b p {_SEP} {_ADJUSTED} {_NUMQUAL} \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A bare "P" standing as its own token (common GWAS convention: "P", "p SMR",
# "gwas p", plus underscore/dot/hyphen-glued forms like "raw_p", "snp_p",
# "p_nominal"). "Its own token" means not glued to a letter or digit, so gene/
# protein and chemistry names ("p53", "p16", "pH", "protein", "phosphate") and
# words merely ending in p ("top value", "group value") stay excluded. A plain
# \bp\b would also reject the underscore-glued forms because \b treats "_" as a
# word char, hence the explicit alphanumeric lookarounds.
BARE_P_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"""
    (?<![A-Za-z0-9]) p (?![A-Za-z0-9])
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Adjustment methods that by themselves imply an adjusted P value, unlike the
# generic words "adjusted"/"corrected" which also modify hazard/odds ratios.
STANDALONE_ADJUSTED_PATTERN: re.Pattern[str] = re.compile(
    r"""
    \b
    (?:
        fdr                       # Benjamini-Hochberg false discovery rate
        | bonferroni              # Bonferroni correction
        | holm                    # Holm-Bonferroni correction
        | false\ discovery\ rate  # FDR spelled out (literal spaces)
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Generic adjustment words that only imply an adjusted P value when a P/Q value
# token is also present (see pvalue_target).
CONTEXTUAL_ADJUSTED_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b
    (?:
        {_ADJUSTED}    # adj / adjusted
        | corrected    # corrected
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Stem of "significant"/"significance": a deliberate substring match (no word
# boundary) so every inflection is caught; these columns are categorical flags.
SIGNIFICANCE_FLAG_PATTERN: re.Pattern[str] = re.compile(
    r"""
    significan
    """,
    re.IGNORECASE | re.VERBOSE,
)


def pvalue_target(name: str) -> str | None:
    """Map a column name to its canonical Biolink-compliant target name.

    Args:
        name: Raw source column name.

    Returns:
        ``"p_value"``, ``"adjusted_p_value"``, or ``None`` if the name does
        not look like a p/q-value column.

    Notes:
        Tokens are delimiter-anchored so "Group value" / "top value" style
        substrings are not falsely matched. A bare ``"P"`` counts when it stands
        as its own token (delimited by whitespace, ``_``, ``.`` or ``-``), covering
        GWAS conventions like ``"P"``, ``"gwas p"`` and ``"raw_p"``/``"snp_p"`` while
        excluding ``"p53"``/``"pH"``/``"protein"``. ``"padj"``/``"p.adj"`` cover DESeq2
        conventions. Value/adjusted tokens may carry a trailing numeric qualifier
        (``"pvalue1"``, ``"padj_1"``). ``"adj"``/``"adjusted"``/``"corrected"`` only count
        alongside a p/q-value token, since they are generic words also used for
        adjusted hazard/odds ratios (unlike ``fdr``/``bonferroni``/``holm``).
        ``"significance"``/``"significant"`` columns are categorical flags, not the
        numeric value, so they are excluded unless a p/q-value token is also present.
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
        target: str | None = pvalue_target(n)
        if target:
            buckets.setdefault(target, []).append(n)

    renames: dict[str, str] = {}
    for target, candidates in buckets.items():
        reference: str = target.replace("_", " ")
        chosen: str = max(candidates, key=lambda c: fuzz.ratio(c, reference))
        if chosen != target:
            renames[chosen] = target

    return lf.rename(renames) if renames else lf


# --- Study-size fragments ----------------------------------------------------
# Population units whose count denotes a study size. Bare "cohort"/"cohort_id"
# still do not match: every pattern requires a count/quantity word alongside.
_UNIT: str = r"samples?|participants?|subjects?|individuals?|patients?|cases?|cohorts?"
# Count nouns following a unit ("sample_count", "participants_n", "cases_size").
_COUNT_WORD: str = r"n|count|number|size"
# Quantity words preceding a unit ("number of samples", "total participants").
_QUANTITY: str = r"n|num|number|count|total"

# Whole-name matches for the canonical study-size labels ("n", "sample_size",
# "study size", "study_n", "n_total", "cohort_size", "supporting_study_size").
# "samplesize" needs no separate alternative: sample<_SEP>size matches it at zero
# separator width.
STUDY_SIZE_EXACT_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    ^
    (?:
        n
        | total {_SEP} n
        | n {_SEP} total
        | sample {_SEP} size
        | study {_SEP} size
        | study {_SEP} n
        | cohort {_SEP} size
        | supporting {_SEP} study {_SEP} size
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A population unit followed by a count noun ("sample_count", "participant_count",
# "cohort_count", "enrollment_count", "enrolled_count", "samples_n").
STUDY_SIZE_COUNT_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b
    (?:
        {_UNIT}
        | enrollment
        | enrolled
    )
    {_SEP}
    (?: {_COUNT_WORD} )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A quantity word (optionally followed by "of") before a population unit
# ("number of samples", "num_samples", "n_samples", "total participants").
STUDY_SIZE_PREFIX_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b
    (?: {_QUANTITY} )
    {_SEP}
    (?: of {_SEP} )?
    (?: {_UNIT} )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# A population unit followed by a bare "n" ("samples_n", "participants_n"). A
# documented subset of COUNT (which also matches unit<_SEP>n), kept to make the
# trailing-n convention explicit.
STUDY_SIZE_SUFFIX_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b
    (?: {_UNIT} )
    {_SEP}
    n
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Whole-name singular labels that unambiguously denote a study size on their own
# ("participants", "enrollment", "enrolled").
STUDY_SIZE_SINGLETON_PATTERN: re.Pattern[str] = re.compile(
    r"""
    ^
    (?:
        participants
        | enrollment
        | enrolled
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def study_size_target(name: str) -> str | None:
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
        return "supporting_study_size"  # pragma: no cover -- documented subset of COUNT (unit<_SEP>n); kept explicit, never reached
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


# --- Effect-type name fragments ----------------------------------------------
# Labels naming WHICH statistic an effect size is expressed in. Whole-name matches
# cover bare labels ("metric", "effect type"); token matches cover qualified forms
# ("effect size type"). "_" is a word char for \b, so identifier-like names such as
# "metric_value" fall through.
EFFECT_TYPE_EXACT_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    ^
    (?:
        effect {_SEP} type
        | effect {_SEP} metric
        | statistic(?:al)? {_SEP} type
        | metric {_SEP} type
        | metric
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
EFFECT_TYPE_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b
    (?:
        effect {_SEP} type
        | effect {_SEP} metric
        | effect {_SEP} size {_SEP} type
        | statistic(?:al)? {_SEP} type
        | metric {_SEP} type
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# --- Effect-size fragments -----------------------------------------------------
# Numeric effect/statistic column labels. Whole-name matches cover bare statistic
# tokens ("ES", "OR", "HR", "beta", "rho", "r", plus the old ``relationship_strength``
# name); token matches require a multi-word statistic label, and word-boundary
# anchoring keeps identifier columns out ("correlation_id" fails the trailing \b
# because "_" is a word char).
EFFECT_SIZE_EXACT_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    ^
    (?:
        effect {_SEP} size
        | relationship {_SEP} strength
        | es
        | beta
        | beta {_SEP} coefficients?
        | log2 {_SEP} fc
        | log2 {_SEP} fold {_SEP} change
        | odds {_SEP} ratio
        | or
        | hazard {_SEP} ratio
        | hr
        | risk {_SEP} ratio
        | correlation
        | rho
        | r
    )
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)
EFFECT_SIZE_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    \b
    (?:
        effect {_SEP} size
        | relationship {_SEP} strength
        | beta {_SEP} coefficients?
        | log2 {_SEP} fc
        | log2 {_SEP} fold {_SEP} change
        | odds {_SEP} ratio
        | hazard {_SEP} ratio
        | risk {_SEP} ratio
        | correlation {_SEP} coefficients?
        | correlation
        | spearman(?:s)? {_SEP} rho
        | pearson(?:s)? {_SEP} r
        | kendall(?:s)? {_SEP} tau
        | rho
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Token matches are \b-anchored like the p-value/study-size patterns: the explicit
# multi-token alternatives span separators ("spearman rho", "spearman_rho"), while
# unknown underscore-glued forms deliberately fall through.


def effect_type_target(name: str) -> str | None:
    """Map effect-type-like column names to the canonical ``effect_type`` slot.

    Args:
        name: Raw source column name.

    Returns:
        ``"effect_type"`` when the name looks like an effect-type/metric label
        ("effect type", "effect metric", "statistic type", "metric type", bare
        "metric"), else ``None``.
    """
    if EFFECT_TYPE_EXACT_PATTERN.search(name):
        return "effect_type"
    if EFFECT_TYPE_TOKEN_PATTERN.search(name):
        return "effect_type"
    return None


def effect_size_target(name: str) -> str | None:
    """Map effect-size-like column names to the canonical ``effect_size`` slot.

    Args:
        name: Raw source column name.

    Returns:
        ``"effect_size"`` when the name matches any of the effect-size
        patterns, else ``None``.

    Notes:
        Effect-type/metric labels are categorical, not the numeric size, so
        they are excluded here and belong to ``effect_type_target`` (mirroring
        the significance-flag exclusion in ``pvalue_target``). The old
        ``relationship_strength`` name is a candidate, so it is renamed
        forward to ``effect_size``.
    """
    if effect_type_target(name):
        return None
    if EFFECT_SIZE_EXACT_PATTERN.search(name):
        return "effect_size"
    if EFFECT_SIZE_TOKEN_PATTERN.search(name):
        return "effect_size"
    return None


def coerce_effect_size_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename effect-size-like columns to Biolink KGX-compliant ``effect_size`` (Biolink PR #1774).

    Picks a single best fuzzy match and leaves other candidate columns
    untouched.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the chosen column renamed (no-op if no match).

    Notes:
        The old ``relationship_strength`` name is among the candidates, so
        existing configs are renamed forward to ``effect_size``.
    """
    # Picks a single best fuzzy match and leaves other candidate columns untouched.
    from rapidfuzz import fuzz

    names: list[str] = lf.collect_schema().names()
    candidates: list[str] = [n for n in names if effect_size_target(n)]
    if not candidates:
        return lf

    target: str = "effect_size"
    reference: str = target.replace("_", " ")
    chosen: str = max(candidates, key=lambda c: fuzz.ratio(c, reference))
    if chosen == target:
        return lf
    return lf.rename({chosen: target})


# --- Effect-type value coercion ------------------------------------------------
# Raw value aliases -> canonical EffectTypes values (Biolink PR #1774). Keys are
# matched case/separator-insensitively: both sides are lower-cased with every
# separator/apostrophe stripped ("Cohen's d" -> "cohensd").
_EFFECT_TYPE_ALIASES: tuple[tuple[str, str], ...] = (
    ("Cohen's d", "cohens_d"),
    ("odds ratio", "odds_ratio"),
    ("OR", "odds_ratio"),
    ("hazard ratio", "hazard_ratio"),
    ("HR", "hazard_ratio"),
    ("risk ratio", "relative_risk"),
    ("RR", "relative_risk"),
    ("relative risk", "relative_risk"),
    ("Spearman", "spearmans_rho"),
    ("spearman rho", "spearmans_rho"),
    ("spearman's rho", "spearmans_rho"),
    ("Pearson", "pearsons_r"),
    ("pearson r", "pearsons_r"),
    ("Kendall", "kendalls_tau"),
    ("log2FC", "log2_fold_change"),
    ("log2 fold change", "log2_fold_change"),
    ("beta", "regression_coefficient"),
    ("regression coefficient", "regression_coefficient"),
    ("SMD", "standardized_mean_difference"),
    ("eta squared", "eta_squared"),
    ("eta2", "eta_squared"),
    ("omega squared", "omega_squared"),
    ("MCC", "matthews_correlation_coefficient"),
    ("matthews", "matthews_correlation_coefficient"),
    ("wald", "wald_ratio"),
    ("IVW", "inverse_variance_weighted"),
    ("MR-Egger", "mr_egger"),
    ("weighted median", "weighted_median"),
    ("Glass", "glasss_delta"),
    ("polychoric", "polychoric_correlation"),
    ("Goodman-Kruskal", "goodman_kruskal_gamma"),
    ("r2", "r2_linkage_disequilibrium"),
    ("LD r2", "r2_linkage_disequilibrium"),
    ("Hedges", "hedges_g"),
    ("SSMD", "strictly_standardized_mean_difference"),
    ("correlation coefficient", "correlation_coefficient"),
)
# Minimum fuzz.ratio for a raw value to count as a fuzzy hit against the canonical
# values; anything lower is dropped to null (the biolink range is the enum).
_EFFECT_TYPE_FUZZY_SCORE: float = 80.0


def _normalize_effect_type(value: str) -> str:
    """Lower-case and strip separators/apostrophes for case/separator-insensitive matching."""
    # \u2019 is the right single quote (curly apostrophe) so both apostrophe styles normalize alike.
    return re.sub(r"[\s_.\-'\u2019]+", "", value.lower())


@cache
def _effect_type_vocab() -> tuple[dict[str, str], tuple[str, ...]]:
    """Normalized alias table and canonical values for the 25 ``EffectTypes`` values (cached).

    Returns:
        Tuple of ``(alias table, canonical values)``: normalized-key ->
        canonical-value map covering every permissible value plus the common
        spellings/abbreviations, and the 25 canonical values verbatim.
    """
    from tablassert.biolink import EFFECT_TYPE_VALUES

    table: dict[str, str] = {_normalize_effect_type(value): value for value in EFFECT_TYPE_VALUES}
    for raw, canonical in _EFFECT_TYPE_ALIASES:
        table[_normalize_effect_type(raw)] = canonical
    return table, EFFECT_TYPE_VALUES


@lru_cache(maxsize=4096)
def _map_effect_type_value(raw: Any) -> str | None:
    """Map one raw ``effect_type`` value to a canonical value, or null when nothing matches.

    Args:
        raw: Raw cell value (possibly null).

    Returns:
        The canonical ``EffectTypes`` value on an exact/alias hit or a fuzzy
        hit scoring at least ``_EFFECT_TYPE_FUZZY_SCORE``; ``None`` otherwise.
    """
    if raw is None:
        return None
    text: str = str(raw).strip()
    if not text:
        return None
    table, values = _effect_type_vocab()
    key: str = _normalize_effect_type(text)
    hit: str | None = table.get(key)
    if hit is not None:
        return hit
    # Fuzzy fallback against the 25 canonical values only.
    from rapidfuzz import fuzz

    best: str = max(values, key=lambda v: fuzz.ratio(key, v))
    if fuzz.ratio(key, best) >= _EFFECT_TYPE_FUZZY_SCORE:
        return best
    return None


def coerce_effect_type_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename effect-type-like columns to ``effect_type`` and coerce their values (Biolink PR #1774).

    Picks a single best fuzzy column match, then maps every value to one of the
    25 permissible ``EffectTypes`` enum values.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the chosen column renamed to ``effect_type`` and its
        values coerced (no-op if no effect-type-like column is present).

    Notes:
        Values are matched case/separator-insensitively against an exact/alias
        table first, then by rapidfuzz fallback against the 25 canonical
        values. The Biolink range of ``effect_type`` is the enum, so values
        matching nothing are dropped to null rather than carried through.

    Warnings:
        Biolink class rule (PR #1774): ``effect_type`` may only be populated
        when ``effect_size`` is populated. After value coercion, ``effect_type``
        is nulled on every row where ``effect_size`` is null, and nulled
        entirely when no ``effect_size`` column is present (the same shape of
        class rule ``sig`` documents for the significance qualifier).
    """
    # Picks a single best fuzzy match and leaves other candidate columns untouched.
    from rapidfuzz import fuzz

    names: list[str] = lf.collect_schema().names()
    candidates: list[str] = [n for n in names if effect_type_target(n)]
    if not candidates:
        return lf

    target: str = "effect_type"
    reference: str = target.replace("_", " ")
    chosen: str = max(candidates, key=lambda c: fuzz.ratio(c, reference))
    lf = lf.rename({chosen: target}) if chosen != target else lf

    mapped: pl.Expr = pl.col(target).cast(pl.String).map_elements(_map_effect_type_value, return_dtype=pl.String)
    lf = lf.with_columns(mapped.alias(target))

    # Biolink class rule: effect_type may only be populated when effect_size is populated.
    if "effect_size" in lf.collect_schema().names():
        populated: pl.Expr = pl.col("effect_size").cast(pl.Float64, strict=False).is_not_null()
        guarded: pl.Expr = pl.when(populated).then(pl.col(target)).otherwise(pl.lit(None, dtype=pl.String))
        return lf.with_columns(guarded.alias(target))
    return lf.with_columns(pl.lit(None, dtype=pl.String).alias(target))
