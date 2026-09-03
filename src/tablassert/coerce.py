from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, NamedTuple

from tablassert._lazy import LazyModule

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")


def sig(lf: pl.LazyFrame, col: str = "p_value", out: str = "statistical_significance_qualifier") -> pl.LazyFrame:
    """Create the ``statistical_significance_qualifier`` column (Biolink PR #1766).

    Picks the p-value column via ``pvalue_target`` classification (raw ``p_value``
    preferred, canonical-wins), then buckets the value into one of five significance bands.

    Candidate selection mirrors :func:`coerce_pvalue_columns`: a column counts
    only when :func:`pvalue_target` accepts it (not by naive substring), an
    existing canonical column always wins over a higher-scoring spaced alias,
    and ties break by ``rapidfuzz`` ratio against ``target.replace("_", " ")``.
    A raw ``p_value`` column is preferred over ``adjusted_p_value``.

    Args:
        lf: Source LazyFrame.
        col: Preferred canonical target (``"p_value"`` raw or
            ``"adjusted_p_value"``); the other bucket is the fallback. Kept for
            API compatibility — selection still runs through :func:`pvalue_target`.
        out: Output qualifier column name.

    Returns:
        LazyFrame with the new qualifier column. No-op when no p-value-like
        column is present.

    Notes:
        Five-band cascade matches ``StatisticalSignificanceQualifierEnum``
        verbatim, bare tokens included: biolink-model 4.4.4 (PR #1766/#1770
        lineage) attaches the slot to ``Association`` with the enum as its
        range, so the emitted values are the enum tokens themselves --
        ``"strongly_significant"``, never a ``biolink:``-prefixed CURIE.
        Boundaries are hardcoded because the enum definitions are canonical.

    Warnings:
        Edges are never dropped here. No p-value column → qualifier absent;
        null p-value → null qualifier. The qualifier may only be set when
        ``p_value``/``adjusted_p_value`` is populated (Biolink class rule).
    """
    names: list[str] = lf.collect_schema().names()
    # The significance source is chosen by the very same plan ``coerce_pvalue_columns`` runs --
    # one implementation, not a mirrored one. A column is a candidate only when ``pvalue_target``
    # accepts it, NOT by naive substring, so non-p-value columns that merely contain the
    # reference text stay out of the qualifier.
    plan: list[_Choice] = _plan(names, (_PVALUE_RULE,))
    if not plan:
        # Biolink class rule: qualifier may only be set when p_value/adjusted_p_value is populated.
        return lf
    # Prefer the requested target (raw ``p_value`` by default); fall back to whichever p-value
    # bucket is present. Raw p-value is the canonical significance source; adjusted is the fallback.
    # ``plan`` is in first-seen target order, so ``plan[0]`` is that fallback.
    choice: _Choice = next((c for c in plan if c.target == col), plan[0])
    chosen: str = choice.chosen
    expr: pl.Expr = pl.col(chosen).cast(pl.Float64, strict=False)
    # A -log10(p) score column must be un-logged before banding, or the bands
    # invert (a score of 8 means p = 1e-8, not p = 8.0 -> not_significant).
    if is_neglog10_column(chosen):
        expr = _unlog10(expr)
    band: pl.Expr = (
        pl.when(expr.is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(expr <= 0.001)
        .then(pl.lit("very_strongly_significant"))
        .when(expr <= 0.01)
        .then(pl.lit("strongly_significant"))
        .when(expr <= 0.05)
        .then(pl.lit("significant"))
        .when(expr <= 0.10)
        .then(pl.lit("suggestive"))
        .otherwise(pl.lit("not_significant"))
    )
    return lf.with_columns(band.alias(out))


# --- Column-name classification fragments ------------------------------------
# Real-world column labels separate tokens with spaces, underscores, hyphens, or
# dots ("p value", "p_value", "p-value", "p.value"); treat any run of these as an
# optional token separator shared by every pattern below.
_SEP: str = r"[\s_.\-]*"
# "value" spelled val / value, optionally plural (vals / values).
_VALUE: str = r"val(?:ue)?s?"
# "adjusted" spelled adj / adjust / adjusted / adjustment. The longer forms matter because
# "p_adjust" is a real DESeq2-family header: without them it falls through to the RAW
# p-value bucket and an FDR-adjusted number ships as `p_value`.
_ADJUSTED: str = r"adj(?:ust(?:ed|ment)?)?"
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
# token is also present (see pvalue_target). Anchored with alphanumeric lookarounds
# rather than \b for the same reason BARE_P_TOKEN_PATTERN is: "_" is a word char, so
# \b never fires between "adjusted" and "_p_value" and the underscore-glued spellings
# ("adjusted_p_value", "adj_p_value", "corrected_p_value") silently read as RAW
# p-values -- shipping an FDR-adjusted number in the `p_value` slot.
CONTEXTUAL_ADJUSTED_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    (?<![A-Za-z0-9])
    (?:
        {_ADJUSTED}    # adj / adjust / adjusted / adjustment
        | corrected    # corrected
    )
    (?![A-Za-z0-9])
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
# A -log10(p/q) score column: an explicit negation marker (negative / negated /
# neg / -) before a log/log10 p-or-q token ("negative log p value" — the
# mokg-v12 HOYER1 spelling, "negative log10 p value", "-log10(p)",
# "neg log10 q value"). The trailing token must be a *complete* p/q-value
# token (or a bare delimited P), so the [pq] cannot swallow the first letter
# of an unrelated word ("negative log protein" stays out). A plain
# "log10 p value" without a negation marker does NOT match: the sign
# convention is ambiguous there, so those columns keep riding verbatim rather
# than being un-logged on a guess.
NEGLOG10_PVALUE_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    (?<![A-Za-z0-9])
    (?: negative | negated | neg | - )
    [\s_.\-()]*
    log (?: 10 )?
    [\s_.\-()]*
    (?:
        [pq] {_SEP} {_VALUE} {_NUMQUAL}    # complete p/q-value token ("log p value", "log q-value", "log pvalue")
        | p (?! [A-Za-z0-9])               # bare P standing alone ("-log10(p)", "-log10 P")
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_neglog10_column(name: str) -> bool:
    """Return True when a column reports a -log10(p/q) score instead of the raw p/q value.

    Args:
        name: Raw source column name.

    Returns:
        True when the name carries an explicit negation marker before a
        log/log10 p-or-q token, else False. Plain ``log``/``log10`` spellings
        without a marker are deliberately excluded (sign convention
        ambiguous), as is everything without a log token at all.
    """
    return bool(NEGLOG10_PVALUE_PATTERN.search(name))


def _unlog10(expr: pl.Expr) -> pl.Expr:
    """Convert a -log10 score expression back to its original scale (10**-x).

    Float64 underflow floors extreme scores at ``0.0`` — indistinguishable
    from ``p ~ 0`` in practice, and the significance band is identical either
    way. Nulls stay null.
    """
    return pl.lit(10.0) ** (-expr)


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
    A chosen column that reports a -log10(p/q) score (see
    :func:`is_neglog10_column`) is un-logged (``p = 10**-x``) as it is renamed,
    so the slot receives the p-value the model types it as.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the chosen columns renamed (no-op if no candidates).
    """
    return _apply(lf, (_PVALUE_RULE,))


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
# The Biolink ``Association`` slot ``number_of_cases`` counts cases carrying the
# phenotype/disease, not the study population. It would otherwise match
# PREFIX ("number of cases") and be destroyed by the rename to ``study_size``,
# so the exact slot (separator-tolerant, like the patterns above) is exempt.
STUDY_SIZE_EXEMPT_PATTERN: re.Pattern[str] = re.compile(
    rf"""
    ^
    number {_SEP} of {_SEP} cases
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def study_size_target(name: str) -> str | None:
    """Map study-size-like column names to the canonical ``study_size`` slot.

    Bare ``"n"`` is allowed, but other matches need explicit sample/study-size
    context.

    Args:
        name: Raw source column name.

    Returns:
        ``"study_size"`` when the name matches any of the study-size patterns,
        else ``None``.
    """
    if STUDY_SIZE_EXEMPT_PATTERN.search(name):
        return None
    if STUDY_SIZE_EXACT_PATTERN.search(name):
        return "study_size"
    if STUDY_SIZE_COUNT_PATTERN.search(name):
        return "study_size"
    if STUDY_SIZE_PREFIX_PATTERN.search(name):
        return "study_size"
    if STUDY_SIZE_SUFFIX_PATTERN.search(name):
        return "study_size"  # pragma: no cover -- documented subset of COUNT (unit<_SEP>n); kept explicit, never reached
    if STUDY_SIZE_SINGLETON_PATTERN.search(name):
        return "study_size"
    return None


def coerce_study_size_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename study-size-like columns to the current Biolink ``study_size`` Study slot.

    Picks a single best fuzzy match and drops the other study-size aliases so
    synonym columns cannot leak into ``supporting_text`` or a ``StudyResult``
    description.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the chosen column renamed (no-op if no match).
    """
    return _apply(lf, (_STUDY_SIZE_RULE,))


# --- Study metadata (Biolink PR #1770 replacements) ---------------------------
# The deprecated ``supporting study *`` association slots each name their exact
# replacement Study node property (``deprecated_element_has_exact_replacement``).
# Whole-name, separator-anchored patterns (same convention as the study-size
# machinery above) accept spaced/underscored/hyphenated/dotted spellings.
STUDY_METADATA_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"^ supporting {_SEP} study {_SEP} cohort $", re.IGNORECASE | re.VERBOSE), "study_cohort"),
    (re.compile(rf"^ supporting {_SEP} study {_SEP} context $", re.IGNORECASE | re.VERBOSE), "study_context"),
    (re.compile(rf"^ supporting {_SEP} study {_SEP} date {_SEP} range $", re.IGNORECASE | re.VERBOSE), "study_date_range"),
    (re.compile(rf"^ supporting {_SEP} study {_SEP} method {_SEP} description $", re.IGNORECASE | re.VERBOSE), "study_method_description"),
    (re.compile(rf"^ supporting {_SEP} study {_SEP} method {_SEP} types? $", re.IGNORECASE | re.VERBOSE), "study_method_types"),
    (re.compile(rf"^ supporting {_SEP} study {_SEP} size $", re.IGNORECASE | re.VERBOSE), "study_size"),
)


def study_metadata_target(name: str) -> str | None:
    """Map a deprecated ``supporting study *`` column name to its ``study_*`` replacement.

    Args:
        name: Raw source column name.

    Returns:
        The current Biolink ``Study`` property name, or ``None`` when the name
        is not a deprecated supporting-study metadata slot.
    """
    for pattern, target in STUDY_METADATA_RENAMES:
        if pattern.search(name):
            return target
    return None


def coerce_study_metadata_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Rename deprecated ``supporting_study_*`` columns onto current ``study_*`` slots.

    Biolink PR #1770 deprecated the six ``supporting study *`` association slots
    and replaced each with a ``Study`` node property; this op performs that exact
    replacement on column names so the values reach the inlined ``Study`` as real
    typed fields. A canonical target wins over a deprecated sibling, and duplicate
    aliases are dropped because they are synonyms rather than independent annotations.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with every renamable deprecated column renamed or dropped when
        its canonical target is already present.
    """
    return _apply(lf, (_STUDY_METADATA_RULE,))


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


# --- Shared coercion engine ---------------------------------------------------
# Every coercion below is the same three steps -- claim column names by a classifier,
# pick one winner per canonical target, then rename/drop/rewrite -- so the steps live
# here once and each coercion is reduced to the data that distinguishes it.


def _is_raw_pvalue(name: str) -> bool:
    """Return True when a p/q-value candidate carries the value itself, not a -log10 score.

    Args:
        name: Candidate column name.

    Returns:
        True unless the name reports a ``-log10(p/q)`` score.

    Notes:
        Used as the p-value rule's ``prefer`` predicate. A raw candidate must always
        beat a ``-log10`` alias of the same statistic: the raw column holds the p/q
        value, while the alias still needs un-logging. Narrowing the pool *before*
        fuzzy ranking is what stops a short raw name (``"P"``, ``"FDR"``) losing to a
        long ``-log10`` alias it would then be un-logged over.
    """
    return not is_neglog10_column(name)


@dataclass(frozen=True, slots=True)
class _Rule:
    """One column coercion, expressed as data rather than as a hand-written op.

    Attributes:
        classify: Maps a raw column name to its canonical target, or ``None`` when
            the rule does not claim the name.
        drop_aliases: Whether losing candidates are dropped. Dropping keeps synonym
            columns from leaking into ``supporting_text`` or a ``StudyResult``
            description as a duplicate of the value that already reached its slot.
        prefer: Optional narrowing predicate applied before fuzzy ranking; when it
            excludes every candidate the full pool is used unchanged.
        fuzzy: When False the first candidate in schema order wins. Set for rules
            whose aliases are exact synonyms of one slot, where a fuzzy score has
            nothing to discriminate between them.
        unlog: Whether a chosen ``-log10`` source must be converted back to its
            original scale as it lands on the numeric slot.
    """

    classify: Callable[[str], str | None]
    drop_aliases: bool
    prefer: Callable[[str], bool] | None = None
    fuzzy: bool = True
    unlog: bool = False


class _Choice(NamedTuple):
    """One resolved coercion: which column wins a canonical slot, and what loses.

    Attributes:
        target: Canonical slot the winner is renamed to.
        chosen: Winning source column (equal to ``target`` when already canonical).
        aliases: Losing candidates, in schema order.
        rule: Rule that claimed them; carries the alias and un-log policy.
    """

    target: str
    chosen: str
    aliases: tuple[str, ...]
    rule: _Rule


# Losing candidates are dropped by EVERY rule. A loser is another spelling of the value that
# already reached the canonical slot, and anything left on the frame is folded into
# ``supporting_text`` by ``lib.fold_unknown_to_supporting_text`` -- so keeping it emits the same
# number twice, once as a typed Biolink slot and once as free text (``effect_size`` 0.85 beside
# ``"odds ratio: 0.85"``, or an un-logged ``p_value`` beside ``"negative log10 p value: 8.0"``,
# which reads as a contradiction rather than a duplicate).
#
# The cost is real and deliberate: a table whose headers are genuinely DIFFERENT statistics that
# all classify onto one slot -- ``beta`` beside ``odds ratio``, or the two-analysis header set
# ``p_value_analysis1``/``p_value_analysis2`` -- keeps only the winner. Give such columns names
# the classifiers do not claim if every one of them must survive.
_PVALUE_RULE: _Rule = _Rule(pvalue_target, drop_aliases=True, prefer=_is_raw_pvalue, unlog=True)
_STUDY_SIZE_RULE: _Rule = _Rule(study_size_target, drop_aliases=True)
# The deprecated ``supporting study *`` spellings are exact replacements for one Study property
# (PR #1770), not independent annotations, so they are ranked by schema order: a fuzzy score has
# nothing to discriminate between exact synonyms.
_STUDY_METADATA_RULE: _Rule = _Rule(study_metadata_target, drop_aliases=True, fuzzy=False)
_EFFECT_SIZE_RULE: _Rule = _Rule(effect_size_target, drop_aliases=True)
_EFFECT_TYPE_RULE: _Rule = _Rule(effect_type_target, drop_aliases=True)

# Order is precedence: the first rule to claim a name owns it. ``coerced_target`` reads
# this order and ``coerce_columns`` applies it, so the classification a config-time
# validator reports and the rename the build performs can no longer drift apart.
_RULES: tuple[_Rule, ...] = (_PVALUE_RULE, _STUDY_SIZE_RULE, _STUDY_METADATA_RULE, _EFFECT_SIZE_RULE, _EFFECT_TYPE_RULE)


def _select(candidates: Sequence[str], target: str, rule: _Rule) -> str:
    """Pick the column a canonical slot should read from.

    Args:
        candidates: Column names the rule bucketed onto ``target``.
        target: Canonical slot name.
        rule: Rule supplying the ``prefer`` predicate and ``fuzzy`` policy.

    Returns:
        The winning column name.

    Notes:
        The canonical-wins half of the selection rule lives in :func:`_plan`, which
        can see the whole schema; this function only ranks among aliases.
    """
    pool: list[str] = [c for c in candidates if rule.prefer(c)] if rule.prefer else list(candidates)
    # A predicate that excludes everything narrows nothing: fall back to the full pool.
    pool = pool or list(candidates)
    if not rule.fuzzy:
        return pool[0]
    from rapidfuzz import fuzz

    reference: str = target.replace("_", " ")
    return max(pool, key=lambda c: fuzz.ratio(c, reference))


def _plan(names: Sequence[str], rules: Sequence[_Rule]) -> list[_Choice]:
    """Resolve which column wins each canonical slot, without touching the frame.

    Args:
        names: Schema column names, in schema order.
        rules: Rules to apply, in precedence order.

    Returns:
        One :class:`_Choice` per claimed target, in rule order and then first-seen
        target order. Empty when no rule claims anything.

    Notes:
        **An existing canonical column always wins, whether or not the classifier
        claims it.** Judging that against the whole schema rather than against the
        candidate bucket matters for the study-metadata rule, whose classifier
        rejects the canonical names it renames onto (``study_metadata_target
        ("study_size")`` is ``None``), so a frame already carrying ``study_size``
        has the canonical column in the schema but not in the bucket. It also buys
        a safety property the caller relies on: a rename can never target a column
        that already exists, so the rename dict can never collide.

        A name claimed by one rule is withheld from later rules, mirroring the
        first-wins precedence :func:`coerced_target` reports. The classifiers are
        already disjoint on the canonical targets, so this is a guard rather than a
        behavior -- and it is what lets the whole plan be computed from the original
        schema in one pass instead of re-reading the schema after every rename.
    """
    present: frozenset[str] = frozenset(names)
    claimed: set[str] = set()
    plan: list[_Choice] = []
    for rule in rules:
        buckets: dict[str, list[str]] = {}
        for name in names:
            if name in claimed:
                continue
            target: str | None = rule.classify(name)
            if target is not None:
                buckets.setdefault(target, []).append(name)
        for target, candidates in buckets.items():
            chosen: str = target if target in present else _select(candidates, target, rule)
            plan.append(_Choice(target, chosen, tuple(c for c in candidates if c != chosen), rule))
            claimed.update(candidates)
    return plan


def _value_exprs(plan: Sequence[_Choice], final_names: frozenset[str]) -> list[pl.Expr]:
    """Build the value-level rewrites that follow the renames.

    Args:
        plan: Resolved choices from :func:`_plan`.
        final_names: Column names the frame will carry after renames and drops.

    Returns:
        Expressions to apply in one ``with_columns``; empty when no choice needs one.
    """
    exprs: list[pl.Expr] = []
    for choice in plan:
        # A -log10(p) score must be un-logged when it lands on the numeric slot, or the
        # bands invert (a score of 8 means p = 1e-8, not p = 8.0 -> not_significant).
        if choice.rule.unlog and is_neglog10_column(choice.chosen):
            exprs.append(_unlog10(pl.col(choice.target).cast(pl.Float64, strict=False)).alias(choice.target))
        # Dispatched on the target rather than carried on the rule on purpose: this is a
        # CROSS-target Biolink class rule -- it reads whether another rule's target
        # survived -- so it is not a property of the effect-type rule itself, and giving
        # the registry a hook general enough to express it would buy nothing.
        if choice.target == "effect_type":
            exprs.append(_effect_type_expr(has_effect_size="effect_size" in final_names).alias("effect_type"))
    return exprs


def _apply(lf: pl.LazyFrame, rules: Sequence[_Rule]) -> pl.LazyFrame:
    """Run a set of coercion rules against a frame in a single pass.

    Args:
        lf: Source LazyFrame.
        rules: Rules to apply, in precedence order.

    Returns:
        LazyFrame with every claimed column renamed onto its canonical slot, losing
        aliases dropped where the rule says so, and the value-level rewrites applied.
        Returns ``lf`` untouched when nothing is claimed, so column order is preserved
        on the no-op path.

    Notes:
        One schema resolution drives the whole pass. The post-rename column set is
        derived from the plan rather than re-read from the frame, which is what lets
        the ``effect_type`` class rule see the renamed ``effect_size`` without a second
        ``collect_schema()``.
    """
    names: list[str] = lf.collect_schema().names()
    plan: list[_Choice] = _plan(names, rules)
    if not plan:
        return lf
    renames: dict[str, str] = {choice.chosen: choice.target for choice in plan if choice.chosen != choice.target}
    drops: list[str] = [alias for choice in plan for alias in choice.aliases if choice.rule.drop_aliases]
    dropped: frozenset[str] = frozenset(drops)
    final: frozenset[str] = frozenset(renames.get(n, n) for n in names if n not in dropped)
    exprs: list[pl.Expr] = _value_exprs(plan, final)
    if not renames and not drops and not exprs:
        return lf
    out: pl.LazyFrame = lf.rename(renames) if renames else lf
    if drops:
        out = out.drop(drops)
    return out.with_columns(exprs) if exprs else out


def coerce_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Apply every column coercion to a frame in one pass.

    The single clean-phase op ``Tcode._source_ops`` wires in. Equivalent to running
    :func:`coerce_pvalue_columns`, :func:`coerce_study_size_columns`,
    :func:`coerce_study_metadata_columns`, :func:`coerce_effect_size_columns` and
    :func:`coerce_effect_type_columns` in that order, which is exactly the precedence
    :func:`coerced_target` reports.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with every recognized statistical column renamed onto its canonical
        Biolink slot, losing aliases dropped, ``-log10`` sources un-logged and
        ``effect_type`` values coerced (no-op when nothing is claimed).

    Notes:
        The five coercions are one op rather than five because they share a phase
        label and a plan: resolving the schema once means the ``effect_type`` class
        rule can read the renamed ``effect_size`` from the plan instead of re-reading
        the frame. The per-coercion functions remain the unit of testing and of
        documentation, and are thin slices of this same code path.

    Warnings:
        Two Biolink class rules are enforced here. ``effect_type`` is nulled on every
        row where ``effect_size`` is null, and nulled entirely when no ``effect_size``
        column is present (PR #1774). A chosen ``-log10(p/q)`` column is un-logged as
        it lands on ``p_value``/``adjusted_p_value``, because the slot is typed as the
        probability, not the score.
    """
    return _apply(lf, _RULES)


def coerced_target(name: str) -> str:
    """Map a column/annotation name to the canonical name the clean phase renames it to.

    Args:
        name: Raw source column or annotation name.

    Returns:
        The canonical slot name a ``coerce_*_columns`` op would rename ``name``
        to, or ``name`` unchanged when no coercion claims it.

    Notes:
        Precedence is ``_RULES`` order -- the p-value rule runs first, so a
        p/q-value alias is claimed before the study-size, study-metadata and
        effect classifiers ever see it. That is the same tuple ``coerce_columns``
        applies, so what a config-time validator reports and what the build
        renames can no longer drift apart. Validators judge this target rather
        than the raw name, so they see a name exactly as the build will.
    """
    for rule in _RULES:
        target: str | None = rule.classify(name)
        if target is not None:
            return target
    return name


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
    return _apply(lf, (_EFFECT_SIZE_RULE,))


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
    ("logFC", "log2_fold_change"),  # limma/edgeR spelling; log-fold-change is base-2 by convention
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


def _effect_type_mapping(raws: Iterable[Any]) -> dict[Any, str]:
    """Resolve raw ``effect_type`` values onto canonical ``EffectTypes`` values.

    Args:
        raws: Raw cell values, usually the DISTINCT values of a column.

    Returns:
        Mapping from each resolvable raw value to its canonical value. Values
        that resolve to nothing -- null, blank, or below
        ``_EFFECT_TYPE_FUZZY_SCORE`` -- are simply absent, so a caller can hand
        the result to ``replace_strict(..., default=None)`` and get the
        drop-to-null the Biolink enum range requires.

    Notes:
        Exact/alias table first (case/separator-insensitive), then one
        ``rapidfuzz.process.extractOne`` per still-unresolved key against the 25
        canonical values. ``score_cutoff`` is inclusive, so the threshold is the
        same ``>=`` comparison this has always used.

        ``process.cdist`` would score every key against every value in one call,
        but it returns a numpy matrix and numpy is NOT a Tablassert dependency
        (it only appears in dev environments via the ``[qc]`` extra), so it would
        raise ``ModuleNotFoundError`` inside the polars UDF. ``extractOne`` is
        pure C and numpy-free.
    """
    from rapidfuzz import fuzz, process

    table, values = _effect_type_vocab()
    resolved: dict[Any, str] = {}
    # Distinct normalized keys are resolved once and reused: several raw spellings
    # ("Cohen's d", "cohens d", "COHENS_D") collapse onto one key, and the fuzzy
    # fallback is by far the most expensive step here.
    by_key: dict[str, str | None] = {}
    for raw in raws:
        if raw is None:
            continue
        key: str = _normalize_effect_type(str(raw).strip())
        if not key:
            continue
        if key not in by_key:
            hit: str | None = table.get(key)
            if hit is None:
                # Fuzzy fallback against the 25 canonical values only.
                found: tuple[str, float, int] | None = process.extractOne(key, values, scorer=fuzz.ratio, score_cutoff=_EFFECT_TYPE_FUZZY_SCORE)
                hit = found[0] if found is not None else None
            by_key[key] = hit
        canonical: str | None = by_key[key]
        if canonical is not None:
            resolved[raw] = canonical
    return resolved


def _map_effect_type_series(values: pl.Series) -> pl.Series:
    """Map a whole ``effect_type`` column onto canonical values.

    Args:
        values: Raw column as a string Series.

    Returns:
        Series of canonical ``EffectTypes`` values, null wherever the raw value
        matched nothing.

    Notes:
        Cost is bounded by the number of DISTINCT values, not by row count: the
        vocabulary is resolved once over ``unique()`` and the column is then
        rewritten by a single native ``replace_strict``. The previous per-row
        ``map_elements`` scaled with rows and, past its 4096-entry cache, spent
        roughly a second per 200k rows re-resolving values it had already seen.
    """
    text: pl.Series = values.cast(pl.String)
    # An empty mapping is fine: every value falls through to `default=None`.
    return text.replace_strict(_effect_type_mapping(text.unique().to_list()), default=None, return_dtype=pl.String)


def _map_effect_type_value(raw: Any) -> str | None:
    """Map one raw ``effect_type`` value to a canonical value, or null when nothing matches.

    Args:
        raw: Raw cell value (possibly null).

    Returns:
        The canonical ``EffectTypes`` value on an exact/alias hit or a fuzzy
        hit scoring at least ``_EFFECT_TYPE_FUZZY_SCORE``; ``None`` otherwise.

    Notes:
        Single-value convenience over :func:`_effect_type_mapping`, which is the
        one implementation. Deliberately uncached: the column path resolves each
        distinct value once already, and a process-global cache keyed on
        arbitrary source-cell values is state this module should not own.
    """
    if raw is None:
        return None
    return _effect_type_mapping((raw,)).get(raw)


def _effect_type_expr(*, has_effect_size: bool) -> pl.Expr:
    """Build the ``effect_type`` column expression: canonical values, then the class rule.

    Args:
        has_effect_size: Whether an ``effect_size`` column survives on the frame.

    Returns:
        One expression producing the final ``effect_type`` column.

    Notes:
        Values are matched case/separator-insensitively against an exact/alias table
        first, then by ``rapidfuzz`` fallback against the 25 canonical values. The
        Biolink range of ``effect_type`` is the enum, so values matching nothing are
        dropped to null rather than carried through.

    Warnings:
        Biolink class rule (PR #1774): ``effect_type`` may only be populated when
        ``effect_size`` is populated. The mapped value is therefore nulled on every
        row where ``effect_size`` is null, and nulled entirely when no ``effect_size``
        column is present -- the same shape of class rule ``sig`` documents for the
        significance qualifier.
    """
    if not has_effect_size:
        return pl.lit(None, dtype=pl.String)
    # `is_elementwise` stays at its default False on purpose: the whole column must arrive
    # in one call for the distinct-value resolution to see the full vocabulary at once.
    mapped: pl.Expr = pl.col("effect_type").cast(pl.String).map_batches(_map_effect_type_series, return_dtype=pl.String)
    populated: pl.Expr = pl.col("effect_size").cast(pl.Float64, strict=False).is_not_null()
    return pl.when(populated).then(mapped).otherwise(pl.lit(None, dtype=pl.String))


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
    return _apply(lf, (_EFFECT_TYPE_RULE,))
