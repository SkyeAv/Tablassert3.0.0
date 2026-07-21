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
    # ? Biolink Is_A Chain -- Leaf To Parent Role For Association Name Matching
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
    # ? Parses {Subject}To{Object}Association Into (Subject, [Object Roles])
    name = name.removesuffix("Association")
    if "To" not in name:
        return None
    subj: str = ""
    rest: str = ""
    subj, rest = name.split("To", 1)
    return (subj, rest.split("Or"))


@cache
def edge_tables() -> tuple[dict[str, str], dict[str, str]]:
    # ? Generates And Caches CATEGORY_ROLE And EDGE_LOOKUP On First Call

    # ? Flattened Leaf -> Root Role (Walks CATEGORY_PARENT Chain To Root)
    CATEGORY_ROLE: dict[str, str] = {}
    for leaf in CATEGORY_PARENT:
        role: str = leaf
        while role in CATEGORY_PARENT:
            role = CATEGORY_PARENT[role]
        CATEGORY_ROLE[leaf] = role

    # ? Auto-Generated (Subject Role, Object Role) -> EdgeCategories
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

    # ? Non-Standard Names -- Explicit Overrides
    EDGE_MAP[("ChemicalEntity", "Gene")] = EdgeCategories.CHEMICAL_GENE_INTERACTION

    # ? Flattened "subj_role|obj_role" -> biolink CURIE (For Polars replace_strict)
    EDGE_LOOKUP: dict[str, str] = {f"{s}|{o}": add("biolink:", ec.value) for (s, o), ec in EDGE_MAP.items()}

    return CATEGORY_ROLE, EDGE_LOOKUP


def edge_category(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ? Adds Derived Edge Category Column Using Native Polars Replace Operations
    cat_role: dict[str, str]
    edge_lookup: dict[str, str]
    cat_role, edge_lookup = edge_tables()
    sr: pl.Expr = pl.col("subject category").str.replace("biolink:", "").replace(cat_role).fill_null("")
    or_: pl.Expr = pl.col("object category").str.replace("biolink:", "").replace(cat_role).fill_null("")
    return lf.with_columns(
        pl.concat_list(
            pl.concat_str([sr, pl.lit("|"), or_]).replace_strict(edge_lookup, default=add("biolink:", EdgeCategories.ASSOCIATION.value))
        ).alias("category")
    )


def value(lf: pl.LazyFrame, col: str, x: str) -> pl.LazyFrame:
    # ? Creates A New Column With A Literal Value
    return lf.with_columns(pl.lit(x).alias(col))


def source_record_urls(lf: pl.LazyFrame, url: str) -> pl.LazyFrame:
    # ? Adds Biolink/Translator Source Record URLs As A List Column
    return lf.with_columns(pl.concat_list(pl.lit(url)).alias("source_record_urls"))


def publications(lf: pl.LazyFrame, curie: str) -> pl.LazyFrame:
    # ? Adds Publication CURIE As A Biolink-Compliant list[str] Column
    return lf.with_columns(pl.concat_list(pl.lit(curie)).alias("publications"))


def column(lf: pl.LazyFrame, col: str, x: str) -> pl.LazyFrame:
    # ? Creates A New Column With From An Old Column
    return lf.with_columns(pl.col(x).alias(col))


def math_op(lf: pl.LazyFrame, col: str, func: str, args: list[Union[Literal[Tokens.VALUES], float, int]]) -> pl.LazyFrame:
    # ? Transform Values In A Column With The Math Module
    # ! Collection Point: Required For map_elements
    # * strict=False tolerates residual non numeric junk in numeric annotation columns
    df: pl.DataFrame = lf.collect()
    expr: pl.Expr = pl.col(col).cast(pl.Float64, strict=False)
    attr: Callable[[Any], Any] = getattr(math, func)
    df = df.with_columns(expr.map_elements(lambda x: attr(*(x if eq(a, Tokens.VALUES) else a for a in args)), return_dtype=pl.Float64).alias(col))
    return df.lazy()


def numeric_columns(names: list[str]) -> list[str]:
    # ? Returns Column Names That Should Be Coerced And Formatted As Numbers
    # * P Value Columns By Substring Plus Exact Strength And Study Size Fields
    exact: set[str] = {"relationship_strength", "sample_size", "supporting_study_size"}
    return [c for c in names if ("p_value" in c.lower()) or (c in exact)]


def clean_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ? Coerces Numeric Annotation Columns To Float64 Dropping Non Numeric Values To Null
    # * Only Touches P Value Relationship Strength And Sample Size Columns
    cols: list[str] = numeric_columns(lf.collect_schema().names())
    if not cols:
        return lf
    return lf.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols])


def format_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ? Formats Numeric Annotation Columns As Strings With Controlled Notation
    # ! Collection Point: numpy Batch Formatting Required For Notation Control
    # * P Value Columns Use Scientific Notation Others Use Decimal General Format
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
    # ? Explodes A Row With Items Into Many Unique Rows By A Delimiter
    expr: pl.Expr = pl.col(col).cast(pl.String).str.split(delimiter)
    lf = lf.with_columns(expr.alias(col))
    return lf.explode(col)


def sig(lf: pl.LazyFrame, col: str = "p_value", out: str = "statistical_significance_qualifier") -> pl.LazyFrame:
    # ? Creates The "statistical_significance_qualifier" Column (Biolink PR #1766)
    # * Five-Band Cascade Matches StatisticalSignificanceQualifierEnum Verbatim;
    # * Boundaries Are Hardcoded Because The Enum Definitions Are Canonical
    # ! Edges Are Never Dropped: No P-Value Column -> Qualifier Absent; Null P-Value -> Null Qualifier
    from rapidfuzz import fuzz

    names: list[str] = lf.collect_schema().names()
    candidates: list[str] = [c for c in names if col in c]
    chosen: Optional[str] = max(candidates, key=lambda c: fuzz.ratio(c, col)) if candidates else None
    if chosen is None:
        # ! Biolink Class Rule: Qualifier May Only Be Set When p_value/adjusted_p_value Is Populated
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
    # ? Maps A Column Name To Its Canonical Biolink Compliant Target Name
    # * Word Boundary Anchored So "Group Value" Style Substrings Are Not Falsely Matched
    # * Bare "P" And "padj"/"p.adj" Cover Common GWAS/DESeq2 Conventions
    # * "Adj"/"Adjusted"/"Corrected" Only Count Alongside A P/Q Value Token Since They Are
    # * Generic Words Also Used For Adjusted Hazard/Odds Ratios, Unlike Fdr/Bonferroni/Holm
    # * "Significance"/"Significant" Columns Are Categorical Flags, Not The Numeric Value, So
    # * They Are Excluded Unless A P/Q Value Token Is Also Present
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
    # ? Renames P Value Like Columns To Biolink KGX Compliant p_value / adjusted_p_value
    # * Picks A Single Best Fuzzy Match Per Target When Multiple Candidates Exist
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
    # ? Maps Study Size Like Column Names To The Canonical Supporting Study Size Slot
    # * Bare "n" Is Allowed, But Other Matches Need Explicit Sample/Study Size Context
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
    # ? Renames Study Size Like Columns To Biolink KGX Compliant supporting_study_size
    # * Picks A Single Best Fuzzy Match And Leaves Other Candidate Columns Untouched
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
    # ? Creates A 1-Based Index Column Recording The Original Source Row
    # * Matches Pre-8.0.0 Behavior; Folded Into supporting_text By compile_graph
    return lf.with_row_index(col, offset=1)


def csv(p: Path, sep: str) -> pl.LazyFrame:
    # ? Reads Source From CSV And TSV As LazyFrame
    return pl.scan_csv(source=p, separator=sep, has_header=False, infer_schema_length=None, truncate_ragged_lines=True)


def excel(p: Path, sheet: str, engine: str = "calamine") -> pl.LazyFrame:
    # ? Reads Source From Excel As LazyFrame
    df: pl.DataFrame = pl.read_excel(
        source=p,
        sheet_name=sheet,
        engine=engine,  # pyright: ignore
        has_header=False,
        infer_schema_length=None,
    )
    return df.lazy()


def crop(lf: pl.LazyFrame, row_slice: list[Union[NonNegativeInt, Literal[Tokens.AUTO]]]) -> pl.LazyFrame:
    # ? Takes A Slice From A LazyFrame
    # ! Collection Point: Requires Height Calculation
    df: pl.DataFrame = lf.collect()
    n: int = df.select(pl.len()).item()
    start: Union[int, Literal[Tokens.AUTO]] = row_slice[0]
    stop: Union[int, Literal[Tokens.AUTO]] = row_slice[1]
    offset: int = 0 if eq(start, Tokens.AUTO) else start  # pyright: ignore
    length: int = n if eq(stop, Tokens.AUTO) else (stop - offset)  # pyright: ignore
    df = df.slice(offset=offset, length=length)
    return df.lazy()


def pick(lf: pl.LazyFrame, rows: list[int]) -> pl.LazyFrame:
    # ? Picks A List Of Rows From A LazyFrame
    # ! Collection Point: take() Requires Eager, Relazy After
    df: pl.DataFrame = lf.collect()
    df = df.select(pl.all().take(indices=rows))  # pyright: ignore
    return df.lazy()


def reindex(df: pl.LazyFrame, col: str, op: Callable, comp: Union[str, int, float], cast: bool = True) -> pl.LazyFrame:
    # ? Reindex A LazyFrame Based On A Condition
    expr: pl.Expr = pl.col(col).cast(pl.Float64) if cast else pl.col(col)
    return df.filter(op(expr, comp))


def drop_not_significant(lf: pl.LazyFrame, col: str = "statistical_significance_qualifier") -> pl.LazyFrame:
    # ? Release-Mode Row Filter: Drops Edges Whose Significance Qualifier Is biolink:not_significant
    # ! Only Filters When The Qualifier Column Exists; No-Op For Sections Without A p_value Column
    # ! ne_missing Keeps Null Qualifiers (Null P-Value -> Null Qualifier -> Kept, Not Dropped)
    names: list[str] = lf.collect_schema().names()
    if col not in names:
        return lf
    return lf.filter(pl.col(col).cast(pl.String).ne_missing("biolink:not_significant"))


def idxname(col: Any) -> str:
    # ? Converts Excel Style Column Names To Polars Column Names
    scol: str = str(col)
    idx: int = 0
    for char in scol:
        idx = idx * 26 + (ord(char) - 65 + 1)

    return f"column_{idx}"


def trim(lf: pl.LazyFrame, regex: str = r"^column_\d+$") -> pl.LazyFrame:
    # ? Removes Columns With The Excel Naming Conventions From LazyFrame
    return lf.select(pl.exclude(regex))


def to_store(lf: pl.LazyFrame, p: Path, config_name: str) -> Path:
    # ? collect and write section parquet; warn if result is empty
    df: pl.DataFrame = lf.collect()

    if df.height == 0:
        logger.warning("Section {config} ({hash}) produced 0 rows", config=config_name, hash=p.stem)
    df.write_parquet(p)

    return p


class Tcode(Section):
    # ? Extends Section To Compile A KG
    config: Path = Field(...)
    store: Path = Field(...)
    log: bool = Field(False)
    qc: bool = Field(False)
    release: bool = Field(False)
    name: Optional[str] = Field(None)

    def encoding(self: Self, x: Encoding, col: str, table_literal: bool = False) -> list[Any]:
        # ? Collect Helper For Encoding Classes
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
        # ? Collect Helper For NodeEncoding Classes -- Encoding Plus NLP Normalization, Before resolve_batch
        encoding: list[Any] = self.encoding(x, col, table_literal=True)
        prep: list[Any] = [(column, (add(col, "_pre_resolution"), col)), (level_one, (col,)), (level_two, (col,))]
        return add(encoding, prep)

    def clean(self: Self, tcode: list[tuple[Callable, Any]]) -> list[tuple[Callable, tuple[Any]]]:
        # ? Cleans Tcode So It Can Be Used With reduce From functools
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
        # ? Code That Tells Tablassert What Actions To While Transforming Data

        if self.store.is_file():
            # * Quick Exit If Subgraph Already Exists
            return self.store

        else:
            # * Subject/Object/Qualifiers Share One resolve_batch Call Instead Of One Per Column
            node_columns: list[tuple[NodeEncoding, str]] = [
                (self.statement.subject, "subject"),
                (self.statement.object, "object"),
                *[(x, x.qualifier) for x in (self.statement.qualifiers or [])],
            ]
            specs: list[ResolveSpec] = [ResolveSpec(col, str(x.taxon) if x.taxon else None, x.prioritize, x.avoid) for x, col in node_columns]

            # * Returns A List Of: (Function, (Arguments))
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
                # * Drop Insignificant Rows Before They Ever Reach The Expensive Fullmap Resolution Below
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
    # ? Maps Each Tcode Op Callable To A Short Lowercase Phase Label For Progress UX
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
    # ? Resolves The Phase Label For An Op, Handling value() Specially Since It Spans Phases
    if fn is value:
        col: str = str(args[0]) if args else ""
        if col == "predicate":
            return "edge"
        if col in _VALUE_PROVENANCE_COLS:
            return "provenance"
        return UNKNOWN_PHASE
    return PHASE_OF.get(fn, UNKNOWN_PHASE)


def compile_subgraph(tcode: list[tuple[Callable, tuple[Any]]], *, on_phase: Optional[Callable[[str], None]] = None) -> Path:
    # ? Executes Tcode To Build Subgraphs As Parquets; on_phase Fires When The Phase Label Changes
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
    # ? Normalized Disparate Node Columns To A Unified Format And Removes Them From Edges
    # * Returns Partial Nodes And Modified Edges As LazyFrames
    cols: list[str] = [col, add(col, "_name"), add(col, "_category"), add(col, "_taxon"), add(col, "_source"), add(col, "_source_version")]
    nodes: pl.LazyFrame = edges.select(cols).unique().rename({k: v for k, v in zip(cols, names)})
    # ? Ensures Category Has biolink: Prefix
    nodes = nodes.with_columns(
        pl.when(pl.col("category").str.starts_with("biolink:"))
        .then(pl.col("category"))
        .otherwise(add(pl.lit("biolink:"), pl.col("category")))
        .alias("category")
    )
    # ? Exports Category Within A List (Null Categories Stay Null For strip_nulls)
    nodes = nodes.with_columns(pl.when(pl.col("category").is_not_null()).then(pl.concat_list(pl.col("category"))).alias("category"))
    edges_out: pl.LazyFrame = edges.drop(cols[1:])
    return nodes, edges_out


def publication_curie(repo: str, publication: str) -> str:
    # ? Builds The Publication CURIE; PMCID Namespace For PubMed Central
    if eq(repo, Repositories.PUBMED_CENTRAL):
        return add("PMCID:", publication)
    return add(repo, add(":", publication))


def infores(name: str) -> str:
    # ? Builds An infores CURIE From A Graph Name In Lower Kebab Case
    return add("infores:", name.lower().replace("_", "-"))


def upstream_resource_ids(repo: Repositories) -> list[str]:
    # ? Maps Publication Repository To Translator InfoRes Upstream Source IDs
    if eq(repo, Repositories.PUBMED_CENTRAL):
        return [InformationResources.PUBMED_CENTRAL.value]
    return [InformationResources.PUBMED.value]


def strip_nulls(r: object, bad: set[str] = {"na", "nan", "null", "none", ""}) -> dict:
    # ? Removes Null Keys From NDJSON
    return {
        k: [strip_nulls(i) if isinstance(i, dict) else i for i in v] if isinstance(v, list) else strip_nulls(v) if isinstance(v, dict) else v
        for k, v in r.items()  # pyright: ignore
        if v and str(v).strip().lower() not in bad
    }


def as_list(v: object) -> list[object]:
    # ? Coerces Scalar And List-Like Values Into A Plain List
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def normalize_biolink_category(v: object) -> Optional[str]:
    # ? Normalizes Category Strings For RIG Target Summaries
    if not isinstance(v, str) or not v:
        return None
    if v.startswith("biolink:"):
        return v
    return add("biolink:", v)


def curie_prefix(v: object) -> Optional[str]:
    # ? Extracts Compact Identifier Prefixes For RIG Node Type Summaries
    if not isinstance(v, str) or ":" not in v:
        return None
    prefix: str = v.split(":", 1)[0]
    return prefix or None


def clean_values(values: list[object]) -> list[str]:
    # ? Removes Empty Values And Deduplicates Stringified RIG Summary Values
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
    # ? Summarizes Raw Edge Columns Into RIG Edge Type Metadata Before Node Normalization
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
    # ? Summarizes Normalized Nodes Into RIG Node Type Metadata
    buckets: dict[str, set[str]] = {}
    for node in nodes:
        prefixes: list[str] = clean_values([curie_prefix(node.get("id"))])
        for category in clean_values([normalize_biolink_category(v) for v in as_list(node.get("category"))]):
            buckets.setdefault(category, set()).update(prefixes)

    return [strip_nulls({"node_category": category, "source_identifier_types": sorted(prefixes)}) for category, prefixes in sorted(buckets.items())]


def unique_dicts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    # ? Deduplicates Small RIG Summary Dictionaries Without Adding A New Dependency
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
    # ? Writes Translator Resource Ingest Guide Metadata Alongside KGX Outputs
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
    # ? Removes Null Values From And Deduplicates NDJSON
    # * Also Adds UUIDs To Edges
    p_out: Path = p_in.with_suffix("")

    if p_out.is_file():
        p_out.unlink()

    rs.dedup_ndjson(p_in, p_out, is_edges, "TABLASSERT")

    p_in.unlink()


def fold_unknown_to_supporting_text(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ? Folds Any Non-Biolink Edge Column Into supporting_text As "col: value" Strings
    # * Stays Fully Lazy; Null/Blank Values Produce No Entry; Sorted For Deterministic Output
    # * Existing list[str] supporting_text Has Derived Entries Appended (Never Clobbered);
    # * Scalar supporting_text (E.G. A method: value Annotation) Is Coerced To list[str] First
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
        # ? Coerce Scalar supporting_text To list[str] First, Then Append Derived Entries
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
    # ? Aggregates Parquets For NDJSON KGX Export Using Lazy Scan
    p: Path = Path(f"./{name}_{version}.tmp")

    e: Path = p.with_suffix(".edges.ndjson.tmp")  # ! For Labeling
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

        # ? Only subject and object become nodes; qualifier columns stay as edge attributes
        originals: list[str] = [c.removesuffix("_pre_resolution") for c in lf.collect_schema().names() if c.endswith("_pre_resolution")]
        node_cols: list[str] = [c for c in originals if c in ("subject", "object")]
        for col in node_cols:
            partial, lf = normalize(lf, col)
            subnodes.append(partial)
        # ? Drop Internal Pre-Resolution Snapshot Columns From Final Edges
        lf = lf.drop([c for c in lf.collect_schema().names() if c.endswith("_pre_resolution")])
        lf = fold_unknown_to_supporting_text(lf)
        subedges.append(lf)

    # ! Collection Point: Appending To Output Files
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
    # ? Drop Internal Pre-Resolution Snapshot Columns To Mirror Edge Output
    df = df.drop([c for c in df.columns if c.endswith("_pre_resolution")])
    return df.to_dicts()
