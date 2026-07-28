from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, NonNegativeInt

from tablassert import rs
from tablassert._lazy import LazyModule
from tablassert.biolink import ALLOWED_EDGE_FIELDS, Categories, EdgeCategories
from tablassert.coerce import coerce_pvalue_columns, coerce_study_size_columns, pvalue_target, sig, study_size_target
from tablassert.enums import EncodingMethods, Files, InformationResources, Repositories, Tokens
from tablassert.fullmap import ResolveSpec, fullmap_db_path, resolve, resolve_batch
from tablassert.log import cat
from tablassert.models import Encoding, NodeEncoding, Section
from tablassert.nlp import level_one, level_two
from tablassert.qc import fullmap_audit
from tablassert.rig import (
    as_list,
    clean_values,
    compile_rig,
    curie_prefix,
    infores,
    normalize_biolink_category,
    rig_edge_type_info,
    rig_node_type_info,
    strip_nulls,
    unique_dicts,
)

if TYPE_CHECKING:
    import polars as pl
else:
    pl = LazyModule("polars")

logger = cat("PIPELINE")

__all__ = [
    "as_list",
    "clean_values",
    "coerce_pvalue_columns",
    "coerce_study_size_columns",
    "compile_rig",
    "curie_prefix",
    "infores",
    "normalize_biolink_category",
    "pvalue_target",
    "rig_edge_type_info",
    "rig_node_type_info",
    "sig",
    "strip_nulls",
    "study_size_target",
    "unique_dicts",
]


CATEGORY_PARENT: dict[str, str] = {
    # Curated leaf -> parent *role* map used to roll concrete categories up to the
    # role names that appear in Biolink association class names (e.g. Protein -> Gene
    # so a Protein subject matches GeneToDiseaseAssociation). This is a semantic
    # roll-up, NOT a literal Biolink is_a step -- in the model `protein` is not is_a
    # `gene`, `haplotype` is not is_a `genotype`, and `Variant` is an association-role
    # name rather than a class -- so it cannot be auto-derived from the model and is
    # maintained here. All keys/values are nonetheless valid Biolink categories.
    "SmallMolecule": "MolecularEntity",
    "MolecularEntity": "ChemicalEntity",
    "Drug": "MolecularMixture",
    "MolecularMixture": "ChemicalMixture",
    "ChemicalMixture": "ChemicalEntity",
    "Protein": "Gene",
    "SequenceVariant": "Variant",
    "Haplotype": "Genotype",
}


def parse_edge_name(name: str) -> tuple[str, list[str]] | None:
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
            parsed: tuple[str, list[str]] | None = parse_edge_name(ec.value)
            if parsed is not None:
                subj: str = ""
                objs: list[str] = []
                subj, objs = parsed
                for obj in objs:
                    EDGE_MAP[(subj, obj)] = ec

    # Non-standard names -- explicit overrides.
    EDGE_MAP[("ChemicalEntity", "Gene")] = EdgeCategories.CHEMICAL_GENE_INTERACTION_ASSOCIATION

    # Flattened "subj_role|obj_role" -> biolink CURIE (for polars replace_strict).
    EDGE_LOOKUP: dict[str, str] = {f"{s}|{o}": f"biolink:{ec.value}" for (s, o), ec in EDGE_MAP.items()}

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
            pl.concat_str([sr, pl.lit("|"), or_]).replace_strict(edge_lookup, default=f"biolink:{EdgeCategories.ASSOCIATION.value}")
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


def math_op(lf: pl.LazyFrame, col: str, func: str, args: list[Literal[Tokens.VALUES] | float | int]) -> pl.LazyFrame:
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
    df = df.with_columns(expr.map_elements(lambda x: attr(*(x if a == Tokens.VALUES else a for a in args)), return_dtype=pl.Float64).alias(col))
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
        Collection point: the column is formatted in one batch so notation is
        controlled across all rows at once.
    """
    # Collection point: batch formatting for notation control.
    # P-value columns use scientific notation; others use decimal general format.
    df: pl.DataFrame = lf.collect()
    cols: list[str] = numeric_columns(df.columns)
    for c in cols:
        df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        fmt: str = "{:.4e}" if "p_value" in c.lower() else "{:.4g}"
        formatted: list[str | None] = [None if v is None else fmt.format(v) for v in df[c].to_list()]
        df = df.with_columns(pl.Series(c, formatted))
    return df.lazy()


def prefix(lf: pl.LazyFrame, col: str, prefix: str) -> pl.LazyFrame:
    expr: pl.Expr = pl.lit(prefix) + pl.col(col).cast(pl.String)
    return lf.with_columns(expr.alias(col))


def suffix(lf: pl.LazyFrame, col: str, suffix: str) -> pl.LazyFrame:
    expr: pl.Expr = pl.col(col).cast(pl.String) + pl.lit(suffix)
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


def crop(lf: pl.LazyFrame, row_slice: list[NonNegativeInt | Literal[Tokens.AUTO]]) -> pl.LazyFrame:
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
    start: int | Literal[Tokens.AUTO] = row_slice[0]
    stop: int | Literal[Tokens.AUTO] = row_slice[1]
    offset: int = 0 if start == Tokens.AUTO else start  # pyright: ignore
    length: int = n if stop == Tokens.AUTO else (stop - offset)  # pyright: ignore
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


HEAD_ROWS: int = 5


def head(lf: pl.LazyFrame, n: int = HEAD_ROWS) -> pl.LazyFrame:
    """Limit a LazyFrame to its first ``min(n, height)`` rows for a fast ``--head`` preview.

    Args:
        lf: Source LazyFrame.
        n: Maximum number of rows to keep.

    Returns:
        LazyFrame with at most ``n`` rows (fewer when the source has fewer than ``n``).

    Notes:
        Stays fully lazy; ``LazyFrame.head`` pushes the limit down to the scan,
        so only ``n`` rows are ever materialized downstream.
    """
    return lf.head(n)


def reindex(df: pl.LazyFrame, col: str, op: Callable, comp: str | int | float, cast: bool = True) -> pl.LazyFrame:
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
    head: bool = Field(False)
    name: str | None = Field(None)

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
            (value, (col, x.encoding)) if x.method == EncodingMethods.VALUE else None,
            (column, (col, idxname(x.encoding))) if x.method == EncodingMethods.COLUMN else None,
            (column, (f"original_{col}", col)) if table_literal else None,
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
        prep: list[Any] = [(column, (f"{col}_pre_resolution", col)), (level_one, (col,)), (level_two, (col,))]
        return encoding + prep

    def clean(self: Self, tcode: list[tuple[Callable, Any]]) -> list[tuple[Callable, tuple[Any]]]:
        """Clean a Tcode list so it can be used with ``reduce`` from functools.

        Args:
            tcode: Raw list possibly containing ``None`` placeholders and
                nested lists of tuples.

        Returns:
            Flat list of ``(callable, tuple[Any, ...])`` pairs with falsy
            entries removed.
        """
        return [op for x in tcode if x for op in (self.clean(x) if isinstance(x, list) else [x])]

    def _source_ops(self: Self) -> list[Any]:
        """Collect the load/filter/clean/significance ops that precede entity resolution.

        Returns:
            Raw (possibly nested, possibly ``None``-padded) op list ready for
            ``clean``; order matches the pre-resolution pipeline exactly.
        """
        return [
            (csv, (self.source.local, self.source.delimiter)) if self.source.kind == Files.TEXT else None,  # pyright: ignore
            (excel, (self.source.local, self.source.sheet)) if self.source.kind == Files.EXCEL else None,  # pyright: ignore
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
            # --head preview: cap rows to min(HEAD_ROWS, height) before any encoding/resolve.
            (head, (HEAD_ROWS,)) if self.head else None,
            [op for x in self.annotations for op in self.encoding(x, x.annotation.lower())] if self.annotations else None,
            (coerce_pvalue_columns, ()),
            (coerce_study_size_columns, ()),
            (clean_numeric, ()),
            # Drop insignificant rows before they ever reach the expensive fullmap resolution below.
            (sig, ()),
            (drop_not_significant, ()) if self.release else None,
        ]

    def _node_ops(self: Self, db: Path) -> list[Any]:
        """Collect the per-node encoding/resolve/QC ops shared across subject/object/qualifiers.

        Args:
            db: Path to the fullmap redb used for entity resolution.

        Returns:
            Raw op list: one ``node_prep`` block per node column, the single
            shared ``resolve_batch`` op, then per-column ``fullmap_audit`` ops
            when QC is enabled.
        """
        # Subject/object/qualifiers share one resolve_batch call instead of one per column.
        node_columns: list[tuple[NodeEncoding, str]] = [
            (self.statement.subject, "subject"),
            (self.statement.object, "object"),
            *[(x, x.qualifier) for x in (self.statement.qualifiers or [])],
        ]
        specs: list[ResolveSpec] = [
            ResolveSpec(col, str(x.taxon) if x.taxon else None, x.prioritize, x.avoid, x.exclude_prefixes, x.exclude_regex) for x, col in node_columns
        ]
        return [
            [self.node_prep(x, col) for x, col in node_columns],
            (resolve_batch, (specs, db, self.log, self.store.stem, self.config.name, True)),
            [(fullmap_audit, (col, self.store.stem, self.config.name, "passed", True)) for _, col in node_columns] if self.qc else None,
        ]

    def _provenance_ops(self: Self) -> list[Any]:
        """Collect the edge/provenance/finalize ops that follow entity resolution.

        Returns:
            Raw op list: predicate and edge category, provenance metadata,
            then the trim/format/write finalize ops.
        """
        return [
            (value, ("predicate", "biolink:" + self.statement.predicate)),
            (edge_category, ()),
            (value, ("upstream_resource_ids", upstream_resource_ids(self.provenance.repo))),
            (value, ("knowledge_level", self.provenance.knowledge_level)),
            (value, ("agent_type", self.provenance.agent_type)),
            (value, ("resource_id", infores(self.name))) if self.name else None,
            (publications, (publication_curie(self.provenance.repo, self.provenance.publication),)),
            (source_record_urls, (str(self.source.url),)),
            (value, ("sheet_name", self.source.sheet)) if self.source.kind == Files.EXCEL else None,  # pyright: ignore
            (trim, ()),
            (format_numeric, ()),
            (to_store, (self.store, self.config.name)),
        ]

    def collect(self: Self, db: Path) -> list[tuple[Callable, tuple[Any]]] | Path:
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

        # Returns a list of: (function, (arguments)).
        tcode: list[Any] = [*self._source_ops(), *self._node_ops(db), *self._provenance_ops()]
        return self.clean(tcode)


PHASE_OF: dict[Callable, str] = {
    # Maps each Tcode op callable to a short lowercase phase label for progress UX.
    csv: "load",
    excel: "load",
    idx: "load",
    crop: "filter",
    pick: "filter",
    reindex: "filter",
    head: "filter",
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

# Ops that accept an ``on_phase`` callback so ``compile_subgraph`` can forward fine-grained sub-phases.
PHASE_AWARE: frozenset[Callable] = frozenset({resolve_batch, fullmap_audit})

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


def compile_subgraph(tcode: list[tuple[Callable, tuple[Any]]], *, on_phase: Callable[[str], None] | None = None) -> Path:
    """Execute a Tcode operation list to build a subgraph parquet.

    Args:
        tcode: Cleaned list of ``(callable, args)`` tuples from ``Tcode.collect``.
        on_phase: Optional callback fired when the phase label changes, and
            forwarded into phase-aware ops (``resolve_batch``/``fullmap_audit``)
            so they can emit fine-grained sub-phases; used to drive progress UX.

    Returns:
        Path to the written subgraph parquet.
    """
    last_phase: str | None = None
    acc: pl.LazyFrame | Path | None = None
    for op in tcode:
        fn: Callable = op[0]
        args: tuple[Any, ...] = op[1]
        if on_phase is not None:
            phase: str = _phase_of(fn, args)
            if phase != last_phase:
                last_phase = phase
                on_phase(phase)
        if fn in PHASE_AWARE and on_phase is not None:
            acc = fn(acc, *args, on_phase=on_phase) if acc is not None else fn(*args, on_phase=on_phase)  # pyright: ignore
        else:
            acc = fn(acc, *args) if acc is not None else fn(*args)  # pyright: ignore
    return acc  # pyright: ignore


def normalize(edges: pl.LazyFrame, col: str, names: list[str] | None = None) -> tuple[pl.LazyFrame, pl.LazyFrame]:
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
    if names is None:
        names = ["id", "name", "category", "taxon", "source", "source_version"]
    cols: list[str] = [col, f"{col}_name", f"{col}_category", f"{col}_taxon", f"{col}_source", f"{col}_source_version"]
    nodes: pl.LazyFrame = edges.select(cols).unique().rename(dict(zip(cols, names, strict=True)))
    # Ensures category has biolink: prefix.
    nodes = nodes.with_columns(
        pl.when(pl.col("category").str.starts_with("biolink:"))
        .then(pl.col("category"))
        .otherwise(pl.lit("biolink:") + pl.col("category"))
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
    if repo == Repositories.PUBMED_CENTRAL:
        return "PMCID:" + publication
    return repo + ":" + publication


def upstream_resource_ids(repo: Repositories) -> list[str]:
    """Map a publication repository to Translator infores upstream source IDs.

    Args:
        repo: Repository enum value.

    Returns:
        Single-element list containing the matching infores identifier.
    """
    if repo == Repositories.PUBMED_CENTRAL:
        return [InformationResources.PUBMED_CENTRAL.value]
    return [InformationResources.PUBMED.value]


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
        existing: pl.Expr = pl.col("supporting_text") if isinstance(schema["supporting_text"], pl.List) else pl.concat_list(pl.col("supporting_text"))
        combined: pl.Expr = existing.list.concat(derived).list.drop_nulls()
    else:
        combined = derived

    return lf.with_columns(combined.alias("supporting_text")).drop(unknown)


def _collect_subframes(
    subgraphs: list[Path],
    edges_tmp: Path,
    ui_explanation: str | None,
    on_phase: Callable[[str], None] | None = None,
    on_subgraph: Callable[[], None] | None = None,
) -> tuple[list[pl.LazyFrame], list[pl.LazyFrame], list[dict[str, object]]]:
    """Scan and normalize subgraph parquets into node/edge subframes.

    Covers the ``scan`` and ``normalize`` phases of ``compile_graph``; the
    phase boundaries inside the loop are the hook points for the US-009
    ``on_phase`` progress callback.

    Args:
        subgraphs: Section parquet paths to merge.
        edges_tmp: Working ``.edges.ndjson.tmp`` path (its de-suffixed name is
            recorded under edge type info ``source_files``).
        ui_explanation: Optional UI explanation for the RIG edge type info.
        on_phase: Optional callback fired with ``"scan"`` then ``"normalize"``
            for each subgraph, used to drive progress UX.
        on_subgraph: Optional callback fired once after each subgraph is
            processed, used to tick the progress bar.

    Returns:
        Tuple of ``(subnodes, subedges, edge_type_info)``: per-section node and
        edge LazyFrames plus the accumulated RIG edge type summaries.
    """
    subnodes: list[pl.LazyFrame] = []
    subedges: list[pl.LazyFrame] = []
    edge_type_info: list[dict[str, object]] = []
    for s in subgraphs:
        # Phase: scan.
        if on_phase is not None:
            on_phase("scan")
        lf: pl.LazyFrame = pl.scan_parquet(s)
        edge_type_info.extend(rig_edge_type_info(lf, edges_tmp.with_suffix(""), ui_explanation))

        # Phase: normalize. Only subject and object become nodes; qualifier columns stay as edge attributes.
        if on_phase is not None:
            on_phase("normalize")
        originals: list[str] = [c.removesuffix("_pre_resolution") for c in lf.collect_schema().names() if c.endswith("_pre_resolution")]
        node_cols: list[str] = [c for c in originals if c in ("subject", "object")]
        for col in node_cols:
            partial, lf = normalize(lf, col)
            subnodes.append(partial)
        # Drop internal pre-resolution snapshot columns from final edges.
        lf = lf.drop([c for c in lf.collect_schema().names() if c.endswith("_pre_resolution")])
        lf = fold_unknown_to_supporting_text(lf)
        subedges.append(lf)
        if on_subgraph is not None:
            on_subgraph()
    return subnodes, subedges, edge_type_info


def _write_ndjson(
    subnodes: list[pl.LazyFrame],
    subedges: list[pl.LazyFrame],
    edge_type_info: list[dict[str, object]],
    nodes_tmp: Path,
    edges_tmp: Path,
    name: str,
    version: str,
    description: str | None,
    contributions: list[str] | None,
    ui_explanation: str | None,
    tables: list[Path] | None,
    on_phase: Callable[[str], None] | None = None,
) -> None:
    """Write, dedup, and RIG the KGX NDJSON outputs.

    Covers the ``write-nodes``, ``write-edges``, ``dedup`` and ``rig`` phases of
    ``compile_graph``; each commented phase boundary below is a hook point for
    the US-009 ``on_phase`` progress callback.

    Args:
        subnodes: Per-section node LazyFrames from ``_collect_subframes``.
        subedges: Per-section edge LazyFrames from ``_collect_subframes``.
        edge_type_info: Accumulated RIG edge type summaries.
        nodes_tmp: Working ``.nodes.ndjson.tmp`` output path.
        edges_tmp: Working ``.edges.ndjson.tmp`` output path.
        name: Graph name (output stems and RIG).
        version: Graph version string.
        description: Optional RIG description.
        contributions: Optional RIG contributor list.
        ui_explanation: Optional RIG UI explanation.
        tables: Optional RIG source table list.
        on_phase: Optional callback fired with ``"write-nodes"``,
            ``"write-edges"``, ``"dedup"`` and ``"rig"`` at each phase
            boundary, used to drive progress UX.
    """
    # Phase: write-nodes. Collection point: appending to output files.
    if on_phase is not None:
        on_phase("write-nodes")
    node_rows: list[dict[str, object]] = []
    with nodes_tmp.open("a") as f:
        for subnode in subnodes:
            eagernode: pl.DataFrame = subnode.collect().unique()
            node_rows.extend(eagernode.to_dicts())
            eagernode.write_ndjson(f)

    # Phase: write-edges.
    if on_phase is not None:
        on_phase("write-edges")
    with edges_tmp.open("a") as f:
        for subedge in subedges:
            eageredge: pl.DataFrame = subedge.collect().unique()
            eageredge.write_ndjson(f)

    # Phase: dedup.
    if on_phase is not None:
        on_phase("dedup")
    dedup_stream(edges_tmp, is_edges=True)
    dedup_stream(nodes_tmp, is_edges=False)

    # Phase: rig.
    if on_phase is not None:
        on_phase("rig")
    compile_rig(
        name,
        version,
        description,
        contributions,
        ui_explanation,
        tables,
        nodes_tmp.with_suffix(""),
        edges_tmp.with_suffix(""),
        rig_node_type_info(node_rows),
        unique_dicts(edge_type_info),
    )


def compile_graph(
    subgraphs: list[Path],
    name: str,
    version: str,
    description: str | None = None,
    contributions: list[str] | None = None,
    ui_explanation: str | None = None,
    tables: list[Path] | None = None,
    on_phase: Callable[[str], None] | None = None,
    on_subgraph: Callable[[], None] | None = None,
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
        on_phase: Optional callback fired with the current phase label
            (``scan`` / ``normalize`` per subgraph, then ``write-nodes`` /
            ``write-edges`` / ``dedup`` / ``rig``), used to drive progress UX.
        on_subgraph: Optional callback fired once per processed subgraph,
            used to tick the progress bar.

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

    subnodes: list[pl.LazyFrame]
    subedges: list[pl.LazyFrame]
    edge_type_info: list[dict[str, object]]
    subnodes, subedges, edge_type_info = _collect_subframes(subgraphs, e, ui_explanation, on_phase, on_subgraph)
    _write_ndjson(subnodes, subedges, edge_type_info, n, e, name, version, description, contributions, ui_explanation, tables, on_phase)


def resolve_many(
    col: str,
    entities: Iterable[str],
    fullmap: Path,
    taxon: str | None = None,
    prioritize: list[Categories] | None = None,
    avoid: list[Categories] | None = None,
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

    lf = column(lf, f"original_{col}", col)
    lf = column(lf, f"{col}_pre_resolution", col)
    lf = level_one(lf, col)
    lf = level_two(lf, col)

    lf = resolve(lf, col, fullmap_db_path(fullmap), taxon=taxon, prioritize=prioritize, avoid=avoid, column_context=column_context)
    if qc:
        lf = fullmap_audit(lf, col, "", "", log=qc)

    df: pl.DataFrame = lf.collect()
    # Drop internal pre-resolution snapshot columns to mirror edge output.
    df = df.drop([c for c in df.columns if c.endswith("_pre_resolution")])
    return df.to_dicts()
