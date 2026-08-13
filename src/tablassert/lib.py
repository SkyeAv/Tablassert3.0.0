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
from tablassert.biolink import (
    ALLOWED_EDGE_FIELDS,
    ENUM_RANGED_QUALIFIERS,
    UNSATISFIABLE_EDGE_FIELDS,
    Categories,
    EdgeCategories,
    association_class,
    class_fields,
    is_multivalued,
    legal_predicates,
    numeric_slot_kind,
    resolve_association_class,
    resolve_node_category,
)
from tablassert.coerce import (
    coerce_effect_size_columns,
    coerce_effect_type_columns,
    coerce_pvalue_columns,
    coerce_study_size_columns,
    effect_size_target,
    effect_type_target,
    pvalue_target,
    sig,
    study_size_target,
)
from tablassert.enums import EncodingMethods, Files, InformationResources, Repositories, Tokens
from tablassert.fullmap import ResolveSpec, fullmap_db_path, resolve, resolve_batch
from tablassert.log import cat
from tablassert.models import Encoding, NodeEncoding, Qualifier, Section
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
    "coerce_effect_size_columns",
    "coerce_effect_type_columns",
    "coerce_pvalue_columns",
    "coerce_study_size_columns",
    "compile_rig",
    "curie_prefix",
    "effect_size_target",
    "effect_type_target",
    "infores",
    "normalize_biolink_category",
    "predicate_options",
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


@cache
def derived_edge_category(subject_category: str, object_category: str) -> str:
    """Return the edge category the build derives for a (subject, object) category pair.

    The pure-Python twin of the ``(subject role, object role)`` lookup :func:`edge_category`
    performs inside a LazyFrame, so an authoring-time caller can ask what class a statement
    would land in without building anything.

    Args:
        subject_category: Subject category, with or without the ``biolink:`` prefix.
        object_category: Object category, with or without the ``biolink:`` prefix.

    Returns:
        The unresolved edge category CURIE, ``"biolink:Association"`` when the pair has no
        specific mapping.
    """
    cat_role: dict[str, str]
    edge_lookup: dict[str, str]
    cat_role, edge_lookup = edge_tables()
    subject: str = subject_category.removeprefix("biolink:")
    obj: str = object_category.removeprefix("biolink:")
    key: str = f"{cat_role.get(subject, subject)}|{cat_role.get(obj, obj)}"
    return edge_lookup.get(key, f"biolink:{EdgeCategories.ASSOCIATION.value}")


def predicate_options(subject_category: str, object_category: str) -> frozenset[str] | None:
    """Return the predicates a (subject, object) category pair may carry without demotion.

    Composes :func:`derived_edge_category` with :func:`biolink.legal_predicates` to answer
    the question config authors (and the agent) actually have: *which predicate keeps this
    edge's specific association class?* A predicate outside this set is not an error -- it
    silently costs the edge its class via :func:`biolink.resolve_association_class`, taking
    every qualifier and evidence slot that class declared with it.

    Args:
        subject_category: Subject category, with or without the ``biolink:`` prefix.
        object_category: Object category, with or without the ``biolink:`` prefix.

    Returns:
        The permitted predicate CURIEs, or ``None`` when the pair derives an association
        class with an open ``predicate`` slot (anything is legal, nothing is specific).
    """
    return legal_predicates(derived_edge_category(subject_category, object_category))


def edge_category(lf: pl.LazyFrame, predicate: str | None = None) -> pl.LazyFrame:
    """Add the derived ``category`` column using native polars replace operations.

    The ``(subject role, object role)`` lookup alone routinely produces a category
    that contradicts the predicate: ``GeneToDiseaseAssociation`` restricts its
    ``predicate`` slot to ``contributes_to|associated_with|affects``, so a
    ``biolink:gene_associated_with_condition`` edge labelled with that category can
    never validate. When ``predicate`` is supplied the raw category is post-resolved
    by :func:`biolink.resolve_association_class`, which walks up the association
    hierarchy only as far as the predicate requires.

    Args:
        lf: Source LazyFrame with ``subject category`` and ``object category``
            columns (biolink-prefixed).
        predicate: Section predicate CURIE used to reconcile the derived category.

    Returns:
        LazyFrame with a new list-typed ``category`` column containing the
        resolved biolink edge-category CURIE (falling back to
        ``biolink:Association`` when no specific mapping matches).
    """
    cat_role: dict[str, str]
    edge_lookup: dict[str, str]
    cat_role, edge_lookup = edge_tables()
    default: str = f"biolink:{EdgeCategories.ASSOCIATION.value}"
    if predicate:
        # The predicate is a section constant, so the reconciliation collapses to a
        # small raw-category -> resolved-category remap resolved once at plan time.
        edge_lookup = {k: f"biolink:{resolve_association_class(v, predicate).__name__}" for k, v in edge_lookup.items()}
        default = f"biolink:{resolve_association_class(default, predicate).__name__}"
    names: list[str] = lf.collect_schema().names()
    subject_col: str = "subject_category" if "subject_category" in names else "subject category"
    object_col: str = "object_category" if "object_category" in names else "object category"
    sr: pl.Expr = pl.col(subject_col).str.replace("biolink:", "").replace(cat_role).fill_null("")
    or_: pl.Expr = pl.col(object_col).str.replace("biolink:", "").replace(cat_role).fill_null("")
    return lf.with_columns(pl.concat_list(pl.concat_str([sr, pl.lit("|"), or_]).replace_strict(edge_lookup, default=default)).alias("category"))


def prune_to_class(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Null out edge columns the row's own association class does not declare.

    ``ALLOWED_EDGE_FIELDS`` is a per-*family* allow-list: it says a column is a slot
    of *some* association class. Whether the specific class chosen for a given row
    accepts it is a separate question, and getting it wrong is the single largest
    source of ``extra_forbidden`` failures (``species_context_qualifier`` and friends
    on a class that has no such slot).

    Categories vary per row within a section, so this masks per row rather than
    dropping columns: values are nulled where the row's class rejects them, and the
    Rust null-stripper then removes the key entirely. Scalars are wrapped where the
    class declares the slot multivalued.

    Args:
        lf: Edges LazyFrame carrying a resolved ``category`` column.

    Returns:
        LazyFrame whose every remaining value is legal for its own row's class.
    """
    schema: pl.Schema = lf.collect_schema()
    names: list[str] = schema.names()
    if "category" not in names:
        return lf

    core: frozenset[str] = frozenset({"category", "subject", "object", "predicate", "id"})
    candidates: list[str] = [c for c in names if c not in core]
    if not candidates:
        return lf

    categories: list[str] = [f"biolink:{c.value}" for c in EdgeCategories]
    first: pl.Expr = pl.col("category").list.first()
    updates: list[pl.Expr] = []
    rescued: list[pl.Expr] = []
    for col in candidates:
        text: pl.Expr = (
            pl.col(col).list.eval(pl.element().cast(pl.String)).list.join(", ") if isinstance(schema[col], pl.List) else pl.col(col).cast(pl.String)
        )
        accepts: dict[str, bool] = {cat: col in class_fields(association_class(cat)) for cat in categories}
        # A closed-vocabulary slot additionally constrains the *value*. A qualifier
        # encoded from a column carries whatever the sheet holds, so the token can only
        # be checked here -- config-time validation sees no data.
        vocabulary: frozenset[str] | None = ENUM_RANGED_QUALIFIERS.get(col)
        ok: pl.Expr = first.replace_strict(accepts, default=False) if not all(accepts.values()) else pl.lit(True)
        if vocabulary is not None:
            ok = ok & text.is_in(list(vocabulary))
        if all(accepts.values()) and vocabulary is None:
            keep: pl.Expr = pl.col(col)
        elif not any(accepts.values()):
            continue  # Handled upstream by the allow-list / study routing.
        else:
            keep = pl.when(ok).then(pl.col(col)).otherwise(None)
            # Preserve what the class refuses rather than deleting it outright; the
            # value is real evidence, it just has no slot on this association class.
            rescued.append(pl.when(ok | text.is_null()).then(None).otherwise(pl.concat_str([pl.lit(f"{col}="), text])))
        # Biolink makes the same slot multivalued on some classes and scalar on others.
        listed: dict[str, bool] = {cat: is_multivalued(association_class(cat), col) for cat in categories}
        if any(listed.values()) and not isinstance(schema[col], pl.List):
            keep = pl.when(first.replace_strict(listed, default=False)).then(pl.concat_list(keep)).otherwise(keep)
        updates.append(keep.alias(col))
    if rescued:
        updates.append(pl.concat_list(rescued).list.drop_nulls().alias(PRUNED_COLUMN))
    return lf.with_columns(updates) if updates else lf


def value(lf: pl.LazyFrame, col: str, x: object) -> pl.LazyFrame:
    """Add a new column populated with a literal value.

    Args:
        lf: Source LazyFrame.
        col: Name of the new column.
        x: Literal value to populate every row with.

    Returns:
        LazyFrame with the new literal column appended.
    """
    return lf.with_columns(pl.lit(x).alias(col))


def derive_species_context(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Derive ``species_context_qualifier`` from resolved node taxon columns.

    Uses ``subject_taxon`` first and falls back to ``object_taxon``. Null values
    indicate no resolved taxon metadata and are later stripped from NDJSON
    output by ``dedup_stream``.

    Args:
        lf: Source LazyFrame after subject/object fullmap resolution.

    Returns:
        LazyFrame with an auto-derived ``species_context_qualifier`` edge column.
    """
    return lf.with_columns(pl.coalesce(pl.col("subject_taxon"), pl.col("object_taxon")).alias("species_context_qualifier"))


def _retrieval_source(resource_id: str, resource_role: str, upstream: list[str] | None = None, urls: list[str] | None = None) -> pl.Expr:
    """Build one ``RetrievalSource`` struct expression.

    Every entry declares the same four fields so that :func:`retrieval_sources` can
    ``concat_list`` them into a single ``list[struct]`` column; absent list fields are
    typed nulls, which the Rust null-stripper removes from the emitted JSON.
    """
    empty: pl.Expr = pl.lit(None, dtype=pl.List(pl.String))
    return pl.struct(
        # RetrievalSource.id is required; translator-ingests sets it to the resource_id.
        pl.lit(resource_id).alias("id"),
        pl.lit(resource_id).alias("resource_id"),
        pl.lit(resource_role).alias("resource_role"),
        (pl.concat_list([pl.lit(x) for x in upstream]) if upstream else empty).alias("upstream_resource_ids"),
        (pl.concat_list([pl.lit(x) for x in urls]) if urls else empty).alias("source_record_urls"),
    )


PRUNED_COLUMN: str = "_pruned_by_class"
"""Internal handoff column: values `prune_to_class` removed, for the study to absorb.

Never reaches output -- :func:`inline_supporting_study` folds it into the
``StudyResult`` description and drops it.
"""


def inline_supporting_study(lf: pl.LazyFrame, study_id: str, sheet: str | None) -> pl.LazyFrame:
    """Attach table provenance and homeless statistics as an inlined Biolink ``Study``.

    Follows the COHD/ICEES pattern in ``translator-ingests``: the edge carries
    ``has_supporting_studies`` (``dict[str, Study]``, ``inlined: true`` on
    ``Association``) and each ``Study`` carries ``has_study_results``. The Study is
    deliberately *not* written to the nodes file, matching those ingests.

    Two kinds of column are routed here rather than left on the edge:

    * the sheet name and source row number, which previously became
      ``"sheet_name: Table_S7"`` strings inside ``supporting_text`` -- a slot whose
      Biolink meaning is a supporting sentence, not a key/value dump;
    * any column in :data:`biolink.UNSATISFIABLE_EDGE_FIELDS`, i.e. declared in the
      LinkML schema but attached to no Pydantic class under the *installed*
      biolink-model. With ``biolink/biolink-model#1770`` applied the
      ``supporting_study_*`` slots become real ``Association`` fields and are left
      flat on the edge instead; nothing here is hardcoded to either state.

    ``study_id`` is a per-section constant, so it can key a static struct field.

    Args:
        lf: Edges LazyFrame after annotation and provenance ops.
        study_id: Stable study identifier (``"<publication>#<sheet>"``).
        sheet: Worksheet name, when the source is a spreadsheet.

    Returns:
        LazyFrame with ``has_supporting_studies`` appended and the routed columns dropped.
    """
    names: list[str] = lf.collect_schema().names()
    routed: list[str] = sorted(c for c in names if c in UNSATISFIABLE_EDGE_FIELDS)
    row: str = "extracted_from_row_number"
    has_row: bool = row in names

    result_id: pl.Expr = pl.concat_str([pl.lit(f"{study_id}#row"), pl.col(row).cast(pl.String)]) if has_row else pl.lit(f"{study_id}#result")
    label: str = f"{sheet} row " if sheet else "row "
    result_name: pl.Expr = pl.concat_str([pl.lit(label), pl.col(row).cast(pl.String)]) if has_row else pl.lit(sheet or study_id)

    # Statistics with no Association slot are preserved as a readable summary rather
    # than silently dropped; `StudyResult.has_attribute` is `list[str]` (not inlined),
    # so typed Attributes would require emitting Attribute rows into the nodes file.
    fields: list[pl.Expr] = [result_id.alias("id"), result_name.alias("name")]
    pruned: bool = PRUNED_COLUMN in names
    if routed or pruned:
        parts: list[pl.Expr] = []
        for col in routed:
            text: pl.Expr = pl.col(col).cast(pl.String).str.strip_chars()
            blank: pl.Expr = text.is_null() | (text.str.len_chars() == 0)
            parts.append(pl.when(blank).then(pl.lit(None, dtype=pl.String)).otherwise(pl.concat_str([pl.lit(f"{col}="), text])))
        # Qualifiers the resolved association class refuses (see `prune_to_class`) are
        # appended to the routed statistics. Build from whichever sources exist: an
        # empty list literal would be a zero-length series and fail to broadcast.
        summary: pl.Expr
        if parts and pruned:
            summary = pl.concat_list(parts).list.drop_nulls().list.concat(pl.col(PRUNED_COLUMN))
        elif parts:
            summary = pl.concat_list(parts)
        else:
            summary = pl.col(PRUNED_COLUMN)
        fields.append(summary.list.drop_nulls().list.join("; ").alias("description"))

    study: pl.Expr = pl.struct(
        pl.lit(study_id).alias("id"), pl.lit(sheet or study_id).alias("name"), pl.concat_list(pl.struct(fields)).alias("has_study_results")
    )
    out: pl.LazyFrame = lf.with_columns(pl.struct(study.alias(study_id)).alias("has_supporting_studies"))
    drop: list[str] = [*routed, *([row] if has_row else []), *(["sheet_name"] if "sheet_name" in names else []), *([PRUNED_COLUMN] if pruned else [])]
    return out.drop(drop)


def retrieval_sources(lf: pl.LazyFrame, primary: str, upstream: list[str], urls: list[str]) -> pl.LazyFrame:
    """Add the Biolink ``sources`` retrieval-provenance column.

    ``upstream_resource_ids`` and ``source_record_urls`` have ``domain: retrieval
    source`` in the Biolink Model, so they are properties of an entry in ``sources``
    -- not of the association. Emitting them flat on the edge makes every record fail
    validation with ``extra_forbidden``.

    Mirrors ``build_association_knowledge_sources()`` from
    ``translator-ingests/util/biolink.py``: the primary knowledge source carries the
    source record URLs and lists the upstream resources, and each upstream resource
    additionally appears as its own ``supporting_data_source`` entry.

    Args:
        lf: Source LazyFrame.
        primary: Infores CURIE of the primary knowledge source.
        upstream: Infores CURIEs of upstream/supporting data sources.
        urls: Source record URLs for the primary entry.

    Returns:
        LazyFrame with a ``sources`` ``list[struct]`` column appended.
    """
    entries: list[pl.Expr] = [_retrieval_source(primary, "primary_knowledge_source", upstream, urls)]
    entries.extend(_retrieval_source(x, "supporting_data_source") for x in upstream)
    return lf.with_columns(pl.concat_list(entries).alias("sources"))


def publications(lf: pl.LazyFrame, curies: str | list[str]) -> pl.LazyFrame:
    """Add publication CURIEs as a Biolink-compliant ``list[str]`` column.

    Args:
        lf: Source LazyFrame.
        curies: One publication CURIE or a list of publication CURIEs.

    Returns:
        LazyFrame with the new ``publications`` list column appended.
    """
    values: list[str] = [curies] if isinstance(curies, str) else curies
    return lf.with_columns(pl.concat_list([pl.lit(curie) for curie in values]).alias("publications"))


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

    P-value columns by substring plus the exact ``effect_size`` and
    study-size fields. The old ``sample_size`` / ``relationship_strength``
    names are absent on purpose: the column coercions rename them to
    ``supporting_study_size`` / ``effect_size`` before ``clean_numeric`` /
    ``format_numeric`` run.

    Args:
        names: Schema column names to filter.

    Returns:
        Subset of ``names`` destined for numeric coercion/formatting.
    """
    # P-value columns by substring plus exact effect-size and study-size fields.
    exact: set[str] = {"effect_size", "supporting_study_size"}
    return [c for c in names if ("p_value" in c.lower()) or (c in exact)]


def clean_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Coerce numeric annotation columns to Float64, dropping non-numeric values to null.

    Only touches p-value, effect-size and study-size columns.

    Args:
        lf: Source LazyFrame.

    Returns:
        LazyFrame with the matched columns cast to Float64 (no-op if none match).
    """
    # Only touches p-value, effect-size and study-size columns.
    cols: list[str] = numeric_columns(lf.collect_schema().names())
    if not cols:
        return lf
    return lf.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in cols])


def format_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize numeric annotation columns for output.

    Columns that map to a numeric Biolink slot are emitted as real JSON numbers:
    ``p_value`` and ``adjusted_p_value`` are typed ``float`` in the model (and
    ``supporting_study_size`` ``integer`` once ``biolink/biolink-model#1770`` lands),
    so writing ``"6.5200e-06"`` produces a file that strict consumers reject even
    though Pydantic's lax mode happens to coerce it.

    Columns with no numeric Biolink slot keep the controlled string notation --
    p-value-like names use scientific (``{:.4e}``), others decimal general
    (``{:.4g}``) -- because they end up in human-readable text (the inlined
    ``StudyResult`` description or ``supporting_text``). Null values stay null.

    Args:
        lf: Source LazyFrame.

    Returns:
        New LazyFrame (eagerly collected then re-lazied) with matched columns
        typed or formatted.

    Notes:
        Collection point: the column is formatted in one batch so notation is
        controlled across all rows at once.
    """
    # Collection point: batch formatting for notation control.
    df: pl.DataFrame = lf.collect()
    for c in numeric_columns(df.columns):
        kind: str | None = numeric_slot_kind(c)
        if kind == "float":
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))
            continue
        if kind == "int":
            df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).round().cast(pl.Int64, strict=False).alias(c))
            continue
        df = df.with_columns(pl.col(c).cast(pl.Float64, strict=False).alias(c))
        fmt: str = "{:.4e}" if "p_value" in c.lower() else "{:.4g}"
        formatted: list[str | None] = [None if v is None else fmt.format(v) for v in df[c].to_list()]
        df = df.with_columns(pl.Series(c, formatted))
    return df.lazy()


def split_expr(col: str, delimiter: str) -> pl.Expr:
    """Build the shared "split a delimited cell into items" expression.

    The single splitting primitive behind both delimiter-driven ops: ``explode_by``
    fans the items out into rows (node encodings), ``split_by`` keeps them as a real
    JSON array on the row (annotations). Only what happens to the items afterwards
    differs, so the parsing rules stay defined in exactly one place.

    Items are trimmed and blanks are dropped -- a trailing or doubled separator
    (``"a;b;"``, ``"a;;b"``) is a delimited-text artifact, not a value. A null cell
    stays null rather than becoming a one-element list of null.

    Args:
        col: Column whose string values should be split.
        delimiter: Separator to split on.

    Returns:
        Expression yielding a ``list[str]`` column (null preserved).
    """
    text: pl.Expr = pl.col(col).cast(pl.String)
    items: pl.Expr = text.str.split(delimiter).list.eval(pl.element().str.strip_chars()).list.drop_nulls()
    return pl.when(text.is_null()).then(None).otherwise(items.list.eval(pl.element().filter(pl.element() != "")))


def split_list(lf: pl.LazyFrame, col: str, delimiter: str) -> pl.LazyFrame:
    """Split a delimited cell into a real JSON array, in place.

    A column encoding is scalar by construction, so a multivalued Biolink slot such as
    ``has_evidence`` fed from an aggregated cell would otherwise be emitted as a single
    joined string -- and ``mask_illegal_edge_fields`` wraps that scalar into a
    one-element list, so the value survives Biolink validation while consumers iterate a
    single ``"a|b|c"`` blob instead of three ids.

    Same split as ``explode``, minus the fan-out: this is the one multivalued
    encoding (``split_by``), turning each row's cell into its own JSON array.

    Args:
        lf: Source LazyFrame.
        col: Annotation column to split.
        delimiter: Separator to split on.

    Returns:
        LazyFrame with ``col`` converted to a ``list[str]`` column, blanks dropped.
    """
    return lf.with_columns(split_expr(col, delimiter).alias(col))


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

    Notes:
        Shares ``split_expr`` with ``split_list`` (the ``split_by`` annotation op), so
        both read a delimited cell the same way: items trimmed, blanks dropped. A
        trailing/doubled separator therefore no longer fans out rows carrying ``""``,
        which only ever failed entity resolution and dropped the edge downstream.
    """
    return lf.with_columns(split_expr(col, delimiter).alias(col)).explode(col)


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
        Collection point: ``gather()`` requires an eager frame, so the result
        is re-lazied afterwards.
    """
    df: pl.DataFrame = lf.collect()
    df = df.select(pl.all().gather(indices=rows))
    return df.lazy()


HEAD_ROWS: int = 5


def head(lf: pl.LazyFrame, n: int = HEAD_ROWS) -> pl.LazyFrame:
    """Randomly sample up to ``min(n, height)`` rows for a fast ``--head`` preview.

    Args:
        lf: Source LazyFrame.
        n: Maximum number of rows to keep.

    Returns:
        LazyFrame with at most ``n`` randomly chosen rows (fewer when the source
        has fewer than ``n``).

    Notes:
        ``LazyFrame`` has no ``.sample`` and ``DataFrame.sample`` raises when ``n``
        exceeds the height, so the frame is collected and sampled with
        ``min(n, height)`` (mirroring ``pick``), then re-lazied.
    """
    df: pl.DataFrame = lf.collect()
    return df.sample(n=min(n, df.height), shuffle=True).lazy()


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
    infores: str | None = Field(None)

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
            # --head preview: randomly sample min(HEAD_ROWS, height) rows before any encoding/resolve.
            (head, (HEAD_ROWS,)) if self.head else None,
            [
                op
                for x in self.annotations
                for op in [*self.encoding(x, x.annotation.lower()), *([(split_list, (x.annotation.lower(), x.split_by))] if x.split_by else [])]
            ]
            if self.annotations
            else None,
            (coerce_pvalue_columns, ()),
            (coerce_study_size_columns, ()),
            (coerce_effect_size_columns, ()),
            (coerce_effect_type_columns, ()),
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
        # Enum-ranged qualifiers are excluded from resolution: their range is a closed
        # Biolink vocabulary, so sending them through the fullmap would turn the required
        # token `increased` into the CURIE `UMLS:C0205217`, which the slot rejects.
        qualifiers: list[Qualifier] = self.statement.qualifiers or []
        node_columns: list[tuple[NodeEncoding, str]] = [
            (self.statement.subject, "subject"),
            (self.statement.object, "object"),
            *[(x, x.qualifier) for x in qualifiers if x.resolved],
        ]
        literals: list[Qualifier] = [x for x in qualifiers if not x.resolved]
        # A nullable qualifier keeps its edge when the cell is blank or unresolvable (the
        # column stays null and the null-stripper omits the key); subject/object and
        # strict qualifiers drop the row as before.
        specs: list[ResolveSpec] = [
            ResolveSpec(
                col,
                str(x.taxon) if x.taxon else None,
                x.prioritize,
                x.avoid,
                x.exclude_prefixes,
                x.exclude_regex,
                x.nullable if isinstance(x, Qualifier) else False,
            )
            for x, col in node_columns
        ]
        return [
            [self.node_prep(x, col) for x, col in node_columns],
            # Encode only: no pre-resolution snapshot and no NLP normalization, both of
            # which exist to feed entity resolution these columns never undergo.
            [self.encoding(x, x.qualifier) for x in literals],
            (resolve_batch, (specs, db, self.log, self.store.stem, self.config.name, True)),
            # QC audits only the strict columns: a nullable qualifier's nulls are expected
            # (blank cell / no match), not resolution errors for the audit to delete.
            [
                (fullmap_audit, (col, self.store.stem, self.config.name, "passed", True))
                for x, col in node_columns
                if not (isinstance(x, Qualifier) and x.nullable)
            ]
            if self.qc
            else None,
        ]

    def _provenance_ops(self: Self) -> list[Any]:
        """Collect the edge/provenance/finalize ops that follow entity resolution.

        Returns:
            Raw op list: predicate and edge category, provenance metadata,
            then the trim/format/write finalize ops.
        """
        override = self.provenance.override
        primary_knowledge_source: str | None = self.infores or (infores(self.name) if self.name else None)
        upstream_ids = override.upstream_resource_ids if override else upstream_resource_ids(self.provenance.repo)
        knowledge_level = override.knowledge_level if override else self.provenance.knowledge_level
        agent_type = override.agent_type if override else self.provenance.agent_type
        publication_values = override.publications if override else [publication_curie(self.provenance.repo, self.provenance.publication or "")]
        # The study is the table itself: one publication, one worksheet. Both are
        # section constants, so the study id can key a static struct field.
        sheet: str | None = self.source.sheet if self.source.kind == Files.EXCEL else None  # pyright: ignore
        publication: str = publication_values[0] if publication_values else (self.config.name or "study")
        study_id: str = f"{publication}#{sheet}" if sheet else publication
        return [
            (derive_species_context, ()),
            (value, ("predicate", "biolink:" + self.statement.predicate)),
            (edge_category, ("biolink:" + self.statement.predicate,)),
            (value, ("knowledge_level", knowledge_level)),
            (value, ("agent_type", agent_type)),
            # Biolink `primary_knowledge_source` is a scalar; `sources` carries the
            # structured retrieval provenance (roles, upstream ids, record urls).
            (value, ("primary_knowledge_source", primary_knowledge_source)) if primary_knowledge_source else None,
            (retrieval_sources, (primary_knowledge_source, upstream_ids, [str(u) for u in self.source.url])) if primary_knowledge_source else None,
            (publications, (publication_values,)) if publication_values else None,
            # Prune first so class-rejected values are handed to the study rather than lost.
            (prune_to_class, ()),
            (inline_supporting_study, (study_id, sheet)),
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
    coerce_effect_size_columns: "clean",
    coerce_effect_type_columns: "clean",
    clean_numeric: "clean",
    level_one: "resolve",
    level_two: "resolve",
    resolve: "resolve",
    resolve_batch: "resolve",
    fullmap_audit: "qc",
    column: "encode",
    derive_species_context: "edge",
    edge_category: "edge",
    publications: "provenance",
    retrieval_sources: "provenance",
    inline_supporting_study: "provenance",
    prune_to_class: "finalize",
    split_list: "encode",
    sig: "significance",
    drop_not_significant: "significance",
    trim: "finalize",
    format_numeric: "finalize",
    to_store: "write",
}

# Ops that accept an ``on_phase`` callback so ``compile_subgraph`` can forward fine-grained sub-phases.
PHASE_AWARE: frozenset[Callable] = frozenset({resolve_batch, fullmap_audit})

UNKNOWN_PHASE: str = "transform"

_VALUE_PROVENANCE_COLS: frozenset[str] = frozenset({"knowledge_level", "agent_type", "primary_knowledge_source", "sheet_name"})


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


def normalize(edges: pl.LazyFrame, col: str, names: list[str] | None = None, infores_id: str | None = None) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Normalize disparate node columns into a unified format and remove them from edges.

    Emits Biolink ``NamedThing`` slots only. The fullmap's ``<col>_source`` (a file
    name such as ``gene.txt``) and ``<col>_source_version`` are build provenance, not
    node properties -- ``source``/``source_version`` are ``extra_forbidden`` on every
    Biolink node class, and the version belongs at graph level (the RIG), matching
    ``translator-ingests/util/metadata.py``. The taxon is emitted as ``in_taxon`` plus
    ``in_taxon_label``, the slots ``translator-ingests`` uses.

    Args:
        edges: Source edges LazyFrame containing ``<col>``, ``<col>_name``,
            ``<col>_category``, ``<col>_taxon``, ``<col>_source``, and
            ``<col>_source_version`` columns.
        col: Base node column name (e.g. ``"subject"``).
        names: Output column names for the produced nodes frame.
        infores_id: Graph-level infores CURIE recorded as ``provided_by``.

    Returns:
        Tuple of ``(partial_nodes, modified_edges)`` as LazyFrames.
    """
    if names is None:
        names = ["id", "name", "category", "in_taxon", "in_taxon_label"]
    cols: list[str] = [col, f"{col}_name", f"{col}_category", f"{col}_taxon", f"{col}_taxon_label"]
    available: list[str] = edges.collect_schema().names()
    # `<col>_taxon_label` is not produced by every fullmap revision.
    pairs: list[tuple[str, str]] = [(c, n) for c, n in zip(cols, names, strict=True) if c in available]
    nodes: pl.LazyFrame = edges.select([c for c, _ in pairs]).unique().rename(dict(pairs))
    # Ensures category has biolink: prefix.
    nodes = nodes.with_columns(
        pl.when(pl.col("category").str.starts_with("biolink:"))
        .then(pl.col("category"))
        .otherwise(pl.lit("biolink:") + pl.col("category"))
        .alias("category")
    )
    # Entity resolution can land on a class that cannot be emitted as a KGX node --
    # `Publication` requires `publication_type`, `ClinicalAttribute` requires
    # `has_attribute_type`, and mixins like `GenomicEntity` reject their own name in the
    # `category` literal. Demote those to the nearest emittable ancestor.
    # `replace` (not `replace_strict`) so nulls and unrecognized spellings pass through
    # untouched -- only categories that genuinely need demoting are rewritten.
    emittable: dict[str, str] = {
        curie: resolved for c in Categories for curie in (f"biolink:{c.value}",) if (resolved := resolve_node_category(curie)) != curie
    }
    nodes = nodes.with_columns(pl.col("category").replace(emittable).alias("category"))
    # Exports category within a list (null categories stay null for strip_nulls).
    nodes = nodes.with_columns(pl.when(pl.col("category").is_not_null()).then(pl.concat_list(pl.col("category"))).alias("category"))
    if "in_taxon" in nodes.collect_schema().names():
        # `in_taxon` is multivalued on Biolink `thing with taxon`.
        nodes = nodes.with_columns(pl.when(pl.col("in_taxon").is_not_null()).then(pl.concat_list(pl.col("in_taxon"))).alias("in_taxon"))
    if infores_id:
        nodes = nodes.with_columns(pl.concat_list(pl.lit(infores_id)).alias("provided_by"))
    # Drop every derived node column from the edges, including the ones not emitted.
    drop: list[str] = [c for c in (*cols[1:], f"{col}_source", f"{col}_source_version") if c in available]
    edges_out: pl.LazyFrame = edges.drop(drop)
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
        # List-typed columns cannot be cast to String directly; join their elements so
        # a folded `sources`-style column degrades readably instead of raising.
        s: pl.Expr = (
            pl.col(col).list.eval(pl.element().cast(pl.String)).list.join(", ") if isinstance(schema[col], pl.List) else pl.col(col).cast(pl.String)
        ).str.strip_chars()
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
    infores_id: str | None = None,
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
            partial, lf = normalize(lf, col, infores_id=infores_id)
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
    infores_id: str | None,
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
        infores_id: Optional graph-level infores CURIE for the RIG.
        on_phase: Optional callback fired with ``"write-nodes"``,
            ``"write-edges"``, ``"dedup"`` and ``"rig"`` at each phase
            boundary, used to drive progress UX.
    """
    # Phase: write-nodes. Collection point: appending to output files.
    if on_phase is not None:
        on_phase("write-nodes")
    node_rows: list[dict[str, object]] = []
    with nodes_tmp.open("a", encoding="utf-8") as f:
        for subnode in subnodes:
            eagernode: pl.DataFrame = subnode.collect().unique()
            node_rows.extend(eagernode.to_dicts())
            eagernode.write_ndjson(f)

    # Phase: write-edges.
    if on_phase is not None:
        on_phase("write-edges")
    with edges_tmp.open("a", encoding="utf-8") as f:
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
        infores_id,
    )


def compile_graph(
    subgraphs: list[Path],
    name: str,
    version: str,
    description: str | None = None,
    contributions: list[str] | None = None,
    ui_explanation: str | None = None,
    tables: list[Path] | None = None,
    infores_id: str | None = None,
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
        infores_id: Optional graph-level infores CURIE for the RIG.
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
    subnodes, subedges, edge_type_info = _collect_subframes(subgraphs, e, ui_explanation, on_phase, on_subgraph, infores_id)
    _write_ndjson(subnodes, subedges, edge_type_info, n, e, name, version, description, contributions, ui_explanation, tables, infores_id, on_phase)


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
        taxon: Optional NCBITaxon constraint applied to taxon-bearing matches while retaining rows with no taxon metadata.
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
