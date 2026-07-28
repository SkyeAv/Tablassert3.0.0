"""Biolink Model vocabulary derived from the ``biolink-model`` package.

This module sources Tablassert's Biolink categories, predicates, qualifiers,
knowledge levels, agent types, edge (association) categories, and allowed edge
fields directly from the official ``biolink-model`` PyPI package -- the same
LinkML-generated Pydantic serialization that ``NCATSTranslator/translator-ingests``
imports (``from biolink_model.datamodel.pydanticmodel_v2 import NamedThing,
Association, ...``) and that Koza validates against.

Deriving these values from the model -- instead of hand-maintaining copies that
can drift -- guarantees Tablassert's KGX output uses only terms that are valid
for the pinned Biolink version. Downstream translator-ingests can therefore
consume Tablassert output without running Koza or re-validating against the
Biolink Model.

Source mapping
--------------
- ``Categories`` / ``EdgeCategories`` / ``KnowledgeLevels`` / ``AgentTypes``:
  introspected from ``biolink_model.datamodel.pydanticmodel_v2`` (entity and
  association Pydantic classes plus the ``KnowledgeLevelEnum`` /
  ``AgentTypeEnum`` value sets).
- ``Predicates`` / ``Qualifiers``: walked from the bundled
  ``biolink_model.yaml`` slot hierarchy via ``linkml-runtime`` (a pinned
  dependency of ``biolink-model``), because predicates and qualifiers are LinkML
  *slots* rather than Pydantic classes.
- ``ALLOWED_EDGE_FIELDS``: the Biolink ``Association`` model fields (walked over
  the MRO) unioned with the qualifier slot names and a curated set of KGX /
  Tablassert edge columns that are not Biolink Association fields.

The enums are built dynamically at runtime from the model. For static type
checking, ``TYPE_CHECKING`` stub classes (declaring only the members referenced
by name elsewhere in the codebase) stand in for the dynamic classes; at runtime
the ``else`` branch builds the full enums. ``tablassert.cli`` imports the modules
that depend on this one (``models`` / ``lib`` / ``fullmap``) inside its pipeline
functions, so the one-time ``linkml-runtime`` parse cost is paid only when a build
or validate actually runs -- never on ``tablassert --help``.
"""

from __future__ import annotations

import inspect
import re
from enum import Enum
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any, cast

import biolink_model.datamodel.pydanticmodel_v2 as _bm

if TYPE_CHECKING:
    from linkml_runtime.utils.schemaview import SchemaView

__all__ = ["ALLOWED_EDGE_FIELDS", "BIOLINK_VERSION", "AgentTypes", "Categories", "EdgeCategories", "KnowledgeLevels", "Predicates", "Qualifiers"]


def _screaming_snake(camel: str) -> str:
    """Convert a Biolink CamelCase name to a SCREAMING_SNAKE enum member name.

    Handles acronym boundaries so ``PhenotypicFeature`` -> ``PHENOTYPIC_FEATURE``
    and ``ChemicalGeneInteractionAssociation`` ->
    ``CHEMICAL_GENE_INTERACTION_ASSOCIATION``.

    Args:
        camel: CamelCase class name (e.g. ``"GeneToDiseaseAssociation"``).

    Returns:
        Upper-snake-cased member name (e.g. ``"GENE_TO_DISEASE_ASSOCIATION"``).
    """
    split: str = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", camel)
    split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", split)
    return split.upper()


def _build_str_enum(name: str, values: list[str]) -> type[Enum]:
    """Build a ``str``-mixed :class:`~enum.Enum` from a list of string values.

    Member names are derived via :func:`_screaming_snake`; values are the verbatim
    Biolink terms. The resulting class is a subclass of both ``str`` and ``Enum``
    (equivalent to ``class Name(str, Enum)``), so it works as a Pydantic field type
    and compares equal to its string value.

    Args:
        name: Class name for the generated enum.
        values: Biolink term values (deduplicated, order-insensitive).

    Returns:
        A new ``str``/``Enum`` subclass.

    Raises:
        ValueError: If two distinct values map to the same member name.
    """
    members: dict[str, str] = {}
    for value in sorted(set(values)):
        member: str = _screaming_snake(value)
        if member in members:
            raise ValueError(f"{name}: duplicate enum member {member!r} for values {members[member]!r} and {value!r}")
        members[member] = value
    return cast("type[Enum]", Enum(name, members, type=str, module=__name__))


def _category_name(cls: Any) -> str:
    """Return the bare Biolink category name for a Pydantic model class.

    Reads the class's ``category`` field default (``["biolink:X", ...]``) and strips
    the ``biolink:`` prefix, falling back to the class name.

    Args:
        cls: A Pydantic class from ``pydanticmodel_v2``.

    Returns:
        The CamelCase category name (e.g. ``"Gene"``).
    """
    field: Any = cls.model_fields.get("category")
    if field is not None and isinstance(field.default, list):
        for value in field.default:
            if isinstance(value, str) and value.startswith("biolink:"):
                return value[len("biolink:") :]
    return str(cls.__name__)


def _entity_category_names() -> list[str]:
    """Collect all Biolink entity category names from the Pydantic model.

    A category is any ``pydanticmodel_v2`` class that subclasses ``Entity`` but is
    not an ``Association`` subclass (associations are edge categories, handled
    separately). This mirrors the Biolink notion of a node category while also
    including the upper-level ``Entity`` / ``NamedThing`` classes.

    Returns:
        Sorted unique category names.
    """
    names: set[str] = set()
    for cls in vars(_bm).values():
        if inspect.isclass(cls) and cls.__module__ == _bm.__name__ and issubclass(cls, _bm.Entity) and not issubclass(cls, _bm.Association):
            names.add(_category_name(cls))
    return sorted(names)


def _association_names() -> list[str]:
    """Collect all Biolink association (edge category) names from the Pydantic model.

    Returns:
        Sorted unique association class names, including ``Association`` itself.
    """
    names: set[str] = set()
    for cls in vars(_bm).values():
        if inspect.isclass(cls) and cls.__module__ == _bm.__name__ and issubclass(cls, _bm.Association):
            names.add(_category_name(cls))
    return sorted(names)


@cache
def _schema() -> SchemaView:
    """Load the bundled Biolink LinkML schema (cached, parsed once).

    Returns:
        A ``linkml-runtime`` ``SchemaView`` over ``biolink_model.yaml``.
    """
    from linkml_runtime.utils.schemaview import SchemaView

    yaml_path: str = str(files("biolink_model").joinpath("schema/biolink_model.yaml"))
    return SchemaView(yaml_path)


def _snake(name: str) -> str:
    """Convert a spaced LinkML slot name to snake_case (``"related to"`` -> ``"related_to"``)."""
    return name.replace(" ", "_")


@cache
def _slot_descendants(root: str) -> frozenset[str]:
    """Return all slot names that descend from ``root`` via ``is_a`` (exclusive of ``root``).

    Args:
        root: Spaced LinkML slot name (e.g. ``"related to"``).

    Returns:
        Frozenset of descendant slot names (spaced form).
    """
    children: dict[str, list[str]] = {}
    for name, slot in _schema().all_slots(imports=False).items():
        if slot.is_a:
            children.setdefault(slot.is_a, []).append(name)
    out: set[str] = set()
    stack: list[str] = [root]
    while stack:
        current: str = stack.pop()
        for child in children.get(current, []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return frozenset(out)


def _predicate_values() -> list[str]:
    """All Biolink predicates: the ``related to`` slot hierarchy, snake_cased (root included)."""
    return sorted({_snake(p) for p in (_slot_descendants("related to") | {"related to"})})


def _qualifier_values() -> list[str]:
    """All Biolink qualifier slot names: the ``qualifier`` slot hierarchy, snake_cased (root included)."""
    return sorted({_snake(q) for q in (_slot_descendants("qualifier") | {"qualifier"})})


def _association_model_fields() -> set[str]:
    """All field names on the Biolink ``Association`` Pydantic class, including inherited ones."""
    fields: set[str] = set()
    for klass in _bm.Association.__mro__:
        model_fields: Any = getattr(klass, "model_fields", None)
        if model_fields:
            fields |= set(model_fields.keys())
    return fields


def _biolink_enum_values(enum_cls: type[Enum]) -> list[str]:
    """Sorted string values of a Biolink Pydantic enum class (e.g. ``KnowledgeLevelEnum``)."""
    return sorted(str(member.value) for member in enum_cls)


# Edge columns Tablassert / KGX emit that are neither Biolink ``Association`` model
# fields nor qualifier slot names: synonym carryover from NamedThing, KGX provenance
# and denormalized fields, supporting-study evidence slots, and Tablassert pipeline
# fields (``source_record_urls``, ``upstream_resource_ids``). Kept curated and unioned
# with the derived Biolink fields so ``ALLOWED_EDGE_FIELDS`` is always a superset of
# what the pipeline may emit (so ``lib.fold_unknown_to_supporting_text`` never starts
# folding legitimate edge columns into ``supporting_text``).
TABLASERT_EDGE_EXTRAS: frozenset[str] = frozenset(
    [
        "broad_synonym",
        "equivalent_identifiers",
        "evidence_direction",
        "evidence_type",
        "exact_synonym",
        "full_name",
        "information_content",
        "narrow_synonym",
        "provided_by",
        "related_synonym",
        "relation",
        "relationship_strength",
        "source_record_urls",
        "statistical_significance_qualifier",
        "supporting_documents",
        "supporting_study_cohort",
        "supporting_study_context",
        "supporting_study_date_range",
        "supporting_study_method_description",
        "supporting_study_method_types",
        "supporting_study_size",
        "synonym",
        "taxon",
        "upstream_resource_ids",
        "xref",
    ]
)


if TYPE_CHECKING:
    # Static stubs for type checking only. They declare just the members referenced
    # by name elsewhere in the codebase; at runtime the ``else`` branch builds the
    # full enums dynamically from the Biolink Model.

    class Categories(str, Enum):
        DISEASE: Categories
        GENE: Categories
        PHENOTYPIC_FEATURE: Categories
        PROTEIN: Categories

    class EdgeCategories(str, Enum):
        ASSOCIATION: EdgeCategories
        CHEMICAL_GENE_INTERACTION_ASSOCIATION: EdgeCategories

    class Predicates(str, Enum):
        RELATED_TO: Predicates
        TREATS: Predicates

    class Qualifiers(str, Enum):
        DISEASE_CONTEXT_QUALIFIER: Qualifiers
        OBJECT_DIRECTION_QUALIFIER: Qualifiers
        SUBJECT_CONTEXT_QUALIFIER: Qualifiers

    class KnowledgeLevels(str, Enum):
        NOT_PROVIDED: KnowledgeLevels
        STATISTICAL_ASSOCIATION: KnowledgeLevels

    class AgentTypes(str, Enum):
        DATA_ANALYSIS_PIPELINE: AgentTypes
        MANUAL_AGENT: AgentTypes

else:
    Categories = _build_str_enum("Categories", _entity_category_names())
    EdgeCategories = _build_str_enum("EdgeCategories", _association_names())
    Predicates = _build_str_enum("Predicates", _predicate_values())
    Qualifiers = _build_str_enum("Qualifiers", _qualifier_values())
    KnowledgeLevels = _build_str_enum("KnowledgeLevels", _biolink_enum_values(cast("type[Enum]", _bm.KnowledgeLevelEnum)))
    AgentTypes = _build_str_enum("AgentTypes", _biolink_enum_values(cast("type[Enum]", _bm.AgentTypeEnum)))


_schema_definition: Any = _schema().schema
BIOLINK_VERSION: str = str(_schema_definition.version) if _schema_definition is not None else "unknown"
"""Version of the Biolink Model these values were derived from (e.g. ``"4.4.3"``)."""

ALLOWED_EDGE_FIELDS: frozenset[str] = frozenset(_association_model_fields()) | {q.value for q in Qualifiers} | TABLASERT_EDGE_EXTRAS
"""Authoritative biolink-compliant edge column allow-list.

Any column on an edge frame that is not in this set is folded into the
``supporting_text`` ``list[str]`` field by ``lib.fold_unknown_to_supporting_text()``
as a ``"column: value"`` string. Composed of the derived Biolink ``Association``
fields, the derived qualifier slot names, and the curated ``TABLASERT_EDGE_EXTRAS``.
"""
