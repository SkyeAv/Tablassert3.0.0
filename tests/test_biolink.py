"""Tests for the Biolink-derived vocabulary in :mod:`tablassert.biolink`.

These replace the hand-maintained-enum tests that used to live in
``test_enums.py``. Beyond membership checks, they act as *drift guards*: the
derived sets are cross-checked against the ``biolink-model`` package itself (the
source of truth), and specific Biolink 4.4.3 renames / additions / removals are
asserted so that a change in the underlying model is surfaced loudly.
"""

from __future__ import annotations

import inspect
from enum import Enum
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import pytest

import biolink_model.datamodel.pydanticmodel_v2 as bm
from tablassert.biolink import ALLOWED_EDGE_FIELDS, BIOLINK_VERSION, AgentTypes, Categories, EdgeCategories, KnowledgeLevels, Predicates, Qualifiers

if TYPE_CHECKING:
    from linkml_runtime.utils.schemaview import SchemaView


@pytest.fixture(scope="module")
def schema() -> SchemaView:
    """An independent ``SchemaView`` over the bundled Biolink schema (test oracle)."""
    from linkml_runtime.utils.schemaview import SchemaView

    return SchemaView(str(files("biolink_model").joinpath("schema/biolink_model.yaml")))


def _slot_names_snake(schema: SchemaView) -> set[str]:
    return {name.replace(" ", "_") for name in schema.all_slots()}


def _category_name(cls: Any) -> str:
    """Bare Biolink category name for a Pydantic model class (read on ``Any`` to dodge narrowing)."""
    field: Any = cls.model_fields.get("category")
    if field is not None and isinstance(field.default, list):
        for value in field.default:
            if isinstance(value, str) and value.startswith("biolink:"):
                return value[len("biolink:") :]
    return str(cls.__name__)


def _biolink_category_names() -> set[str]:
    """Canonical Biolink entity category names, read directly from the Pydantic model."""
    return {
        _category_name(cls)
        for cls in vars(bm).values()
        if inspect.isclass(cls) and cls.__module__ == bm.__name__ and issubclass(cls, bm.Entity) and not issubclass(cls, bm.Association)
    }


def _biolink_association_names() -> set[str]:
    """Canonical Biolink association names, read directly from the Pydantic model."""
    return {
        _category_name(cls) for cls in vars(bm).values() if inspect.isclass(cls) and cls.__module__ == bm.__name__ and issubclass(cls, bm.Association)
    }


# --- enum shape ---------------------------------------------------------------------------------


@pytest.mark.parametrize("enum", [Categories, EdgeCategories, Predicates, Qualifiers, KnowledgeLevels, AgentTypes], ids=lambda e: e.__name__)
def test_is_str_enum(enum: type) -> None:
    """Every derived vocabulary is a ``str``/``Enum`` subclass (usable as a Pydantic field type)."""
    assert issubclass(enum, str)
    assert issubclass(enum, Enum)


# --- spot-check members referenced by name elsewhere -------------------------------------------


def test_categories_has_gene() -> None:
    assert Categories.GENE == "Gene"


def test_categories_has_disease() -> None:
    assert Categories.DISEASE == "Disease"


def test_categories_has_protein() -> None:
    assert Categories.PROTEIN == "Protein"


def test_predicates_has_treats() -> None:
    assert Predicates.TREATS == "treats"


def test_predicates_has_related_to() -> None:
    assert Predicates.RELATED_TO == "related_to"


def test_qualifiers_has_disease_context() -> None:
    assert Qualifiers.DISEASE_CONTEXT_QUALIFIER == "disease_context_qualifier"


def test_knowledge_levels_has_statistical_association() -> None:
    assert KnowledgeLevels.STATISTICAL_ASSOCIATION == "statistical_association"


def test_knowledge_levels_has_not_provided() -> None:
    assert KnowledgeLevels.NOT_PROVIDED == "not_provided"


def test_agent_types_has_data_analysis_pipeline() -> None:
    assert AgentTypes.DATA_ANALYSIS_PIPELINE == "data_analysis_pipeline"


def test_agent_types_has_manual_agent() -> None:
    assert AgentTypes.MANUAL_AGENT == "manual_agent"


# --- provenance ---------------------------------------------------------------------------------


def test_biolink_version_present() -> None:
    """The derived vocabulary records a concrete Biolink Model version."""
    assert BIOLINK_VERSION
    assert BIOLINK_VERSION != "unknown"
    assert "." in BIOLINK_VERSION


# --- exact match against the biolink-model source of truth --------------------------------------


def test_knowledge_levels_match_biolink() -> None:
    """KnowledgeLevels is exactly the Biolink ``KnowledgeLevelEnum`` value set."""
    assert {e.value for e in KnowledgeLevels} == {e.value for e in bm.KnowledgeLevelEnum}


def test_agent_types_match_biolink() -> None:
    """AgentTypes is exactly the Biolink ``AgentTypeEnum`` value set."""
    assert {e.value for e in AgentTypes} == {e.value for e in bm.AgentTypeEnum}


def test_categories_match_biolink() -> None:
    """Categories is exactly the set of Biolink entity category names."""
    assert {c.value for c in Categories} == _biolink_category_names()


def test_edge_categories_match_biolink() -> None:
    """EdgeCategories is exactly the set of Biolink association names."""
    assert {e.value for e in EdgeCategories} == _biolink_association_names()


def test_predicates_are_biolink_slots(schema: SchemaView) -> None:
    """Every predicate is a real Biolink slot (guarantees translator-ingests needs no re-validation)."""
    slots: set[str] = _slot_names_snake(schema)
    for predicate in Predicates:
        assert predicate.value in slots, predicate.value


def test_qualifiers_are_biolink_slots(schema: SchemaView) -> None:
    """Every qualifier is a real Biolink slot."""
    slots: set[str] = _slot_names_snake(schema)
    for qualifier in Qualifiers:
        assert qualifier.value in slots, qualifier.value


# --- Biolink 4.4.3 drift markers ----------------------------------------------------------------


def test_categories_reflect_4_4_3_renames() -> None:
    """Biolink 4.4.3 renamed the RNA classes to acronym CamelCase; the old spellings are gone."""
    values: set[str] = {c.value for c in Categories}
    assert "MicroRNA" in values
    assert "SiRNA" in values
    assert "RNAProduct" in values
    assert "Microrna" not in values
    assert "Sirna" not in values


def test_categories_include_upper_levels() -> None:
    values: set[str] = {c.value for c in Categories}
    assert "Entity" in values
    assert "NamedThing" in values
    assert "Gene" in values
    assert "PhenotypicFeature" in values


def test_predicates_reflect_4_4_3_changes() -> None:
    """4.4.3 added sensitivity predicates and retired the older response predicates."""
    values: set[str] = {p.value for p in Predicates}
    assert "affects_sensitivity_to" in values
    assert "related_to" in values
    assert "treats" in values
    # Retired/renamed in 4.4.3 -- must no longer be emitted.
    assert "assesses" not in values
    assert "response_affected_by" not in values


def test_qualifiers_reflect_4_4_3_additions() -> None:
    values: set[str] = {q.value for q in Qualifiers}
    assert "qualifier" in values
    assert "process_qualifier" in values
    assert "response_context_qualifier" in values


def test_edge_categories_include_associations() -> None:
    values: set[str] = {e.value for e in EdgeCategories}
    assert "Association" in values
    assert "GeneToDiseaseAssociation" in values
    assert "ChemicalGeneInteractionAssociation" in values
    assert len(values) > 100  # Biolink 4.4.3 defines ~106 association subclasses.


# --- ALLOWED_EDGE_FIELDS ------------------------------------------------------------------------


def test_allowed_edge_fields_covers_required_columns() -> None:
    """Core KGX / Translator edge columns remain allowed (not folded into supporting_text)."""
    required: list[str] = [
        "subject",
        "predicate",
        "object",
        "category",
        "knowledge_level",
        "agent_type",
        "qualifiers",
        "qualified_predicate",
        "primary_knowledge_source",
        "publications",
        "source_record_urls",
        "upstream_resource_ids",
    ]
    for col in required:
        assert col in ALLOWED_EDGE_FIELDS, col


def test_allowed_edge_fields_includes_new_qualifiers() -> None:
    """New 4.4.3 qualifier slots are allowed edge columns."""
    assert "process_qualifier" in ALLOWED_EDGE_FIELDS


def test_allowed_edge_fields_is_superset_of_qualifiers() -> None:
    """Every qualifier slot name is an allowed edge column."""
    assert {q.value for q in Qualifiers} <= set(ALLOWED_EDGE_FIELDS)
