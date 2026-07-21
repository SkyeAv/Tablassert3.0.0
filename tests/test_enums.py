from __future__ import annotations

from enum import Enum

from tablassert.enums import (
    AgentTypes,
    Categories,
    Comparisons,
    Contributions,
    EncodingMethods,
    Files,
    FillMethods,
    Functions,
    InformationResources,
    KnowledgeLevels,
    Predicates,
    Qualifiers,
    Repositories,
    Tokens,
)


def test_tokens_is_str_enum() -> None:
    """all enums are str, enum subclasses."""
    assert issubclass(Tokens, str)
    assert issubclass(Tokens, Enum)


def test_repositories_is_str_enum() -> None:
    assert issubclass(Repositories, str)
    assert issubclass(Repositories, Enum)


def test_information_resources_is_str_enum() -> None:
    """InformationResources is str, enum subclass."""
    assert issubclass(InformationResources, str)
    assert issubclass(InformationResources, Enum)


def test_comparisons_is_str_enum() -> None:
    assert issubclass(Comparisons, str)
    assert issubclass(Comparisons, Enum)


def test_contributions_is_str_enum() -> None:
    assert issubclass(Contributions, str)
    assert issubclass(Contributions, Enum)


def test_functions_is_str_enum() -> None:
    assert issubclass(Functions, str)
    assert issubclass(Functions, Enum)


def test_files_is_str_enum() -> None:
    assert issubclass(Files, str)
    assert issubclass(Files, Enum)


def test_encoding_methods_is_str_enum() -> None:
    assert issubclass(EncodingMethods, str)
    assert issubclass(EncodingMethods, Enum)


def test_fill_methods_is_str_enum() -> None:
    assert issubclass(FillMethods, str)
    assert issubclass(FillMethods, Enum)


def test_categories_is_str_enum() -> None:
    assert issubclass(Categories, str)
    assert issubclass(Categories, Enum)


def test_predicates_is_str_enum() -> None:
    assert issubclass(Predicates, str)
    assert issubclass(Predicates, Enum)


def test_qualifiers_is_str_enum() -> None:
    assert issubclass(Qualifiers, str)
    assert issubclass(Qualifiers, Enum)


def test_knowledge_levels_is_str_enum() -> None:
    """KnowledgeLevels and AgentTypes are str, enum subclasses."""
    assert issubclass(KnowledgeLevels, str)
    assert issubclass(KnowledgeLevels, Enum)


def test_agent_types_is_str_enum() -> None:
    assert issubclass(AgentTypes, str)
    assert issubclass(AgentTypes, Enum)


def test_tokens_values() -> None:
    """enum values are lowercase strings."""
    assert Tokens.AUTO == "auto"
    assert Tokens.VALUES == "values"


def test_repositories_values() -> None:
    assert Repositories.PUBMED_CENTRAL == "PMC"
    assert Repositories.PUBMED == "PMID"


def test_information_resources_values() -> None:
    """InformationResources values."""
    assert InformationResources.PUBMED == "infores:pubmed"
    assert InformationResources.PUBMED_CENTRAL == "infores:pubmed-central"


def test_comparisons_values() -> None:
    assert Comparisons.GT == "gt"
    assert Comparisons.EQ == "eq"
    assert Comparisons.NE == "ne"


def test_contributions_values() -> None:
    assert Contributions.CURATION == "curation"
    assert Contributions.VALIDATION == "validation"
    assert Contributions.TOOL == "tool"


def test_files_values() -> None:
    assert Files.TEXT == "text"
    assert Files.EXCEL == "excel"


def test_comparisons_membership() -> None:
    """enum membership."""
    names: list[str] = [e.value for e in Comparisons]
    assert "gt" in names
    assert "ge" in names
    assert "lt" in names
    assert "le" in names
    assert "eq" in names
    assert "ne" in names
    assert len(names) == 6


def test_fill_methods_membership() -> None:
    names: list[str] = [e.value for e in FillMethods]
    assert "forward" in names
    assert "backward" in names
    assert "zero" in names
    assert "one" in names
    assert len(names) == 7


def test_enum_compares_as_string() -> None:
    """enum string comparison."""
    assert Tokens.AUTO == "auto"
    assert Repositories.PUBMED_CENTRAL == "PMC"


def test_categories_has_gene() -> None:
    """category has expected members."""
    assert Categories.GENE == "Gene"


def test_categories_has_disease() -> None:
    assert Categories.DISEASE == "Disease"


def test_categories_has_protein() -> None:
    assert Categories.PROTEIN == "Protein"


def test_predicates_has_treats() -> None:
    """predicate has expected members."""
    assert Predicates.TREATS == "treats"


def test_predicates_has_related_to() -> None:
    assert Predicates.RELATED_TO == "related_to"


def test_qualifiers_has_disease_context() -> None:
    """qualifier has expected members."""
    assert Qualifiers.DISEASE_CONTEXT_QUALIFIER == "disease_context_qualifier"


def test_knowledge_levels_has_statistical_association() -> None:
    """KnowledgeLevels has expected members."""
    assert KnowledgeLevels.STATISTICAL_ASSOCIATION == "statistical_association"


def test_knowledge_levels_has_not_provided() -> None:
    assert KnowledgeLevels.NOT_PROVIDED == "not_provided"


def test_agent_types_has_data_analysis_pipeline() -> None:
    """AgentTypes has expected members."""
    assert AgentTypes.DATA_ANALYSIS_PIPELINE == "data_analysis_pipeline"


def test_agent_types_has_manual_agent() -> None:
    assert AgentTypes.MANUAL_AGENT == "manual_agent"
