from __future__ import annotations

from enum import Enum

from tablassert.enums import (
    Categories,
    Comparisons,
    Contributions,
    EncodingMethods,
    Files,
    FillMethods,
    Functions,
    Predicates,
    Qualifiers,
    Repositories,
    Syntaxes,
    Tokens,
)


# ? All Enums Are str, Enum Subclasses
def test_tokens_is_str_enum() -> None:
    assert issubclass(Tokens, str)
    assert issubclass(Tokens, Enum)


def test_repositories_is_str_enum() -> None:
    assert issubclass(Repositories, str)
    assert issubclass(Repositories, Enum)


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


def test_syntaxes_is_str_enum() -> None:
    assert issubclass(Syntaxes, str)
    assert issubclass(Syntaxes, Enum)

def test_categories_is_str_enum() -> None:
    assert issubclass(Categories, str)
    assert issubclass(Categories, Enum)


def test_predicates_is_str_enum() -> None:
    assert issubclass(Predicates, str)
    assert issubclass(Predicates, Enum)


def test_qualifiers_is_str_enum() -> None:
    assert issubclass(Qualifiers, str)
    assert issubclass(Qualifiers, Enum)


# ? Enum Values Are Lowercase Strings
def test_tokens_values() -> None:
    assert Tokens.AUTO == "auto"
    assert Tokens.VALUES == "values"


def test_repositories_values() -> None:
    assert Repositories.PUBMED_CENTRAL == "PMC"
    assert Repositories.PUBMED == "PMID"


def test_comparisons_values() -> None:
    assert Comparisons.GT == "gt"
    assert Comparisons.EQ == "eq"
    assert Comparisons.NE == "ne"


def test_contributions_values() -> None:
    assert Contributions.CURATION == "curation"
    assert Contributions.VALIDATION == "validation"
    assert Contributions.TOOL == "tool"


def test_syntaxes_values() -> None:
    assert Syntaxes.TC4 == "TC4"
    assert Syntaxes.GC2 == "GC2"

def test_files_values() -> None:
    assert Files.TEXT == "text"
    assert Files.EXCEL == "excel"


# ? Enum Membership
def test_comparisons_membership() -> None:
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


# ? Enum String Comparison
def test_enum_compares_as_string() -> None:
    assert Tokens.AUTO == "auto"
    assert Repositories.PUBMED_CENTRAL == "PMC"
    assert Syntaxes.TC4 == "TC4"


# ? Category Has Expected Members
def test_categories_has_gene() -> None:
    assert Categories.GENE == "Gene"


def test_categories_has_disease() -> None:
    assert Categories.DISEASE == "Disease"


def test_categories_has_protein() -> None:
    assert Categories.PROTEIN == "Protein"


# ? Predicate Has Expected Members
def test_predicates_has_treats() -> None:
    assert Predicates.TREATS == "treats"


def test_predicates_has_related_to() -> None:
    assert Predicates.RELATED_TO == "related_to"


# ? Qualifier Has Expected Members
def test_qualifiers_has_disease_context() -> None:
    assert Qualifiers.DISEASE_CONTEXT_QUALIFIER == "disease_context_qualifier"
