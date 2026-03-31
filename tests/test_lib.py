from __future__ import annotations

from typing import Any

from tablassert.lib import idxname, label_edge, strip_nulls


# ? idxname Converts Single Letter Columns
def test_idxname_single_letter() -> None:
    assert idxname("A") == "column_1"
    assert idxname("B") == "column_2"
    assert idxname("Z") == "column_26"


# ? idxname Converts Double Letter Columns
def test_idxname_double_letter() -> None:
    assert idxname("AA") == "column_27"
    assert idxname("AB") == "column_28"
    assert idxname("AZ") == "column_52"


# ? idxname Converts Triple Letter Columns
def test_idxname_triple_letter() -> None:
    assert idxname("AAA") == "column_703"


# ? idxname Returns Column Prefixed String
def test_idxname_format() -> None:
    result: str = idxname("C")
    assert result.startswith("column_")


# ? strip_nulls Removes Null Like Values
def test_strip_nulls_removes_empty_string() -> None:
    r: dict[str, Any] = {"a": "hello", "b": ""}
    result: dict = strip_nulls(r)
    assert "a" in result
    assert "b" not in result


# ? strip_nulls Removes Na Nan Null None
def test_strip_nulls_removes_null_variants() -> None:
    r: dict[str, Any] = {"a": "na", "b": "nan", "c": "null", "d": "none"}
    result: dict = strip_nulls(r)
    assert len(result) == 0


# ? strip_nulls Case Insensitive
def test_strip_nulls_case_insensitive() -> None:
    r: dict[str, Any] = {"a": "NA", "b": "NaN", "c": "NULL", "d": "None"}
    result: dict = strip_nulls(r)
    assert len(result) == 0


# ? strip_nulls Preserves Valid Values
def test_strip_nulls_preserves_valid() -> None:
    r: dict[str, Any] = {"name": "BRCA1", "score": 0.05, "active": True}
    result: dict = strip_nulls(r)
    assert result["name"] == "BRCA1"
    assert result["score"] == 0.05
    assert result["active"] is True


# ? strip_nulls Handles Nested Dicts
def test_strip_nulls_nested_dict() -> None:
    r: dict[str, Any] = {"outer": {"inner": "na", "keep": "yes"}}
    result: dict = strip_nulls(r)
    assert "keep" in result["outer"]
    assert "inner" not in result["outer"]


# ? strip_nulls Handles Lists Of Dicts
def test_strip_nulls_list_of_dicts() -> None:
    r: dict[str, Any] = {"items": [{"a": "keep", "b": ""}, {"a": "also", "c": "null"}]}
    result: dict = strip_nulls(r)
    assert result["items"][0] == {"a": "keep"}
    assert result["items"][1] == {"a": "also"}


# ? strip_nulls Handles Empty Dict
def test_strip_nulls_empty_dict() -> None:
    r: dict[str, Any] = {}
    result: dict = strip_nulls(r)
    assert result == {}


# ? strip_nulls Strips Whitespace Before Check
def test_strip_nulls_whitespace() -> None:
    r: dict[str, Any] = {"a": "  ", "b": " na "}
    result: dict = strip_nulls(r)
    assert len(result) == 0


# ? label_edge Assigns UUID Under Domain
def test_label_edge_assigns_uuid() -> None:
    r: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    result: dict = label_edge(r)  # pyright: ignore
    assert "uuid" in result
    assert isinstance(result["uuid"], str)
    assert len(result["uuid"]) == 36  # ? Standard UUID string length


# ? label_edge UUID Is Deterministic
def test_label_edge_deterministic() -> None:
    r1: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    r2: dict[str, Any] = {"subject": "A", "object": "B", "predicate": "treats"}
    result1: dict = label_edge(r1)  # pyright: ignore
    result2: dict = label_edge(r2)  # pyright: ignore
    assert result1["uuid"] == result2["uuid"]


# ? label_edge Different Data Produces Different UUIDs
def test_label_edge_different_data() -> None:
    r1: dict[str, Any] = {"subject": "A", "predicate": "treats"}
    r2: dict[str, Any] = {"subject": "B", "predicate": "treats"}
    result1: dict = label_edge(r1)  # pyright: ignore
    result2: dict = label_edge(r2)  # pyright: ignore
    assert result1["uuid"] != result2["uuid"]
