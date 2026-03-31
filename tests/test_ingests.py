from __future__ import annotations

from pathlib import Path
from typing import Any

from tablassert.ingests import fastmerge, from_yaml, to_sections


# ? fastmerge Merges Nested Dicts
def test_fastmerge_nested_dicts() -> None:
    a: dict[str, Any] = {"x": 1, "inner": {"a": 1}}
    b: dict[str, Any] = {"y": 2, "inner": {"b": 2}}
    result: dict[str, Any] = fastmerge(a, b)
    assert result == {"x": 1, "y": 2, "inner": {"a": 1, "b": 2}}


# ? fastmerge Overwrites Scalar Values
def test_fastmerge_overwrites_scalars() -> None:
    a: dict[str, Any] = {"x": 1}
    b: dict[str, Any] = {"x": 2}
    result: dict[str, Any] = fastmerge(a, b)
    assert result == {"x": 2}


# ? fastmerge Extends Lists
def test_fastmerge_extends_lists() -> None:
    a: list[int] = [1, 2]
    b: list[int] = [3, 4]
    result: list[int] = fastmerge(a, b)
    assert result == [1, 2, 3, 4]


# ? fastmerge Adds New Keys From B
def test_fastmerge_adds_new_keys() -> None:
    a: dict[str, Any] = {"x": 1}
    b: dict[str, Any] = {"y": 2}
    result: dict[str, Any] = fastmerge(a, b)
    assert "y" in result
    assert result["y"] == 2


# ? fastmerge Handles List Valued Keys
def test_fastmerge_dict_with_list_values() -> None:
    a: dict[str, Any] = {"items": [1]}
    b: dict[str, Any] = {"items": [2]}
    result: dict[str, Any] = fastmerge(a, b)
    assert result["items"] == [1, 2]


# ? fastmerge Returns B When Types Differ
def test_fastmerge_returns_b_on_type_mismatch() -> None:
    a: dict[str, Any] = {"x": "string"}
    b: dict[str, Any] = {"x": 42}
    result: dict[str, Any] = fastmerge(a, b)
    assert result["x"] == 42


# ? fastmerge Modifies A In Place
def test_fastmerge_in_place() -> None:
    a: dict[str, Any] = {"x": 1}
    b: dict[str, Any] = {"y": 2}
    result: dict[str, Any] = fastmerge(a, b)
    assert result is a


# ? from_yaml Reads File To Dict
def test_from_yaml_reads_file(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section.yaml")
    assert isinstance(data, dict)
    assert "syntax" in data
    assert "source" in data


# ? to_sections Expands Template With Sections
def test_to_sections_expands_template(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section_with_sections.yaml")
    table: Path = Path("test_config.yaml")
    sections: list[list[dict[str, Any]]] = to_sections(data, table)
    assert len(sections) == 2


# ? to_sections Merges Template Into Each Section
def test_to_sections_merges_template(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section_with_sections.yaml")
    table: Path = Path("test_config.yaml")
    sections: list[list[dict[str, Any]]] = to_sections(data, table)
    # ? Each section should inherit provenance from template
    for section in sections:
        assert "provenance" in section


# ? to_sections Adds Config Path
def test_to_sections_adds_config(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section_with_sections.yaml")
    table: Path = Path("test_config.yaml")
    sections: list[list[dict[str, Any]]] = to_sections(data, table)
    for section in sections:
        assert section["config"] == table  # pyright: ignore


# ? to_sections Each Section Has Unique Object
def test_to_sections_unique_objects(fixtures_path: Path) -> None:
    data: Any = from_yaml(fixtures_path / "minimal_section_with_sections.yaml")
    table: Path = Path("test_config.yaml")
    sections: list[list[dict[str, Any]]] = to_sections(data, table)
    # ? Sections override object encoding from the sections list
    obj_encodings: list[str] = []
    for section in sections:
        obj_encoding: Any = section["statement"]["object"]["encoding"]  # pyright: ignore
        obj_encodings.append(obj_encoding)
    assert obj_encodings[0] != obj_encodings[1]
