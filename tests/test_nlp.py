from __future__ import annotations

import polars as pl

from tablassert.nlp import level_one, level_two


# ? Level One Strips Whitespace And Lowercases
def test_level_one_strips_and_lowercases() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["  Hello WORLD  ", "FOO"]}).lazy()
    result: pl.DataFrame = level_one(lf, "name").collect()
    assert result["name"].to_list() == ["hello world", "foo"]


# ? Level One Casts Integers To Strings
def test_level_one_casts_integers() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"val": [1, 2, 3]}).lazy()
    result: pl.DataFrame = level_one(lf, "val").collect()
    assert result["val"].to_list() == ["1", "2", "3"]


# ? Level One Handles Already Clean Strings
def test_level_one_already_clean() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["clean"]}).lazy()
    result: pl.DataFrame = level_one(lf, "name").collect()
    assert result["name"].to_list() == ["clean"]


# ? Level One Preserves Other Columns
def test_level_one_preserves_other_columns() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["  Hello  "], "age": [42]}).lazy()
    result: pl.DataFrame = level_one(lf, "name").collect()
    assert result["age"].to_list() == [42]
    assert result["name"].to_list() == ["hello"]


# ? Level Two Removes Non-Word Characters By Default
def test_level_two_removes_nonword() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello-world", "foo bar"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name").collect()
    assert result["name two"].to_list() == ["helloworld", "foobar"]


# ? Level Two Creates Tagged Column
def test_level_two_creates_tagged_column() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name").collect()
    assert "name two" in result.columns
    assert "name" in result.columns


# ? Level Two With Custom Regex
def test_level_two_custom_regex() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello123world"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name", regex=r"\d+").collect()
    assert result["name two"].to_list() == ["helloworld"]


# ? Level Two With Custom Tag
def test_level_two_custom_tag() -> None:
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello world"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name", tag="_clean").collect()
    assert "name_clean" in result.columns
    assert result["name_clean"].to_list() == ["helloworld"]
