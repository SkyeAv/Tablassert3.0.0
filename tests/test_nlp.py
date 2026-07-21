from __future__ import annotations

import polars as pl

from tablassert.nlp import level_one, level_two


def test_level_one_strips_and_lowercases() -> None:
    """level one strips whitespace and lowercases."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["  Hello WORLD  ", "FOO"]}).lazy()
    result: pl.DataFrame = level_one(lf, "name").collect()
    assert result["name"].to_list() == ["hello world", "foo"]


def test_level_one_casts_integers() -> None:
    """level one casts integers to strings."""
    lf: pl.LazyFrame = pl.DataFrame({"val": [1, 2, 3]}).lazy()
    result: pl.DataFrame = level_one(lf, "val").collect()
    assert result["val"].to_list() == ["1", "2", "3"]


def test_level_one_already_clean() -> None:
    """level one handles already clean strings."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["clean"]}).lazy()
    result: pl.DataFrame = level_one(lf, "name").collect()
    assert result["name"].to_list() == ["clean"]


def test_level_one_preserves_other_columns() -> None:
    """level one preserves other columns."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["  Hello  "], "age": [42]}).lazy()
    result: pl.DataFrame = level_one(lf, "name").collect()
    assert result["age"].to_list() == [42]
    assert result["name"].to_list() == ["hello"]


def test_level_two_removes_nonword() -> None:
    """level two removes non-word characters by default."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello-world", "foo bar"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name").collect()
    assert result["name_two"].to_list() == ["helloworld", "foobar"]


def test_level_two_creates_tagged_column() -> None:
    """level two creates tagged column."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name").collect()
    assert "name_two" in result.columns
    assert "name" in result.columns


def test_level_two_custom_regex() -> None:
    """level two with custom regex."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello123world"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name", regex=r"\d+").collect()
    assert result["name_two"].to_list() == ["helloworld"]


def test_level_two_custom_tag() -> None:
    """level two with custom tag."""
    lf: pl.LazyFrame = pl.DataFrame({"name": ["hello world"]}).lazy()
    result: pl.DataFrame = level_two(lf, "name", tag="_clean").collect()
    assert "name_clean" in result.columns
    assert result["name_clean"].to_list() == ["helloworld"]
