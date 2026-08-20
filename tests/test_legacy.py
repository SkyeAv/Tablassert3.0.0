from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from yaml.constructor import ConstructorError

from tablassert.errors import LegacyDuplicateKeyWarning
from tablassert.legacy import load_legacy_yaml


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "legacy.yaml"
    p.write_text(text)
    return p


def test_duplicate_free_config_matches_safe_load(fixtures_path: Path, recwarn: pytest.WarningsRecorder) -> None:
    """A well-formed legacy config parses identically to ``yaml.safe_load``.

    WHY: the duplicate-merge hook must be a strict no-op on files without duplicates —
    the real TableConfigs-shaped fixture is parsed to an equal object and emits no
    ``LegacyDuplicateKeyWarning``.
    """
    p = fixtures_path / "minimal_section.yaml"
    result = load_legacy_yaml(p)
    assert result == yaml.safe_load(p.read_text())
    assert [w for w in recwarn if issubclass(w.category, LegacyDuplicateKeyWarning)] == []


def test_duplicate_scalar_keys_later_wins_and_warns(tmp_path: Path) -> None:
    """Duplicate scalar keys keep the LAST value but fire exactly one warning naming key and line.

    WHY: ``yaml.safe_load`` silently drops the earlier occurrence; the legacy loader keeps the
    later-wins semantics while surfacing the data loss so a curator can fix the source file.
    """
    p = _write(tmp_path, "name: first\nversion: 1\nname: second\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `name` .+:3") as record:
        result = load_legacy_yaml(p)
    assert result == {"name": "second", "version": 1}
    assert len(record) == 1


def test_duplicate_nested_dicts_deep_merge(tmp_path: Path) -> None:
    """Duplicate keys whose values are dicts deep-merge instead of overwriting.

    WHY: legacy configs routinely split one logical section across a repeated key; a plain
    ``safe_load`` would keep only the last block and silently lose the first block's fields.
    """
    p = _write(tmp_path, "template:\n  subject:\n    method: value\ntemplate:\n  subject:\n    encoding: BRCA1\n  config: ./a.tsv\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `template` .+:4"):
        result = load_legacy_yaml(p)
    assert result == {"template": {"subject": {"method": "value", "encoding": "BRCA1"}, "config": "./a.tsv"}}


def test_duplicate_lists_extend(tmp_path: Path) -> None:
    """Duplicate keys whose values are lists concatenate first + later items.

    WHY: a repeated list key in a legacy config almost always means "more entries for the same
    field"; ``safe_load`` would keep only the last list.
    """
    p = _write(tmp_path, "encodings:\n  - BRCA1\nencodings:\n  - TP53\n  - EGFR\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `encodings` .+:3"):
        result = load_legacy_yaml(p)
    assert result == {"encodings": ["BRCA1", "TP53", "EGFR"]}


def test_duplicate_mixed_types_later_wins(tmp_path: Path) -> None:
    """A duplicate key whose value type changed keeps the later value outright.

    WHY: there is no meaningful merge across types, so the loader matches ``fastmerge``'s
    collision fallback exactly — the two merge paths can never drift.
    """
    p = _write(tmp_path, "source:\n  url:\n    - https://a.example/a.tsv\nsource: ./flat.tsv\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `source` .+:4"):
        result = load_legacy_yaml(p)
    assert result == {"source": "./flat.tsv"}


def test_duplicate_merge_does_not_mutate_aliased_value(tmp_path: Path) -> None:
    """Merging a duplicate key must not rewrite the mapping shared through a YAML alias.

    WHY: the earlier occurrence is deep-copied before the in-place merge; without that copy,
    merging into ``section`` would also rewrite ``defaults``, since the alias shares one dict.
    """
    p = _write(tmp_path, "defaults: &d\n  a: 1\nsection: *d\nsection:\n  b: 2\n")
    with pytest.warns(LegacyDuplicateKeyWarning, match=r"Duplicate key `section` .+:4"):
        result = load_legacy_yaml(p)
    assert result == {"defaults": {"a": 1}, "section": {"a": 1, "b": 2}}


def test_duplicate_loader_matches_safe_load_on_recursive_alias(tmp_path: Path) -> None:
    """A self-referential alias still yields the same self-referential dict ``safe_load`` builds.

    WHY: eager (``deep=True``) construction was a behavior risk for recursive aliases; this
    pins parity — the cached placeholder is returned for the alias, exactly as in ``safe_load``.
    """
    p = _write(tmp_path, "&a {x: *a}\n")
    result = load_legacy_yaml(p)
    expected = yaml.safe_load("&a {x: *a}\n")
    assert isinstance(result, dict)
    assert isinstance(expected, dict)
    assert result["x"] is result
    assert expected["x"] is expected


def test_duplicate_loader_rejects_unhashable_key_like_safe_load(tmp_path: Path) -> None:
    """An explicit complex key is rejected exactly as ``yaml.safe_load`` rejects it.

    WHY: the merge path must not loosen SafeLoader's error behavior for constructs that
    cannot be dict keys; parity with ``safe_load`` keeps the safety stance honest.
    """
    p = _write(tmp_path, "? [1, 2]\n: value\n")
    with pytest.raises(ConstructorError, match="found unhashable key"):
        load_legacy_yaml(p)


def test_duplicate_loader_rejects_mistagged_non_mapping_like_safe_load(tmp_path: Path) -> None:
    """A ``!!map`` tag forced onto a sequence is rejected exactly as ``yaml.safe_load`` rejects it.

    WHY: explicit tags can lie about node shape; refusing to walk a non-mapping as pairs is
    what the base constructor does, and the duplicate hook must not weaken that.
    """
    p = _write(tmp_path, "!!map [1, 2]\n")
    with pytest.raises(ConstructorError, match="expected a mapping node"):
        load_legacy_yaml(p)
