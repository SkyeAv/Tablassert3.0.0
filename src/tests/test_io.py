# 2025 Skye Lane Goetz

from tablassert.io import load_sections
from pathlib import Path
import tempfile
import pytest
import yaml


@pytest.fixture
def eg_dirs(eg_table_config) -> list[str]:
    with tempfile.TemporaryDirectory() as dir1:
        with tempfile.TemporaryDirectory() as dir2:
            yaml1 = Path(dir1) / "valid1.yaml"
            yaml1.write_text(yaml.dump(eg_table_config))
            yaml2 = Path(dir2) / "valid2.yaml"
            yaml2.write_text(yaml.dump(eg_table_config))
            invalid = Path(dir1) / "invalid.yaml"
            invalid.write_text("not: valid: yaml")
            empty = Path(dir2) / "empty_dir"
            empty.mkdir()
            subdir = Path(dir2) / "sub"
            subdir.mkdir()
            yaml3 = subdir / "valid3.yaml"
            yaml3.write_text(yaml.dump(eg_table_config))
            yield [dir1, dir2]


def test_eg_dirs_load_sections(eg_dirs, capsys):
    result = load_sections(eg_dirs)
    assert len(result) == 3
    assert all(isinstance(section, dict) for section in result)
    captured = capsys.readouterr()
    assert "✓ Loaded valid1.yaml" in captured.out
    assert "✓ Loaded valid2.yaml" in captured.out
    assert "✓ Loaded valid3.yaml" in captured.out
    assert "Loaded 3 sections" in captured.out
