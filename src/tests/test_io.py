__author__ = "Skye Lane Goetz"
__status__ = "Development"


from tablassert.io import load_sections, get_root, get_true_root
from importlib.metadata import PackageNotFoundError
from tablassert import io
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


def test_get_true_root_finds_identifier(tmp_path):
    get_root.cache_clear()
    get_true_root.cache_clear()
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[tool.poetry]")
    sub_dir = project_root / "subdir" / "nested"
    sub_dir.mkdir(parents=True)
    result = get_true_root(sub_dir)
    assert result == project_root


def test_get_true_root_raises_if_identifier_not_found(tmp_path):
    get_root.cache_clear()
    get_true_root.cache_clear()
    dir_without_toml = tmp_path / "no_toml"
    dir_without_toml.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        get_true_root(dir_without_toml)


def test_get_root_package(monkeypatch, tmp_path):
    get_root.cache_clear()
    get_true_root.cache_clear()
    fake_package_root = tmp_path / "tablassert"
    fake_package_root.mkdir(parents=True)
    (fake_package_root / "pyproject.toml").write_text("")
    nested_dir = fake_package_root / "subpackage"
    nested_dir.mkdir(parents=True)
    dummy_file = nested_dir / "dummy.py"
    dummy_file.write_text("# test")

    class DummyFile:
        def locate(self):
            return dummy_file

    class DummyFiles:
        def __getitem__(self, index):
            if index == 0:
                return DummyFile()
            raise IndexError()

    def fake_files(package):
        assert package == "tablassert"
        return DummyFiles()

    monkeypatch.setattr(io, "files", fake_files)
    monkeypatch.setattr(io, "__file__", str(dummy_file))
    root = get_root("tablassert")
    assert root == fake_package_root.resolve().as_posix()


def test_get_root_fallback(monkeypatch, tmp_path):
    get_root.cache_clear()
    get_true_root.cache_clear()
    fallback_root = tmp_path / "fallback"
    fallback_root.mkdir()
    (fallback_root / "pyproject.toml").write_text("")

    def fake_files(_):
        raise PackageNotFoundError()

    monkeypatch.setattr(io, "files", fake_files)
    monkeypatch.setattr(io, "__file__", str(fallback_root / "fake.py"))
    root = get_root("tablassert")
    assert root == fallback_root.resolve().as_posix()
