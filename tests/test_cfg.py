# 2025 Skye Lane Goetz

from pydantic import ValidationError
from src.cfg import GraphConfig
import tempfile
import pytest
import os


def temp_sqlite() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(b"SQLite format 3\x00")
    return path


def temp_dir() -> str:
    return tempfile.mkdtemp()


def temp_file() -> str:
    fd, path = tempfile.mkstemp()
    os.close(fd)
    return path


@pytest.fixture
def valid_config() -> dict[str, object]:
    sqlite = temp_sqlite()
    directory = temp_dir()
    return {
        "name": "test_graph",
        "version": "1.0",
        "dirs": [directory],
        "workers": 4,
        "progress_handler": 0.5,
        "identification_accuracy": 0.9,
        "extraction_accuracy": 0.8,
        "cutoff": 0.7,
        "override": sqlite,
        "babel": sqlite,
        "kg2": sqlite,
        "supplement": sqlite,
        "pubmed": sqlite,
        "names": sqlite,
        "predicates": sqlite,
        "training_data": sqlite,
    }


def test_valid_config(valid_config):
    GraphConfig(**{**valid_config})


SQLITES = ["override", "babel", "kg2", "supplement", "pubmed", "names", "predicates"]


def test_nonexistent_files(valid_config):
    random_path = r"random/mc_randomface"
    with pytest.raises(ValidationError):
        GraphConfig(**{**valid_config, "training_data": random_path})
    for field in SQLITES:
        with pytest.raises(ValidationError):
            GraphConfig(**{**valid_config, field: random_path})


def test_nonsqlite_files(valid_config):
    random_file = temp_file()
    for field in SQLITES:
        with pytest.raises(ValueError, match="must be a sqlite database"):
            GraphConfig(**{**valid_config, field: random_file})


def test_empty_dirs_list(valid_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**valid_config, "dirs": []})


def test_incorrect_quantity_of_workers(valid_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**valid_config, "workers": 0})
        GraphConfig(**{**valid_config, "workers": 2.5})
        GraphConfig(**{**valid_config, "workers": 17})


def test_too_large_floats(valid_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**valid_config, "identification_accuracy": 2.2})
        GraphConfig(**{**valid_config, "extraction_accuracy": 2.2})
        GraphConfig(**{**valid_config, "cutoff": 2.2})


def test_negative_floats(valid_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**valid_config, "progress_handler": -0.05})
        GraphConfig(**{**valid_config, "identification_accuracy": -0.05})
        GraphConfig(**{**valid_config, "extraction_accuracy": -0.05})
        GraphConfig(**{**valid_config, "cutoff": -0.05})


def test_string_casting(valid_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**valid_config, "version": 1.0})


def test_float_casting(valid_config):
    GraphConfig(**{**valid_config, "progress_handler": 1})
    GraphConfig(**{**valid_config, "progress_handler": "1"})
