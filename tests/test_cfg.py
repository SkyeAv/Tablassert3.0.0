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
def eg_graph_config() -> dict[str, object]:
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


def test_eg_graph_config(eg_graph_config):
    GraphConfig(**{**eg_graph_config})


SQLITES = ["override", "babel", "kg2", "supplement", "pubmed", "names", "predicates"]


def test_nonexistent_files(eg_graph_config):
    random_path = r"random/mc_randomface"
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "training_data": random_path})
    for field in SQLITES:
        with pytest.raises(ValidationError):
            GraphConfig(**{**eg_graph_config, field: random_path})


def test_nonsqlite_files(eg_graph_config):
    random_file = temp_file()
    for field in SQLITES:
        with pytest.raises(ValueError, match="must be a sqlite database"):
            GraphConfig(**{**eg_graph_config, field: random_file})


def test_empty_dirs_list(eg_graph_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "dirs": []})


def test_incorrect_quantity_of_workers(eg_graph_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "workers": 0})
        GraphConfig(**{**eg_graph_config, "workers": 2.5})
        GraphConfig(**{**eg_graph_config, "workers": 17})


def test_too_large_floats(eg_graph_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "identification_accuracy": 2.2})
        GraphConfig(**{**eg_graph_config, "extraction_accuracy": 2.2})
        GraphConfig(**{**eg_graph_config, "cutoff": 2.2})


def test_negative_floats(eg_graph_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "progress_handler": -0.05})
        GraphConfig(**{**eg_graph_config, "identification_accuracy": -0.05})
        GraphConfig(**{**eg_graph_config, "extraction_accuracy": -0.05})
        GraphConfig(**{**eg_graph_config, "cutoff": -0.05})


def test_string_casting(eg_graph_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "version": 1.0})


def test_float_casting(eg_graph_config):
    GraphConfig(**{**eg_graph_config, "progress_handler": 1})
    GraphConfig(**{**eg_graph_config, "progress_handler": "1"})
