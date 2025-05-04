__author__ = "Skye Lane Goetz"


from tablassert.cfg import GraphConfig, TableConfig
from pydantic import ValidationError
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
        "preds": sqlite,
        "training_data": sqlite,
    }


def test_eg_graph_config(eg_graph_config):
    GraphConfig(**{**eg_graph_config})


def test_missing_graph_subconfigs(eg_graph_config):
    with pytest.raises(ValidationError):
        GraphConfig(**{**eg_graph_config, "name": None})
        GraphConfig(**{**eg_graph_config, "version": None})
        GraphConfig(**{**eg_graph_config, "workers": None})
        GraphConfig(**{**eg_graph_config, "progress_handler": None})
        GraphConfig(**{**eg_graph_config, "identification_accuracy": None})
        GraphConfig(**{**eg_graph_config, "extraction_accuracy": None})
        GraphConfig(**{**eg_graph_config, "cutoff": None})
        GraphConfig(**{**eg_graph_config, "override": None})
        GraphConfig(**{**eg_graph_config, "babel": None})
        GraphConfig(**{**eg_graph_config, "kg2": None})
        GraphConfig(**{**eg_graph_config, "supplement": None})
        GraphConfig(**{**eg_graph_config, "pubmed": None})
        GraphConfig(**{**eg_graph_config, "names": None})
        GraphConfig(**{**eg_graph_config, "preds": None})
        GraphConfig(**{**eg_graph_config, "training_data": None})


SQLITES = ["override", "babel", "kg2", "supplement", "pubmed", "names", "preds"]


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
        with pytest.raises(ValidationError):
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


def test_eg_table_config(eg_table_config):
    TableConfig(**{**eg_table_config})


def test_eg_reinxeding(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "reindexing": [
                        {
                            "when": "before",
                            "mode": "ne",
                            "column": "A",
                            "value": "string",
                        }
                    ],
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "reindexing": [
                        {"when": "AftEr", "mode": "eq", "column": "BC", "value": 0.05}
                    ],
                }
            ],
        }
    )


def test_erroneous_string_values_reindexing(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "reindexing": [
                            {
                                "when": "after",
                                "mode": "gt",
                                "column": "BC",
                                "value": "string",
                            }
                        ],
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "reindexing": [
                            {
                                "when": "after",
                                "mode": "le",
                                "column": "BC",
                                "value": "string",
                            }
                        ],
                    }
                ],
            }
        )


def test_erroneous_when_reindexing(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "reindexing": [
                            {
                                "when": "a",
                                "mode": "gt",
                                "column": "BC",
                                "value": "string",
                            }
                        ],
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "reindexing": [
                            {
                                "when": "_after",
                                "mode": "le",
                                "column": "BC",
                                "value": "string",
                            }
                        ],
                    }
                ],
            }
        )


def test_erroneous_mode_reindexing(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "reindexing": [
                            {
                                "when": "after",
                                "mode": "gtr",
                                "column": "BC",
                                "value": "string",
                            }
                        ],
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "reindexing": [
                            {
                                "when": "before",
                                "mode": "lq",
                                "column": "BC",
                                "value": "string",
                            }
                        ],
                    }
                ],
            }
        )


def test_missing_table_subconfigs(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [{**eg_table_config["sections"][0], "provenance": None}],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [{**eg_table_config["sections"][0], "location": None}],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [{**eg_table_config["sections"][0], "triple": None}],
            }
        )


def test_improper_urls(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "download_from": "file:///Users/username/Documents/report.docx",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "download_from": "/Users/username/Documents/report.docx",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "download_from": "report.docx",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "download_from": "random string",
                        },
                    }
                ],
            }
        )


def test_random_madeup_extensions(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                **eg_table_config["sections"][0]["location"]["params"],
                                "ext": "yml",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                **eg_table_config["sections"][0]["location"]["params"],
                                "ext": "gabagool",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                **eg_table_config["sections"][0]["location"]["params"],
                                "ext": "christopher",
                            },
                        },
                    }
                ],
            }
        )


def test_random_mismatching_location_params(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                **eg_table_config["sections"][0]["location"]["params"],
                                "ext": "xlsx",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                **eg_table_config["sections"][0]["location"]["params"],
                                "ext": "xls",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                **eg_table_config["sections"][0]["location"]["params"],
                                "ext": "pdf",
                            },
                        },
                    }
                ],
            }
        )


def test_xlsx_location_params(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "xlsx", "sheet": "sheet1", "start": 20},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "xlsx", "sheet": "sheet1", "end": 2},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {
                            "ext": "xlsx",
                            "sheet": "sheet1",
                            "start": 12,
                            "end": 1738,
                        },
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {
                            "ext": "xlsx",
                            "sheet": "sheet1",
                            "rows": [1, 2, 56, 78],
                        },
                    },
                }
            ],
        }
    )


def test_erroneous_xlsx_location_params(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {"ext": "xlsx", "sheet": "sheet1", "start": 0},
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {"ext": "xlsx", "sheet": "sheet1", "end": 2},
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {"ext": "xlsx", "sheet": "sheet1"},
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                "ext": "xlsx",
                                "sheet": "sheet1",
                                "start": 1,
                                "end": 2,
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                "ext": "xlsx",
                                "sheet": "sheet1",
                                "start": 1,
                                "end": 2,
                                "rows": [5, 6, 7],
                            },
                        },
                    }
                ],
            }
        )


def test_text_based_image_location_params(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "pdf", "pages": 1, "flavor": "stream"},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "pdf", "pages": "1-5", "flavor": "stream"},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "pdf", "pages": "1,2,3", "flavor": "lattice"},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {
                            "ext": "pdf",
                            "pages": "1,2,4-10",
                            "flavor": "lattice",
                        },
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "pdf", "pages": None, "flavor": "lattice"},
                    },
                }
            ],
        }
    )


def test_erroneous_text_based_image_location_params(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                "ext": "pdf",
                                "pages": "1-10-100",
                                "flavor": "stream",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {"ext": "pdf", "pages": "1", "flavor": "steam"},
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {
                                "ext": "pdf",
                                "pages": "1-20",
                                "flavor": "lettuce",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {"ext": "pdf", "pages": "1-20"},
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": {"ext": "pdf", "pages": None},
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "location": {
                            **eg_table_config["sections"][0]["location"],
                            "params": None,
                        },
                    }
                ],
            }
        )


def test_delimited_file_location_params(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "csv", "delimiter": 9, "start": 1},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "csv", "delimiter": 9.0, "start": 1},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "tsv", "delimiter": "\t", "start": 1},
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "location": {
                        **eg_table_config["sections"][0]["location"],
                        "params": {"ext": "txt", "delimiter": "\t", "start": 1},
                    },
                }
            ],
        }
    )


def test_different_provenance_curies(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "provenance": {
                        **eg_table_config["sections"][0]["provenance"],
                        "publication_id": "PMID:01303890",
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "provenance": {
                        **eg_table_config["sections"][0]["provenance"],
                        "publication_id": "doi:0130aj389/si0",
                    },
                }
            ],
        }
    )


def test_erroneous_provenance_curries(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "provenance": {
                            **eg_table_config["sections"][0]["provenance"],
                            "publication_id": "doi/:0130aj389/si0",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "provenance": {
                            **eg_table_config["sections"][0]["provenance"],
                            "publication_id": "doi/0130aj389/si0",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "provenance": {
                            **eg_table_config["sections"][0]["provenance"],
                            "publication_id": "PMC: 2848474091 ",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "provenance": {
                            **eg_table_config["sections"][0]["provenance"],
                            "publication_id": "0130aj389/si0:doi",
                        },
                    }
                ],
            }
        )


def test_eg_attributes(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "attributes": {
                        "sample_size": {"mode": "column", "value": "A"},
                        "p_value": {"mode": "column", "value": "B"},
                        "fdr": {"mode": "predefined", "value": "string"},
                        "strength": {"mode": "column", "value": "C"},
                        "statistics": {"mode": "predefined", "value": "string"},
                        "notes": "string",
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "attributes": {
                        "sample_size": {"mode": "column", "value": "A"},
                        "p_value": {"mode": "column", "value": "B"},
                        "fdr": {"mode": "predefined", "value": "string"},
                        "strength": {"mode": "column", "value": "C"},
                        "statistics": {"mode": "predefined", "value": "string"},
                        "notes": "string",
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "attributes": {
                        "sample_size": {"mode": "column", "value": "A"},
                        "p_value": {"mode": "column", "value": "B"},
                        "fdr": {"mode": "predefined", "value": "string"},
                        "strength": {"mode": "column", "value": "C"},
                        "statistics": {"mode": "predefined", "value": "string"},
                        "notes": 909,
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "attributes": {
                        "sample_size": {
                            "mode": "column",
                            "value": "A",
                            "math": [
                                {
                                    "attr": "pow",
                                    "args": [None, -2],
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "attributes": {
                        "sample_size": {
                            "mode": "column",
                            "value": "A",
                            "math": [
                                {
                                    "attr": "ceil",
                                    "args": [None],
                                }
                            ],
                        },
                    },
                }
            ],
        }
    )


def test_incorrect_column_names(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "attributes": {
                            "sample_size": {"mode": "column", "value": "string"},
                            "p_value": {"mode": "column", "value": "string"},
                            "fdr": {"mode": "column", "value": "string"},
                            "strength": {"mode": "column", "value": "string"},
                            "statistics": {"mode": "column", "value": "string"},
                            "notes": "string",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "attributes": {
                            "sample_size": {"mode": "predefined", "value": "string"},
                            "p_value": {"mode": "predefined", "value": "string"},
                            "fdr": {"mode": "predefined", "value": "string"},
                            "strength": {"mode": "predefined", "value": "string"},
                            "statistics": {"mode": "column", "value": "string"},
                            "notes": "string",
                        },
                    }
                ],
            }
        )


def test_fake_math_attributes(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "attributes": {
                            "sample_size": {
                                "mode": "column",
                                "value": "A",
                                "math": [
                                    {
                                        "attr": "wonderlaw",
                                        "args": [None, 9],
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "attributes": {
                            "sample_size": {
                                "mode": "column",
                                "value": "A",
                                "math": [
                                    {
                                        "attr": "9+10",
                                        "args": [None, 21],
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "attributes": {
                            "sample_size": {
                                "mode": "column",
                                "value": "A",
                                "math": [
                                    {
                                        "attr": "Jennifer",
                                        "args": [None, -3],
                                    }
                                ],
                            },
                        },
                    }
                ],
            }
        )


def test_fleshed_out_triples(eg_table_config):
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "triple": {
                        **eg_table_config["sections"][0]["triple"],
                        "subj": {
                            "mode": "sccurie",
                            "value": "A",
                            "in_organism": "NCBITaxon:9606",
                            "remove": ["x", "y", "z"],
                            "prefix": "PREFIX:",
                            "suffix": "+suffix",
                            "cfill": "forward",
                            "regex": [{"pattern": r"\W", "replacement": "WOLLONGONG"}],
                            "dexplode": ",",
                        },
                    },
                }
            ],
        }
    )
    TableConfig(
        **{
            **eg_table_config,
            "sections": [
                {
                    **eg_table_config["sections"][0],
                    "triple": {
                        **eg_table_config["sections"][0]["triple"],
                        "subj": {
                            "mode": "value",
                            "value": "string",
                            "in_organism": "NCBITaxon:9606",
                            "remove": ["x", "y", 1],
                            "prefix": "PREFIX:",
                            "suffix": "+suffix",
                            "cfill": "forward",
                            "regex": [{"pattern": r"\W", "replacement": "WOLLONGONG"}],
                            "dexplode": 9,
                        },
                    },
                }
            ],
        }
    )


def test_erroneous_fleshed_out_triples(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "subj": {
                                "mode": "value",
                                "value": "string",
                                "in_organism": "9606",
                                "remove": ["x", "y", 1],
                                "prefix": "PREFIX:",
                                "suffix": "+suffix",
                                "cfill": "forward",
                                "regex": [
                                    {"pattern": r"\W", "replacement": "WOLLONGONG"}
                                ],
                                "dexplode": 9,
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "subj": {
                                "cfill": "fake",
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "subj": {
                                "prefix": ["fake"],
                            },
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "subj": {
                                "remove": "X",
                            },
                        },
                    }
                ],
            }
        )


def test_erroneous_predicates(eg_table_config):
    with pytest.raises(ValidationError):
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "pred": "string",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "pred": "string:string",
                        },
                    }
                ],
            }
        )
        TableConfig(
            **{
                **eg_table_config,
                "sections": [
                    {
                        **eg_table_config["sections"][0],
                        "triple": {
                            **eg_table_config["sections"][0]["triple"],
                            "pred": "biolink:19209",
                        },
                    }
                ],
            }
        )
