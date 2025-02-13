from src.utils import toolkit
import sqlite3
import os


def check_kg_config(kg_config: object) -> None:
    if not isinstance(kg_config, dict):
        raise ValueError("Invalid kg_config: Should be a dictionary")


def check_kg_subconfigs(kg_config: dict) -> None:
    subconfigs = [
        "knowledge_graph_name", "version_number", "max_workers",
        "p_value_cutoff", "config_directories", "override_sqlite",
        "supplement_sqlite", "babel_sqlite", "kg2_sqlite",
        "progress_handler_timeout", "predicates_sqlite",
        "confidence_model", "tfidf_vectorizer"]
    if not all(field in kg_config.keys() for field in subconfigs):
        raise ValueError(
            f"Invalid kg_config: {subconfigs} not found")


def check_str(x: object, attribute: str) -> None:
    if not isinstance(x, str):
        raise ValueError(f"Invalid {attribute}: Should be an string")


def check_int(x: object, attribute: str) -> None:
    if not isinstance(x, int):
        raise ValueError(f"Invalid {attribute}: Should be an integer")


def check_float(x: object, attribute: str) -> None:
    if not isinstance(x, float):
        raise ValueError(f"Invalid {attribute}: Should be a float")


def check_pkl(x: object, attribute: str) -> None:
    if not os.path.isfile(x):
        raise ValueError(f"Invalid {attribute}: {x} is not a file")
    if toolkit.fast_extension(x) != ".pkl":
        raise ValueError(f"Invalid {attribute}: {x} is not a \"pkl\" file")


def check_config_directories(config_directories: object) -> None:
    if not isinstance(config_directories, list):
        raise ValueError("Invalid config_directories: Should be a list")
    for i, config_directory in enumerate(config_directories):
        if not os.path.isdir(config_directory):
            raise ValueError(
                f"Invalid config_directory {i}: Should be a directory")


def is_valid_db(sqlite: str) -> None:
    try:
        conn = sqlite3.connect(sqlite)
        conn.close()
    except sqlite3.OperationalError as e:
        raise ValueError("Invalid sqlite: " + sqlite + " " + str(e))


def check_sqlites(kg_config: dict) -> None:
    for sqlite in [
            "override_sqlite", "supplement_sqlite",
            "babel_sqlite", "kg2_sqlite",
            "predicates_sqlite"]:
        try:
            is_valid_db(kg_config[sqlite])
        except ValueError as e:
            raise ValueError(str(e))


def parse_kg_config(kg_config: object) -> None:
    """
    Checks that the given kg_config is valid.
    If the config is invalid, raises an error and exits.
    """
    try:
        # Check that kg_config is a dictionary
        check_kg_config(kg_config)

        # Check that the required subconfigs are present
        check_kg_subconfigs(kg_config)

        # Check that version_number is a string
        check_str(kg_config["version_number"], "version_number")

        # Check that max_workers is an integer
        check_int(kg_config["max_workers"], "max_workers")

        # Check that p_value_cutoff is a float
        check_float(kg_config["p_value_cutoff"], "p_value_cutoff")

        # Check that progress_handler_timeout is a float
        check_float(
            kg_config["progress_handler_timeout"], "progress_handler_timeout")

        # Check that config_directories is a list of directories
        check_config_directories(kg_config["config_directories"])

        # Check that the pkl files are valid
        check_pkl(kg_config["confidence_model"], "confidence_model")
        check_pkl(kg_config["tfidf_vectorizer"], "tfidf_vectorizer")

        # Check that the sqlite databases are valid
        check_sqlites(kg_config)

    except ValueError as e:
        # Print an error message and exit
        toolkit.exit_with_error(e)
