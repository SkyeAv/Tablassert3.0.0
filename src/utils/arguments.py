from src.utils import toolkit


def check_len_args(args: list) -> None:
    if not args or len(args) > 1:
        raise IndexError("Invalid number of arguments")


def check_config_extension(args: list) -> None:
    if toolkit.fast_extension(args[0]) not in ["yaml", ".yml"]:
        raise ValueError("Invalid config file extension")


def enforce_usage(args: list) -> None:
    """
    Checks that the arguments passed to the script are valid.
    If the arguments are invalid, prints an error message and exits.
    """
    try:
        check_len_args(args)
        check_config_extension(args)
    except (IndexError, ValueError) as e:
        # Print a helpful error message and exit
        toolkit.exit_with_error(
            str(e) +
            "\nSee README.md for usage\n" +
            "Usage: python3 tablassert3everything.py <kg_config_file>\n" +
            "Where <kg_config_file> is the path to a yaml configuration file")
