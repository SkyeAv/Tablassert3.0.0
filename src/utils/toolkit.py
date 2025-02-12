from copy import deepcopy
import requests
import shutil
import gzip
import yaml
import sys
import re
import os


def fast_extension(filename: str) -> str:
    """
    Returns the file extension of the given filename (last 4 characters,
    converted to lower case).
    """
    return filename[-4:].lower()


def exit_with_error(msg: str) -> None:
    print(msg)
    sys.exit(1)


def read_config(filename: str) -> dict:
    try:
        with open(filename) as opened_yaml:
            return yaml.load(opened_yaml, Loader=yaml.CSafeLoader)
    except yaml.YAMLError as e:
        raise ValueError("YAML Error: " + str(e))


def gzip_file(path: str) -> None:
    """
    Gzip the file at the given path.

    If a gzipped version of the file already exists, remove it.
    Then, open the file in binary mode, open the gzipped file in binary-write
    mode, and copy the contents of the file to the gzipped file.
    """
    # Check if the gzipped file already exists
    if os.path.isfile(path + ".gz"):
        # Remove the existing gzipped file
        os.remove(path + ".gz")

    # Open the file in binary mode and the gzipped file in binary-write mode
    with open(path, "rb") as p_in, gzip.open(path + ".gz", "wb") as p_out:
        # Copy the contents of the file to the gzipped file
        shutil.copyfileobj(p_in, p_out)


def nice_it() -> None:
    os.nice(8)


def make_dirs(path: str) -> None:
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as e:
        raise ValueError(
            "Error creating directory: " + path + " " + str(e))


def download_file(url: str, output_path: str) -> None:
    """
    Downloads a file from the given URL, using a custom User-Agent header,
    and saves it to the given output path.

    The custom User-Agent is used to avoid being blocked by sites that
    don't like the default User-Agent used by the requests library.

    If there is an error (e.g. the URL is invalid, the file cannot be
    downloaded, etc.), the error is logged and the function
    returns without doing anything else.

    :param url: The URL of the file to download
    :param output_path: The path to save the downloaded file to
    """
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    )
    headers = {"User-Agent": user_agent}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error downloading file: {e}")
    with open(output_path, "wb") as f:
        f.write(response.content)


def get_dirname(path: str) -> str:
    return os.path.dirname(path)


def get_filename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def tsv_it(path: str) -> str:
    return path + ".tsv"


def get_sections(section_config: dict) -> list:
    """
    Gets all sections from a configuration dictionary.

    If the configuration dictionary contains a "sections" key, it is
    assumed to be a list of dictionaries, where each dictionary is a
    configuration for one section. The function will return a list of
    these section configurations, with appropriate merging.

    If the configuration dictionary does not contain a "sections" key,
    it is assumed to be a configuration for one section, and the
    function will return a list with this one section configuration.

    :param section_config: The configuration dictionary
    :return: A list of section configurations
    """
    if "sections" not in section_config:
        return [section_config]

    sections = []
    for section in section_config["sections"]:
        section_copy = deepcopy(section_config)

        # Merge the current section with the base configuration
        updates = {
            key: (
                value if key not in section_copy else (
                    section_copy[key] + value
                    if isinstance(section_copy[key], list) else (
                        section_copy[key].update(value) or section_copy[key]
                        if isinstance(section_copy[key], dict) else value)))
            for key, value in section.items()}

        section_copy.update(updates)
        sections.append(section_copy)

    return sections


def slash_cleaner(value: str) -> str:
    return re.sub(r"/", "", str(value))


def remove_file_if_exists(file: str) -> None:
    if os.path.isfile(file):
        os.remove(file)


def get_section_configs(dirs: list) -> list:
    section_configs = []
    for dir in dirs:
        for file in os.listdir(dir):
            if fast_extension(file) in ["yaml", ".yml"]:
                section_configs.append(f"{dir}/{file}")
    if len(section_configs) != 0:
        return section_configs
    else:
        exit_with_error("No section config files found")
