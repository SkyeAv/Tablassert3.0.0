from src.utils import toolkit
import requests


def check_section(section: object) -> None:
    if not section:
        raise ValueError("Invalid section config")


def section_subconfig_check(section: dict) -> None:
    section_subconfigs = [
        "column_style", "data_location", "provenance", "domain",
        "subject", "predicate", "object", "attributes", "method_notes"]
    if not all(field in section.keys() for field in section_subconfigs):
        raise ValueError(
            f"Invalid section config: {section_subconfigs} not found")


def check_column_style(column_style: object) -> None:
    if not isinstance(column_style, str):
        raise ValueError("Invalid column_style")


def check_data_location(data_location: str) -> None:
    if not toolkit.fast_extension(data_location) in \
            [".csv", ".tsv", ".txt", "xlsx", ".xls"]:
        raise ValueError("Invalid data_location")


def data_location_case_ladder(data_location: str, subconfig_keys: list) -> str:
    """
    Checks that the proper subconfigs are present for the given data location.

    Args:
        data_location (str): the path to the data file
        subconfig_keys (list): A list of subconfigs present in the data
            location configuration

    Raises:
        ValueError: If the data location configuration is invalid
    """
    # Check the subconfigs for csv/tsv/txt files
    match toolkit.fast_extension(data_location):
        case ".csv" | ".tsv" | ".txt":
            if "delimiter" not in subconfig_keys:
                raise ValueError(
                    "Invalid data_location: [\"delimiter\"] not found")
        # Check the subconfigs for xlsx/xls files
        case "xlsx" | ".xls":
            excel_subconfigs = "sheet_to_use", "first_line", "last_line"
            if not all(field in subconfig_keys for field in excel_subconfigs):
                raise ValueError(
                    f"Invalid data_location: {excel_subconfigs} not found")


def check_download_url(url: str) -> None:
    try:
        response = requests.head(url)
        if response.status_code >= 400 and response.status_code not in [
                403, 502]:
            raise ValueError(f"Invalid URL: {url}")
    except requests.RequestException as e:
        raise RuntimeError(f"Error checking URL: {e}")


def provenance_subconfig_check(provenance: dict) -> None:
    subconfigs = [
        "publication", "journal", "publication_name", "authors",
        "year_published", "table_url", "yaml_curator", "curator_organization"]
    if not all(field in provenance.keys() for field in subconfigs):
        raise ValueError(
            f"Invalid section config: {subconfigs} not found")


def check_predicate(predicate: object) -> None:
    if not isinstance(predicate, str) and "biolink" in predicate:
        raise ValueError("Invalid predicate")


def check_node_options(node: dict) -> None:
    options = [
        "curie", "value", "curie_column_name", "value_column_name",
        "shared_value_column", "shared_curie_column"]
    if not any(field in options for field in node.keys()):
        raise ValueError(f"Invalid node config: one of {options} not found")


def check_fill_methods(fill_method: str) -> None:
    options = ["pad", "ffill", "backfill", "bfill"]
    if fill_method not in options:
        raise ValueError(f"Invalid fill_method: one of {options} not found")


def check_regex_replacements(regex_replacements: list) -> None:
    for i, regex_replacement in enumerate(regex_replacements):
        subconfigs = ["pattern", "replacement"]
        if not all(field in regex_replacement.keys() for field in subconfigs):
            raise ValueError(
                f"Invalid regex_replacement {i}: {subconfigs} not found")


def check_expected_classes(expected_classes: object) -> None:
    if not isinstance(expected_classes, list):
        raise ValueError("Invalid expected_classes: Should be a list")
    for i, expected_class in enumerate(expected_classes):
        if "biolink:" not in expected_class:
            raise ValueError(
                f"Invalid classes {i}: Should be a Biolink class")


def check_prefixes(prefixes: object) -> None:
    if not isinstance(prefixes, list):
        raise ValueError("Invalid prefixes: Should be a list")
    for i, prefix in enumerate(prefixes):
        if not isinstance(prefix, dict):
            raise ValueError(
                f"Invalid prefix {i}: Should be a dictionary")
        if "prefix" not in prefix.keys():
            raise ValueError(
                f"Invalid prefix {i}: Should include a prefix field")


def check_attributes(attributes: dict) -> None:
    subconfigs = [
        "n", "p", "relationship_strength", "relationship_type",
        "p_correction_method", "knowledge_level", "agent_type"]
    if not all(field in attributes.keys() for field in subconfigs):
        raise ValueError(
            f"Invalid attributes: {subconfigs} not found")


def check_p_subconfig(p_subconfig: dict) -> None:
    if "value" in p_subconfig.keys():
        if not isinstance(p_subconfig["value"], float) \
                and not p_subconfig["value"] == "not_applicable" \
                and not isinstance(p_subconfig["value"], int):
            raise ValueError("Invalid p_subconfig value: Should be a float")


def check_math_subconfig(math_subconfig: object) -> None:
    if not isinstance(math_subconfig, list):
        raise ValueError("Invalid math_subconfig: Should be a list")
    for i, config in enumerate(math_subconfig):
        if "operation" not in config.keys():
            raise ValueError(
                f"Invalid math {i}: Should include an operation field")
        if "parameter" in config.keys() or "order_last" in config.keys():
            if not all(
                    field in config.keys() for field in [
                        "parameter", "order_last"]):
                raise ValueError(
                    f"""Invalid math {i}: Should include both a
                        parameter and order_last field""")


def check_sections(sections: object) -> None:
    if not isinstance(sections, list):
        raise ValueError("Invalid sections: Should be a list")
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError(
                f"Invalid section {i}: Subsections should be dictionaries")


def check_reindexing(reindexing: object) -> None:
    if not isinstance(reindexing, list):
        raise ValueError("Invalid reindexing: Should be a list")
    for i, reindex in enumerate(reindexing):
        subconfigs = ["mode", "column", "value"]
        if not all(
                field in reindex.keys() for field in subconfigs):
            raise ValueError(
                f"Invalid reindex {i}: {subconfigs} not found")
        if not reindex["mode"] in [
                    "greater_than_or_equal_to",
                    "less_than_or_equal_to", "not_equal_to", "equal_to"]:
            raise ValueError(
                f"""Invalid reindex {i}: mode should be one of
                greater_than_or_equal_to, less_than_or_equal_to,
                not_equal_to""")
        if isinstance(reindex["value"], str) \
                and reindex["mode"] not in ["not_equal_to", "equal_to"]:
            raise ValueError(
                f"""
                    Invalid reindex {i}: mode cannot be
                    anything other than not_equal_to if value is a string""")


def check_expected_taxa(expected_taxa: object) -> None:
    if not isinstance(expected_taxa, list):
        raise ValueError("Invalid expected_taxa: Should be a list")
    for i, taxon in enumerate(expected_taxa):
        if not isinstance(taxon, str):
            raise ValueError(
                f"Invalid taxon {i}: Should be a string")
        if "NCBITaxon:" not in taxon:
            raise ValueError(
                f"Invalid taxon {i}: Should be a \"NCBITaxon:\" CURIE")


def check_domain(domain: object) -> None:
    if not isinstance(domain, list):
        raise ValueError("Invalid domain: Should be a list")
    for i, d in enumerate(domain):
        if "MESH:" not in d:
            raise ValueError(
                f"Invalid domain {i}: Should be a \"MESH:\" CURIE")


def parse_section(i: int, section: object) -> None:
    """
    Checks that the given sections are valid.

    If any of the sections are invalid, raises an error and exits.

    :param i: The index of the section being checked
    :param section: The section configuration being checked
    """
    try:
        # Check that the section is a dictionary
        check_section(section)
        section_subconfig_check(section)

        # Check that the column style is valid
        check_column_style(section["column_style"])

        # Check that the data location is valid
        check_data_location(section["data_location"]["path_to_file"])
        data_location_case_ladder(
            section["data_location"]["path_to_file"],
            section["data_location"].keys())

        # Check that the provenance is valid
        provenance_subconfig_check(section["provenance"])
        check_download_url(section["provenance"]["table_url"])

        # Check that the predicate is valid
        check_predicate(section["predicate"])

        # Check that the subject and object are valid
        for field in ["subject", "object"]:
            # Check that the node options are valid
            check_node_options(section[field])
            # Check for fill method and regex replacements
            if "fill_method" in section[field].keys():
                check_fill_methods(section[field]["fill_method"])
            if "regex_replacements" in section[field].keys():
                check_regex_replacements(
                    section[field]["regex_replacements"])
            # Check that the expected classes, avoid, and prefixes are valid
            if "expected_classes" in section[field].keys():
                check_expected_classes(section[field]["expected_classes"])
            if "classes_to_avoid" in section[field].keys():
                check_expected_classes(section[field]["classes_to_avoid"])
            if "prefixes" in section[field].keys():
                check_prefixes(section[field]["prefixes"])

        # Check that the attributes are valid
        check_attributes(section["attributes"])
        check_p_subconfig(section["attributes"]["p"])
        for attribute in section["attributes"].keys():
            # Check that the math is valid
            if "math" in section["attributes"][attribute].keys():
                check_math_subconfig(
                    section["attributes"][attribute]["math"])

        # Check that the domain is valid
        check_domain(section["domain"])

        # Check the reindexing and expected_taxa (if present)
        if "reindex" in section.keys():
            check_reindexing(section["reindex"])
        if "expected_taxa" in section.keys():
            check_expected_taxa(section["expected_taxa"])
    except Exception as e:
        raise ValueError(
            "Section %d: %s; See README.md for usage", i, e)
