from src.utils import (
    toolkit, arguments, logging, kgConfigParser,
    tableConfigParser, dataframe, export)
from concurrent.futures import ProcessPoolExecutor
import sys
import os


def process_section(file: str, i: int, section: dict, kg_config: dict) -> str:
    """
    Processes a section in a configuration file.

    Downloads the table data from the specified URL, parses the section
    configuration, and saves the processed DataFrame to a TSV file in the
    "tablassert" directory.

    Args:
        file (str): The name of the configuration file.
        i (int): The section number (1-indexed).
        section (dict): The section configuration.
        kg_config (dict): The overall configuration for the script.

    Returns:
        str: A success message if the section is processed successfully,
            otherwise an error message.
    """
    try:
        # Log that we are processing the current table
        logging.log_current_table(
            f"{toolkit.get_filename_no_ext(file)} section {i}")
        # Check that the section configuration is valid
        tableConfigParser.parse_section(i, section)
        # Get the table data URL and the path to save the data
        data = section["provenance"]["table_url"]
        path = section["data_location"]["path_to_file"]
        # Create the directory if it doesn't exist
        toolkit.make_dirs(toolkit.get_dirname(path))
        # Download the data if it doesn't already exist
        if not os.path.isfile(path):
            toolkit.download_file(data, path)
        # Get the path to save the processed DataFrame
        output_path = "tablassert/" + toolkit.tsv_it(
            toolkit.slash_cleaner(
                str(i) + "_" + toolkit.get_filename_no_ext(file)))
        # Set global variables
        global handler_timeout
        handler_timeout = kg_config["progress_handler_timeout"]
        global predicates_sqlite
        predicates_sqlite = kg_config["predicates_sqlite"]
        global confidence_model
        confidence_model = kg_config["confidence_model"]
        global tfidf_vectorizer
        tfidf_vectorizer = kg_config["tfidf_vectorizer"]
        # Process the DataFrame if it hasn't already been processed
        if not os.path.isfile(output_path):
            dataframe.put_dataframe_togtherinator(
                section, kg_config["p_value_cutoff"], output_path,
                kg_config["kg2_sqlite"], kg_config["babel_sqlite"],
                kg_config["override_sqlite"], kg_config["supplement_sqlite"]
            )
        # Return a success message
        return f"Success: {toolkit.get_filename_no_ext(file)} section {i}"
    except Exception as e:
        # Log an error if there is a problem
        err_msg = f"""Error in {
            toolkit.get_filename_no_ext(file)}, section {i}: {e}"""
        logging.log_error(err_msg)


def main() -> None:
    """
    Main function for the script.

    This function parses the command line arguments, reads the configuration
    file, processes the sections in the configuration file, and aggregates the
    resulting DataFrames into a single DataFrame saved to a TSV file.

    :return: None
    """
    toolkit.nice_it()  # Decrease the priority of the process
    arguments.enforce_usage(sys.argv[1:])  # Check the command line arguments
    try:
        kg_config = toolkit.read_config(
            sys.argv[1])  # Read the configuration file
    except Exception as e:
        toolkit.exit_with_error(str(e))  # Exit if there is an error
    kgConfigParser.parse_kg_config(kg_config)  # Parse the configuration file
    section_config_files = toolkit.get_section_configs(
        kg_config["config_directories"])
    # Create the directory to save the processed DataFrames
    toolkit.make_dirs("tablassert")
    kg_path = toolkit.tsv_it(
        toolkit.slash_cleaner(kg_config["knowledge_graph_name"]))
    toolkit.remove_file_if_exists(
        kg_path)  # Remove the file if it already exists

    futures = []
    with ProcessPoolExecutor(max_workers=kg_config["max_workers"]) as executor:
        for file in section_config_files:
            try:
                section_config = toolkit.read_config(file)
                sections = toolkit.get_sections(section_config)
                for i, section in enumerate(sections, start=1):
                    futures.append(
                        executor.submit(
                            process_section, file, i, section, kg_config))
            except Exception as e:
                logging.log_error(
                    f"Error in {toolkit.get_filename_no_ext(file)}: {e}")

    for future in futures:
        try:
            result = future.result()
            if result:
                print(str(result))  # Print a success message
        except Exception as e:
            print(f"Error: {e}")  # Print an error message

    try:
        kg_version = kg_config["version_number"]
        kg_name = kg_config["knowledge_graph_name"]
        export.kg_aggregatinator(kg_path)  # Aggregate the DataFrames into one
        export.kgx_formatinator(
            kg_path, kg_name, kg_version)  # Formats the knowledge graph
    except ValueError as e:
        toolkit.exit_with_error(str(e))  # Exit if there is an error


if __name__ == "__main__":
    main()
