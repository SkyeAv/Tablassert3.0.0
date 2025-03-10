from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from src.utils import logging, nlp, toolkit
import pandas as pd
import numpy as np
import sqlite3
import joblib
import time
import math
import re


def set_pandas_settings() -> None:
    # Prevent pandas from truncating floats
    pd.options.display.float_format = "{:.4g}".format

    # Prevent pandas from truncating columns
    pd.set_option("display.max_columns", None)

    # Prevent pandas from truncating column widths
    pd.set_option("display.max_colwidth", None)


def four_zip(x: object) -> float:
    if str(x).lower() in ["not_applicable", "not applicable"]:
        return float(404.0)
    else:
        return x


def four_unzip(x: str) -> str:
    if str(x) in ["404.0", "404"]:
        return "not_applicable"
    return x


def create_df(data_location: str) -> object:
    """
    Creates a DataFrame from a given data location.

    The function determines the file extension of the data file and reads
    the file accordingly. It supports CSV, TSV, TXT, XLSX, and XLS file
    formats.

    Args:
        data_location (dict): A dictionary containing the path to the data
            file and any additional parameters needed for reading the file.

    Returns:
        DataFrame: A pandas DataFrame created from the data file.

    Raises:
        ValueError: If the file extension is invalid or if there is an
            error creating the DataFrame.
    """
    try:
        # Get the file extension
        ext = toolkit.fast_extension(data_location["path_to_file"])

        # Read the file based on its extension
        if ext in [".csv", ".tsv", ".txt"]:
            # For CSV, TSV, or TXT files, use the delimiter specified
            return pd.read_csv(
                data_location["path_to_file"],
                sep=data_location["delimiter"])
        elif ext in ["xlsx", ".xls"]:
            # For XLSX or XLS files, use the specified sheet name
            return pd.read_excel(
                data_location["path_to_file"],
                sheet_name=data_location["sheet_to_use"])
        else:
            # Raise an error for unsupported file extensions
            raise ValueError("Invalid file extension")
    except Exception as e:
        # Raise a ValueError if there is an issue creating the DataFrame
        raise ValueError("Error creating DataFrame: " + str(e))


def save_dateframe(df: object, output_path: str) -> None:
    df.to_csv(output_path, sep="\t", index=False)


def get_letters(i: int) -> str:
    """
    Converts a given integer into a string of uppercase letters,
    similar to Excel column naming (e.g., 0 -> 'A', 25 -> 'Z', 26 -> 'AA').

    Args:
        i (int): The integer to convert.

    Returns:
        str: The corresponding string of uppercase letters.
    """
    letters = ""
    while i >= 0:
        # Calculate the current letter and prepend it to the result
        letters = chr(i % 26 + 65) + letters
        # Move to the next "digit"
        i = i // 26 - 1
    return letters


def alphabetic_columns(df: object) -> object:
    df.columns = [get_letters(i) for i in range(len(df.columns))]
    return df


def numeric_columns(df: object) -> object:
    df.columns = [str(i + 1) for i in range(len(df.columns))]
    return df


def column_stylinator(df: object, column_style: str) -> object:
    if column_style == "numeric":
        return numeric_columns(df)
    elif column_style == "alphabetic":
        return alphabetic_columns(df)
    elif column_style == "normal":
        return df
    else:
        raise ValueError("Invalid column_style")


def column_truncinator(df: object) -> object:
    columns = [
        "subject", "predicate", "object", "domain", "subject_name",
        "object_name", "edge_score", "n", "relationship_strength", "p",
        "relationship_type", "p_correction_method", "knowledge_level",
        "agent_type", "publication", "journal", "publication_name",
        "authors", "year_published", "table_url", "sheet_to_use",
        "yaml_curator", "curator_organization", "method_notes",
        "subject_category", "object_category"]
    try:
        return df[columns]
    except KeyError as e:
        raise ValueError("Error truncating columns: " + str(e))


def dataframe_slicnator(df: object, first_line: int, last_line: int) -> object:
    df = df.iloc[(first_line - 1):(last_line - 1)].copy()
    return df


def basic_key_value_column_addinator(df: object, subconfig: dict) -> object:
    """
    Adds columns to a given DataFrame with the specified keys and values.

    Args:
        df (object): The DataFrame to which columns should be added.
        subconfig (dict): A dictionary containing the column names and
            values to be added.

    Returns:
        object: The updated DataFrame with the added columns.

    Raises:
        ValueError: If there is an issue adding the columns.
    """
    for column, value in subconfig.items():
        # Add the column to the DataFrame
        df[column] = value
    return df


def column_dropinator(df: object, columns_to_drop: list) -> object:
    return df.drop(columns=columns_to_drop, axis=1)


def drop_duplicates_by_columnsinator(df: object, columns: list) -> object:
    return df.drop_duplicates(subset=columns).reset_index(drop=True)


def column_renamanator(df: object, renamements: dict) -> object:
    return df.rename(columns=renamements)


def column_nan_dropinator(df: object, columns_to_drop: list) -> object:
    return df.dropna(subset=columns_to_drop)


def non_numeric_regexinator(val: str) -> str:
    if str(val).lower() not in ["not_applicable", "not applicable"]:
        return re.sub(r"[^\d.]", "", str(val))
    else:
        return val


def numeric_row_convertinator(df: object, column: str = "p") -> object:
    df[column] = df[column].apply(lambda x: non_numeric_regexinator(str(x)))
    df["p"] = df["p"].apply(lambda x: four_zip(str(x)))
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = column_nan_dropinator(df, [column])
    return df


def math_attributinator(df: object, column: str, subconfig: list) -> object:
    for config in subconfig:
        op = getattr(math, config["operation"])
        param = float(config.get("parameter", 0.0))
        order = config.get("order_last", False)
        if order:
            df[column] = df[column].apply(lambda x: op(x, param))
        elif order is None:
            df[column] = df[column].apply(lambda x: op(x))
        else:
            df[column] = df[column].apply(lambda x: op(param, x))
    return df


def first_elementinator(iterable: object) -> object:
    return next(iter(iterable))


def attribute_addinator(df: object, subconfig: dict) -> object:
    """
    Processes attributes in the DataFrame according to the provided subconfig.

    Args:
        df (object): The DataFrame to process.
        subconfig (dict): A dictionary containing attribute configurations.

    Returns:
        object: The updated DataFrame with processed attributes.
    """
    for attribute, config in subconfig.items():

        # Drop the existing attribute column if its name differs from config
        if attribute in df.columns and config.get("column_name") != attribute:
            df = column_dropinator(df, [attribute])

        # Add a new column with a specified value or rename an existing column
        if "value" in config.keys():
            df = basic_key_value_column_addinator(
                df, {attribute: config["value"]})
        else:
            df = column_renamanator(df, {config["column_name"]: attribute})

        # Process the "p" attribute, if present
        if attribute == "p":
            df = numeric_row_convertinator(df)

        # Apply mathematical operations if specified in the configuration
        if "math" in config.keys():
            df = math_attributinator(df, attribute, config["math"])

    return df


def column_explodinator(df: object, column: str) -> object:
    return df.explode(column)


def split_explodinator(df: object, column: str, delimiter: str) -> object:
    df[column] = df[column].str.split(delimiter)
    return column_explodinator(df, column)


def fill_naninator(df: object, column: str, method: str) -> object:
    df[column] = df[column].fillna(method=method)
    return df


def regular_expressionator(df: object, column: str, subconfig: dict) -> object:
    for expression in subconfig:
        repl = expression["replacement"] or ""
        df[column] = df[column].apply(
            lambda x: re.sub(expression["pattern"], repl, str(x)))
    return df


def thing_prefixinator(df: object, column: str, subconfig: dict) -> object:
    for config in subconfig:
        df[column] = config["prefix"] + df[column]
    return df


def thing_suffixinator(df: object, column: str, subconfig: dict) -> object:
    for config in subconfig:
        df[column] = df[column] + config["suffix"]
    return df


def remove_unmapped_rowsinator(dataframe: object, column: str) -> object:
    """
    Removes rows from the DataFrame where the specified column contains a list.

    Args:
        dataframe (object): The DataFrame to process.
        column (str): The name of the column to check for lists.

    Returns:
        object: A new DataFrame with rows containing lists in the
        specified column removed.
    """
    # Identify rows where the column value is not a list
    is_not_list = dataframe[column].apply(
        lambda val: not isinstance(val, list))

    # Filter the DataFrame to include only rows where
    # the column value is not a list
    cleaned_dataframe = dataframe[is_not_list].reset_index(drop=True)

    return cleaned_dataframe


def empty_check(df: object, column: str, after_what: str) -> None:
    if df.empty:
        raise ValueError(
            f"Error processing {column}: Empty Column after {after_what}")


def remove_before_colon(val: str) -> str:
    return re.sub(r"^.*?:", "", str(val))


def biolink_it(val: str) -> str:
    return "biolink:" + val


def progress_handler() -> int:
    global start_time
    global handler_timeout
    if start_time is None:
        start_time = time.time()
    elapsed_time = time.time() - start_time
    if elapsed_time > handler_timeout:
        return 1
    return 0


def full_map2_base_executinator(
        cursor: object, query: str,
        params: tuple, db: str) -> str:
    """
    Executes a query and returns the results as a tuple of
    (preferred name, class, curie). If the query returns no results,
    returns None. If the query raises an OperationalError, logs the
    error and returns None.

    :param cursor: The database connection to use for the query
    :param query: The query to execute
    :param params: The parameters to pass to the query
    :param db: The name of the database being queried
    :return: A tuple of (preferred name, class, curie) or None
    """
    try:
        cursor.execute(query, params)
        result = cursor.fetchone()
        if result:
            category = result[2]
            if db in ["babel", "kg2"]:
                category = biolink_it(category)
            return (result[0], result[1], category)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(params, f"full_map2_base {db}", e)


def full_map2_classed_taxonless_executinator(
        cursor: object, query: str, params: tuple,
        db: str, classes: list, avoid: list) -> str:
    """
    Executes a query and returns the results as a tuple of
    (preferred name, class, curie). If the query returns no results,
    returns None. If the query raises an OperationalError, logs the
    error and returns None.

    This function is similar to full_map2_base_executinator but will
    only return results if the class is in the list of classes and
    not in the list of classes to avoid.

    :param cursor: The database connection to use for the query
    :param query: The query to execute
    :param params: The parameters to pass to the query
    :param db: The name of the database being queried
    :param classes: The list of classes to include in the results
    :param avoid: The list of classes to exclude from the results
    :return: A tuple of (preferred name, class, curie) or None
    """
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        if results:
            for result in results:
                category = result[2]
                if db in ["babel", "kg2"]:
                    category = biolink_it(category)
                if category in classes and category not in avoid:
                    return (result[0], result[1], category)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(
            params, f"full_map2_classed_taxonless {db}", e)


def full_map2_classless_with_taxon_executinator(
        cursor: object, query: str,
        params: tuple, db: str, taxa: list) -> str:
    """
    Executes a query and returns the results as a tuple of
    (preferred name, class, curie). If the query returns no results,
    returns None. If the query raises an OperationalError, logs the
    error and returns None.

    This function is similar to full_map2_base_executinator but will
    only return results if the class is not "Gene" or if the taxon
    is in the list of taxa.

    :param cursor: The database connection to use for the query
    :param query: The query to execute
    :param params: The parameters to pass to the query
    :param db: The name of the database being queried
    :param taxa: The list of taxa to include in the results
    :return: A tuple of (preferred name, class, curie) or None
    """
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        if results:
            for result in results:
                # if the class is "Gene", only return results if the
                # taxon is in the list of taxa
                category = result[2]
                if db in ["babel", "kg2"]:
                    category = biolink_it(category)
                if "biolink:Gene" in category:
                    if result[3] is not None and result[3] in taxa:
                        return (result[0], result[1], category)
                # otherwise, return results regardless of the taxon
                else:
                    return (result[0], result[1], category)
    except sqlite3.OperationalError as e:
        # log the error and return None
        logging.log_slow_query(
            params, f"full_map2_classless_taxon {db}", e)


def full_map2_classed_with_taxon_executinator(
        cursor: object, query: str, params: tuple, db: str,
        classes: list, avoid: list, taxa: list) -> str:
    """
    Executes a query and returns the results as a tuple of
    (preferred name, class, curie). If the query returns no results,
    returns None. If the query raises an OperationalError, logs the
    error and returns None.

    This function is similar to full_map2_base_executinator but will
    only return results if the class is in the list of classes and
    not in the list of classes to avoid. If the class is "Gene", only
    returns results if the taxon is in the list of taxa.

    :param cursor: The database connection to use for the query
    :param query: The query to execute
    :param params: The parameters to pass to the query
    :param db: The name of the database being queried
    :param classes: The list of classes to include in the results
    :param avoid: The list of classes to exclude from the results
    :param taxa: The list of taxa to include in the results if the
        class is "Gene"
    :return: A tuple of (preferred name, class, curie) or None
    """
    try:
        cursor.execute(query, params)
        results = cursor.fetchall()
        if results:
            for result in results:
                # if the class is "Gene", only return results if the
                # taxon is in the list of taxa
                category = result[2]
                if db in ["babel", "kg2"]:
                    category = biolink_it(category)
                if "biolink:Gene" in category:
                    if result[3] is not None and result[3] in taxa and \
                            category in classes and category not in avoid:
                        return (result[0], result[1], category)
                # otherwise, return results regardless of the taxon
                else:
                    if category in classes and category not in avoid:
                        return (result[0], result[1], category)
    except sqlite3.OperationalError as e:
        # log the error and return None
        logging.log_slow_query(
            params, f"full_map2_classed_with_taxon {db}", e)


def full_map2(
        val: str, taxa: list, classes: list, avoid: list,
        cur_kg2: object, cur_babel: object,
        cur_override: object, cur_supplement: object) -> object:
    """
    Maps a given term to a CURIE, preferred name, and category.

    Args:
        val (str): The term to map.
        taxa (list): A list of expected_taxa to restrict the search.
        classes (list): A list of classes to restrict the search to.
        avoid (list): A list of classes to exclude from the results.
        cur_kg2 (object): A connection to the KG2 database.
        cur_babel (object): A connection to the Babel database.
        cur_override (object): A connection to the override database.
        cur_supplement (object): A connection to the supplement database.

    Returns:
        object: A tuple of (preferred name, class, curie) or the original
        term if no mapping is found.
    """
    try:
        # Set progress handlers for database connections
        cur_override.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_babel.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_kg2.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_supplement.connection.set_progress_handler(
            lambda: progress_handler(), 1)

        # Define SQL queries for different database operations
        os_base = """
            SELECT
                name_to_curie.curie,
                curie_to_preferred_name.preferred_name,
                curie_to_class.class
            FROM name_to_curie
            INNER JOIN curie_to_preferred_name
                ON name_to_curie.curie = curie_to_preferred_name.curie
            INNER JOIN curie_to_class
                ON name_to_curie.curie = curie_to_class.curie
            WHERE name_to_curie.name = ?;"""
        os_hash = """
            SELECT
                hashed_name_to_curie.curie,
                curie_to_preferred_name.preferred_name,
                curie_to_class.class
            FROM hashed_name_to_curie
            INNER JOIN curie_to_preferred_name
                ON hashed_name_to_curie.curie = curie_to_preferred_name.curie
            INNER JOIN curie_to_class
                ON hashed_name_to_curie.curie = curie_to_class.curie
            WHERE hashed_name_to_curie.hashed_name = ?;"""
        os_token = """
            SELECT
                tokenized_name_to_curie.curie,
                curie_to_preferred_name.preferred_name,
                curie_to_class.class
            FROM tokenized_name_to_curie
            INNER JOIN curie_to_preferred_name
                ON tokenized_name_to_curie.curie =
                    curie_to_preferred_name.curie
            INNER JOIN curie_to_class
                ON tokenized_name_to_curie.curie = curie_to_class.curie
            WHERE tokenized_name_to_curie.tokenized_name = ?;"""
        babel_base = """
            SELECT
                COALESCE(MAP.PREFERRED, NAMES.CURIE) AS NORM,
                NAMES.NAME,
                NAMES.CATEGORY,
                NAMES.TAXON
            FROM NAMES
            INNER JOIN SYNONYMS ON NAMES.CURIE = SYNONYMS.CURIE
            LEFT JOIN MAP ON NAMES.CURIE = MAP.ALIAS
            WHERE SYNONYMS.SYNONYM = ?;"""
        babel_hash = """
            SELECT
                COALESCE(MAP.PREFERRED, NAMES.CURIE) AS NORM_ID,
                NAMES.NAME,
                NAMES.CATEGORY,
                NAMES.TAXON
            FROM NAMES
            INNER JOIN HASHES ON NAMES.CURIE = HASHES.CURIE
            LEFT JOIN MAP ON NAMES.CURIE = MAP.ALIAS
            WHERE HASHES.HASH = ?;"""
        kg2_base = """
            SELECT
                COALESCE(clusters.cluster_id, nodes.id) AS norm_id,
                COALESCE(clusters.name, nodes.name) AS norm_name,
                COALESCE(clusters.category, nodes.category) as norm_cat
            FROM nodes
            LEFT JOIN clusters ON nodes.id = clusters.cluster_id
            WHERE nodes.name = ?;"""
        kg2_simp = """
            SELECT
                COALESCE(clusters.cluster_id, nodes.id) AS norm_id,
                COALESCE(clusters.name, nodes.name) AS norm_name,
                COALESCE(clusters.category, nodes.category) as norm_cat
            FROM nodes
            LEFT JOIN clusters ON nodes.id = clusters.cluster_id
            WHERE nodes.name_simplified = ?;"""

        global start_time
        start_time = time.time()

        # Make avoid optional
        if not avoid:
            avoid = []

        if classes and not taxa:
            # Query execution for class-specific mappings
            # without taxa restrictions
            queries = [
                (full_map2_classed_taxonless_executinator, cur_override,
                    os_base, (val,), "override"),
                (full_map2_classed_taxonless_executinator, cur_override,
                    os_hash, (nlp.hash_it(val),), "override"),
                (full_map2_classed_taxonless_executinator, cur_override,
                    os_token, (nlp.tokenize_it(val),), "override"),
                (full_map2_classed_taxonless_executinator, cur_babel,
                    babel_base, (val,), "babel"),
                (full_map2_classed_taxonless_executinator, cur_babel,
                    babel_hash, (nlp.hash_it(val),), "babel"),
                (full_map2_classed_taxonless_executinator, cur_kg2, kg2_base,
                    (val,), "kg2"),
                (full_map2_classed_taxonless_executinator, cur_kg2, kg2_simp,
                    (nlp.nonword_regex(val),), "kg2"),
                (full_map2_classed_taxonless_executinator, cur_supplement,
                    os_base, (val,), "supplement"),
                (full_map2_classed_taxonless_executinator, cur_supplement,
                    os_hash, (nlp.hash_it(val),), "supplement"),
                (full_map2_classed_taxonless_executinator, cur_supplement,
                    os_token, (nlp.tokenize_it(val),), "supplement")]
            for func, cursor, query, params, db in queries:
                start_time = time.time()
                result = func(cursor, query, params, db, classes, avoid)
                if result is not None:
                    logging.log_mapped_edge(
                        val, result, f"full_map2_classed_taxonless {db}")
                    return result
            logging.log_dropped_edge(
                val, "dropped\tfull_map2\tclassed_taxonless")
            return [val]

        if not classes and taxa:
            # Query execution for taxa-specific mappings
            # without class restrictions
            queries = [
                (full_map2_base_executinator,
                    (cur_override, os_base, (val,), "override")),
                (full_map2_base_executinator,
                    (cur_override, os_hash, (nlp.hash_it(val),), "override")),
                (full_map2_base_executinator,
                    (cur_override, os_token,
                        (nlp.tokenize_it(val),), "override")),
                (full_map2_classless_with_taxon_executinator,
                    (cur_babel, babel_base, (val,), "babel", taxa)),
                (full_map2_classless_with_taxon_executinator,
                    (cur_babel, babel_hash, (nlp.hash_it(val),),
                        "babel", taxa)),
                (full_map2_base_executinator,
                    (cur_kg2, kg2_base, (val,), "kg2")),
                (full_map2_base_executinator,
                    (cur_kg2, kg2_simp, (nlp.nonword_regex(val),), "kg2")),
                (full_map2_base_executinator,
                    (cur_supplement, os_base, (val,), "supplement")),
                (full_map2_base_executinator,
                    (cur_supplement, os_hash,
                        (nlp.hash_it(val),), "supplement")),
                (full_map2_base_executinator,
                    (cur_supplement, os_token,
                        (nlp.tokenize_it(val),), "supplement"))]

            # Iterate through the function calls
            for func, args in queries:
                start_time = time.time()
                result = func(*args)
                if result is not None:
                    logging.log_mapped_edge(
                        val, result, f"classless_with_taxon {db}")
                    return result
            logging.log_dropped_edge(
                val, "dropped\tfull_map2\tclassless_with_taxon")
            return [val]

        if classes and taxa:
            # Query execution for mappings
            # with both class and taxa restrictions
            queries = [
                (full_map2_classed_taxonless_executinator,
                    (cur_override, os_base, (val,),
                        "override", classes, avoid)),
                (full_map2_classed_taxonless_executinator,
                    (cur_override, os_hash, (nlp.hash_it(val),),
                        "override", classes, avoid)),
                (full_map2_classed_taxonless_executinator,
                    (cur_override, os_token, (nlp.tokenize_it(val),),
                        "override", classes, avoid)),
                (full_map2_classed_with_taxon_executinator,
                    (cur_babel, babel_base, (val,), "babel",
                        classes, avoid, taxa)),
                (full_map2_classed_with_taxon_executinator,
                    (cur_babel, babel_hash, (nlp.hash_it(val),),
                        "babel", classes, avoid, taxa)),
                (full_map2_classed_taxonless_executinator,
                    (cur_kg2, kg2_base, (val,), "kg2",
                        classes, avoid)),
                (full_map2_classed_taxonless_executinator,
                    (cur_kg2, kg2_simp, (nlp.nonword_regex(val),),
                        "kg2", classes, avoid)),
                (full_map2_classed_taxonless_executinator,
                    (cur_supplement, os_base, (val,), "supplement",
                        classes, avoid)),
                (full_map2_classed_taxonless_executinator,
                    (cur_supplement, os_hash, (nlp.hash_it(val),),
                        "supplement", classes, avoid)),
                (full_map2_classed_taxonless_executinator,
                    (cur_supplement, os_token, (nlp.tokenize_it(val),),
                        "supplement", classes, avoid))]

            # Iterate through the function calls
            for func, args in queries:
                start_time = time.time()
                result = func(*args)
                if result is not None:
                    logging.log_mapped_edge(
                        val, result, f"classed_with_taxon {db}")
                    return result
            logging.log_dropped_edge(
                val, "dropped\tfull_map2\tclassed_with_taxon")
            return [val]

        else:
            # Query execution for general mappings
            # without specific restrictions
            queries = [
                (full_map2_base_executinator, cur_override, os_base, (val,),
                    "override"),
                (full_map2_base_executinator, cur_override, os_hash,
                    (nlp.hash_it(val),), "override"),
                (full_map2_base_executinator, cur_override, os_token,
                    (nlp.tokenize_it(val),), "override"),
                (full_map2_base_executinator, cur_babel, babel_base,
                    (val,), "babel"),
                (full_map2_base_executinator, cur_babel, babel_hash,
                    (nlp.hash_it(val),), "babel"),
                (full_map2_base_executinator, cur_kg2, kg2_base, (val,),
                    "kg2"),
                (full_map2_base_executinator, cur_kg2, kg2_simp,
                    (nlp.nonword_regex(val),), "kg2"),
                (full_map2_base_executinator, cur_supplement, os_base, (val,),
                    "supplement"),
                (full_map2_base_executinator, cur_supplement, os_hash,
                    (nlp.hash_it(val),), "supplement"),
                (full_map2_base_executinator, cur_supplement, os_token,
                    (nlp.tokenize_it(val),), "supplement")
            ]
            for func, cursor, query, params, db in queries:
                start_time = time.time()
                result = func(cursor, query, params, db)
                if result is not None:
                    logging.log_mapped_edge(
                        val, result, f"full_map2_base {db}")
                    return result
            logging.log_dropped_edge(val, "dropped\tfull_map2\tbase")
            return [val]

    except Exception as e:
        raise ValueError(f"{val} broke full_map2\t{e}")


def half_map2_executinator(
        cursor: object, query: str,
        params: tuple, db: str) -> str:
    """
    Execute a query on a database and return the first result.

    Args:
        cursor (object): A database cursor object.
        query (str): The query to execute.
        params (tuple): The parameters to pass to the query.
        db (str): The name of the database.

    Returns:
        str: The first result of the query, or None if no result is found.
    """
    try:
        # Execute the query with the given parameters
        cursor.execute(query, params)
        # Fetch one result
        result = cursor.fetchone()
        # Return the first result, or None if no result is found
        return result[0] if result else None
    except sqlite3.OperationalError as e:
        # Log any errors that occur
        logging.log_slow_query(params, f"half_map2 {db}", e)


def half_map2(
        curie: str, cur_kg2: object, cur_babel: object,
        cur_override: object, cur_supplement: object):
    """
    Maps the given curie to the preferred name, class, and curie in the
    Babel, KG2, and supplement databases.

    Args:
        curie (str): The curie to map.
        cur_kg2 (object): The KG2 database connection.
        cur_babel (object): The Babel database connection.
        cur_override (object): The override database connection.
        cur_supplement (object): The supplement database connection.

    Returns:
        tuple: A tuple containing the preferred name, class, and curie of the
            given curie.
    """
    try:
        # Set progress handlers on the database connections
        cur_override.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_babel.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_kg2.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_supplement.connection.set_progress_handler(
            lambda: progress_handler(), 1)

        # Set global start time
        global start_time

        # Attempt to normalize the curie to its preferred name
        norm_queries = [
            (cur_babel, "SELECT PREFERRED FROM MAP WHERE ALIAS = ?",
                (curie,), "babel"),
            (cur_babel, "SELECT cluster_id FROM nodes WHERE id = ?", (curie,),
                "kg2")]
        for cursor, query, params, db in norm_queries:
            start_time = time.time()
            normalized_curie = half_map2_executinator(
                cursor, query, params, db)
            if normalized_curie is not None:
                logging.log_mapped_edge(curie, normalized_curie, db)
                curie = normalized_curie
                break

        # Attempt to retrieve the preferred name
        name_queries = [
            (cur_override,
                """SELECT preferred_name FROM
                curie_to_preferred_name WHERE curie = ?""",
                (curie,), "override"),
            (cur_babel, "SELECT NAME FROM NAMES WHERE CURIE = ?",
                (curie,), "babel"),
            (cur_babel, "SELECT name FROM clusters WHERE cluster_id = ?",
                (curie,), "kg2"),
            (cur_supplement,
                """SELECT preferred_name FROM
                curie_to_preferred_name WHERE curie = ?""",
                (curie,), "supplement")]
        for cursor, query, params, db in name_queries:
            start_time = time.time()
            name = half_map2_executinator(cursor, query, params, db)
            if name is not None:
                logging.log_mapped_edge(curie, name, db)
                break

        # If no name is found, log an error
        if not name:
            logging.log_dropped_edge(curie, "dropped\thalf_map2\tno name")
            return [curie]

        # Attempt to retrieve the category
        category_queries = [
            (cur_override, "SELECT class FROM curie_to_class WHERE curie = ?",
                (curie,), "override"),
            (cur_babel, "SELECT CATEGORY FROM NAMES WHERE CURIE = ?",
                (curie,), "babel"),
            (cur_babel, "SELECT category FROM clusters WHERE cluster_id = ?",
                (curie,), "kg2"),
            (cur_supplement,
                "SELECT class FROM curie_to_class WHERE curie = ?",
                (curie,), "supplement")]
        for cursor, query, params, db in category_queries:
            start_time = time.time()
            category = half_map2_executinator(cursor, query, params, db)
            if category is not None:
                if db in ["babel", "kg2"]:
                    category = biolink_it(category)
                logging.log_mapped_edge(curie, category, db)
                break

        # If no category is found, log an error
        if not category:
            logging.log_dropped_edge(curie, "dropped\thalf_map2\tno category")
            return [curie]

        # Return the preferred name, class, and curie
        result = (curie, name, category)
        logging.log_mapped_edge(curie, result, "half_map2")
        return result
    except Exception as e:
        raise ValueError(
            f"{curie} broke half_map2\t{e}")


def check_that_curie_case(
            curie: object, babel: object, kg2: object,
            override: object, supplement: object) -> None:
    if not isinstance(curie, str):
        raise ValueError(
            f"Invalid value: {curie} should be instance str")
    if not isinstance(half_map2(
            curie, kg2, babel, override, supplement), tuple):
        raise ValueError(
            f"Invalid value: {curie} does not map via half_map")


def check_that_value_case(
            value: object, babel: object, kg2: object, override: object,
            supplement: object, expected_taxa: object,
            classes: object, avoid: object) -> None:
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid value: {value} should be instance str")
    if not isinstance(full_map2(
                value, expected_taxa, classes, avoid,
                kg2, babel, override, supplement), tuple):
        raise ValueError(
            f"Invalid value: {value} does not map via full_map")


def node_columninator(
        df: object, subconfig: dict, column: str,
        kg2: object, babel: object, override: object,
        supplement: object, expected_taxa: object,
        classes: object, avoid: object) -> object:
    """
    Processes a specified column in the DataFrame according to the provided
    subconfig.

    Args:
        df (object): The DataFrame to process.
        subconfig (dict): A dictionary containing column configurations.
        column (str): The name of the column to process.
        kg2 (object): A database connection to the KG2 database.
        babel (object): A database connection to the Babel database.
        override (object): A database connection to the override database.
        supplement (object): A database connection to the supplement database.
        expected_taxa (object): A list of expected_taxa to restrict the search.
        classes (object): A list of classes to restrict the search to.

    Returns:
        object: The updated DataFrame with processed columns.
    """
    try:
        # Add the column to the DataFrame if it doesn't exist
        if "curie" in subconfig or "value" in subconfig:
            df = basic_key_value_column_addinator(
                df, {column: subconfig.get("curie", subconfig.get("value"))})
        # Rename the column if specified
        elif "curie_column_name" in subconfig \
                or "value_column_name" in subconfig:
            df = column_renamanator(
                df, {subconfig.get(
                        "curie_column_name", subconfig.get("value_column_name")
                        ): column})
        # Duplicate the column if specified
        elif "shared_curie_column" in subconfig or \
                "shared_value_column" in subconfig:
            df = basic_key_value_column_addinator(
                df, {column: df[
                    subconfig.get(
                        "shared_curie_column",
                        subconfig.get("shared_value_column"))].values})
        empty_check(df, column, "column creation")
        # Fill NaN values in the column with the specified method
        if "fill_values" in subconfig.keys():
            df = fill_naninator(df, column, subconfig["fill_values"])
        # Split the column into multiple columns if specified
        if "explode_column" in subconfig.keys():
            df = split_explodinator(df, column, subconfig["explode_column"])
        # Perform regular expression replacements if specified
        if "regex_replacements" in subconfig.keys():
            df = regular_expressionator(
                df, column, subconfig["regex_replacements"])
        # Add a prefix to the column if specified
        if "prefix" in subconfig.keys():
            df = thing_prefixinator(df, column, subconfig["prefix"])
        # Add a suffix to the column if specified
        if "suffix" in subconfig.keys():
            df = thing_suffixinator(df, column, subconfig["suffix"])
        # Check for empty values in the column
        empty_check(df, column, "regex_replacements")
        # Map the values in the column to CURIEs,
        # preferred names, and categories
        with (
                sqlite3.connect(kg2) as conn_kg2,
                sqlite3.connect(babel) as conn_babel,
                sqlite3.connect(override) as conn_override,
                sqlite3.connect(supplement) as conn_supplement):
            cur_kg2 = conn_kg2.cursor()
            cur_kg2.execute("PRAGMA cache_size = -64000")
            cur_babel = conn_babel.cursor()
            cur_babel.execute("PRAGMA cache_size = -64000")
            cur_override = conn_override.cursor()
            cur_override.execute("PRAGMA cache_size = -64000")
            cur_supplement = conn_supplement.cursor()
            cur_supplement.execute("PRAGMA cache_size = -64000")
            if "curie" in subconfig.keys():
                check_that_curie_case(
                        str(subconfig["curie"]), cur_babel, cur_kg2,
                        cur_override, cur_supplement)
                df[column] = (
                    [half_map2(
                        str(subconfig["curie"]), cur_kg2, cur_babel,
                        cur_override, cur_supplement)]
                    * len(df[column].to_list()))
            elif "value" in subconfig.keys():
                check_that_value_case(
                        str(subconfig["value"]), cur_babel, cur_kg2,
                        cur_override, cur_supplement, expected_taxa,
                        classes, avoid)
                df[column] = (
                    [full_map2(
                        str(subconfig["value"]), expected_taxa, classes, avoid,
                        cur_kg2, cur_babel, cur_override, cur_supplement)]
                    * len(df[column].to_list()))
            elif "curie_column_name" in subconfig.keys():
                df[column] = df[column].apply(
                    lambda x, *args: half_map2(str(x), *args), args=(
                        cur_kg2, cur_babel, cur_override, cur_supplement))
            elif "shared_curie_column" in subconfig.keys():
                df = column_renamanator(
                    df, {subconfig["shared_curie_column"]: column})
                df[column] = df[column].apply(
                    lambda x, *args: half_map2(str(x), *args), args=(
                        cur_kg2, cur_babel, cur_override, cur_supplement))
            elif "value_column_name" in subconfig.keys():
                df = column_renamanator(
                    df, {subconfig["value_column_name"]: column})
                df[column] = df[column].apply(
                    lambda x, *args: full_map2(str(x), *args), args=(
                        expected_taxa, classes, avoid, cur_kg2, cur_babel,
                        cur_override, cur_supplement))
            elif "shared_value_column" in subconfig.keys():
                df[column] = df[column].apply(
                    lambda x, *args: full_map2(str(x), *args), args=(
                        expected_taxa, classes, avoid, cur_kg2, cur_babel,
                        cur_override, cur_supplement))
            cur_kg2.close()
            cur_babel.close()
            cur_override.close()
            cur_supplement.close()
        # Remove the rows with unmapped values
        df = remove_unmapped_rowsinator(df, column)
        # Check for empty values in the column
        empty_check(df, column, "mapping")
        return df
    except ValueError as e:
        raise ValueError(
            f"Error processing {column} column: {str(e)}")


def post_normalizatinator(df: object, column: str) -> object:
    """
    Expands a specified column in the DataFrame into multiple columns.

    Args:
        df (object): The DataFrame to process.
        column (str): The name of the column to expand.

    Returns:
        object: The updated DataFrame with new columns added.
    """
    # Extract the data from the specified column as a list
    normalization_data = df[column].tolist()

    # Define new column names for the expanded data
    normalization_columns = [column, f"{column}_name", f"{column}_category"]

    # Create a DataFrame from the list and assign it to new columns
    df[normalization_columns] = pd.DataFrame(
        [x for x in normalization_data], index=df.index)

    return df


def enforce_p_value_threshold(df: object, threshold: float) -> object:
    """
    Enforces a p-value threshold in a given DataFrame.

    Args:
        df (object): The DataFrame to process.
        threshold (float): The p-value threshold to enforce.

    Returns:
        object: A new DataFrame with rows containing p-values above the
        threshold removed, except for rows where p is equal to 404, which are
        always kept.

    Raises:
        ValueError: If the DataFrame does not contain a column named 'p'.
    """
    return df[(df["p"] <= threshold) | (df["p"] == 404)].reset_index(drop=True)


def drop_duplicatesinator(df: object) -> object:
    return df.drop_duplicates().reset_index(drop=True)


def dropnaninator(df: object) -> object:
    return df.dropna().reset_index(drop=True)


def greater_than_or_equal_toinator(
        df: object, column: str, value: float) -> object:
    return df[df[column] >= value].reset_index(drop=True)


def less_than_or_equal_toinator(
        df: object, column: str, value: float) -> object:
    return df[df[column] < value].reset_index(drop=True)


def is_not_equal_toinator(df: object, column: str, value: object) -> object:
    return df[df[column] != value].reset_index(drop=True)


def is_equal_toinator(df: object, column: str, value: object) -> object:
    return df[df[column] == value].reset_index(drop=True)


def reindexinator(df: object, reindexing: list) -> object:
    for subconfig in reindexing:
        mode = subconfig["mode"]
        if mode == "greater_than_or_equal_to":
            df = greater_than_or_equal_toinator(
                df, subconfig["column"], subconfig["value"])
        elif mode == "less_than_or_equal_to":
            df = less_than_or_equal_toinator(
                df, subconfig["column"], subconfig["value"])
        elif mode == "not_equal_to":
            df = is_not_equal_toinator(
                df, subconfig["column"], subconfig["value"])
        elif mode == "equal_to":
            df = is_equal_toinator(
                df, subconfig["column"], subconfig["value"])
    return df


def join_domainsinator(domains: list) -> object:
    return ",".join(domains)


def null_zip(x: object) -> object:
    """
    Converts certain string representations of
    'not applicable' to a small float.

    Args:
        x (object): The input value to be processed.

    Returns:
        object: 1e-10 if the input is a string indicating 'not applicable',
                otherwise returns the input value unchanged.
    """
    # Check if input is a string representing 'not applicable'
    if str(x).strip().lower() in [
            "not_applicable", "not applicable", "notapplicable"]:
        return 1e-10
    else:
        return x


def null_unzip(x: object) -> object:
    """
    Converts a small float to a string indicating 'not applicable'.

    Args:
        x (object): The input value to be processed.

    Returns:
        object: "not_applicable" if the input is a small float (1e-10),
                otherwise returns the input value unchanged.
    """
    # Check if input is a small float indicating 'not applicable'
    if float(x) == 1e-10:
        return "not_applicable"
    else:
        return x


def log10(x: object) -> float:
    """
    Calculates the base 10 logarithm of the given object.

    If the object is zero, returns the logarithm of a very small value
    (1e-10) instead. This is done to avoid returning negative infinity.

    Args:
        x (object): The input value to be processed.

    Returns:
        float: The base 10 logarithm of the input value.
    """
    return float(
        math.log(float(x), 10)) if int(x) != 0 else float(
            math.log(float(0.0000000001), 10))


def p_zip(x: object) -> float:
    return -log10(x)


def abslog10(x: object) -> float:
    return log10(abs(float(x)))


def p_pentalty(x: object) -> object:
    threshold = 0.05
    if float(x) >= threshold and float(x) <= 1:
        return -1
    else:
        return 0


def score_predicate(x: str, cursor: object) -> object:
    cursor.execute(
        "SELECT score FROM specificity WHERE predicate = ?", (x,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return 0


def score_zip(
        predicate: str, n: int, p: float,
        relationship_strength: float, relationship_type: str,
        p_correction_method: str, method_notes: str,
        cursor: object, model: object, vectorizer: object) -> float:
    """
    Calculates the score of a given association based on its properties.

    The score is calculated using a weighted sum of the following components:

    1. The logarithm of the number of observations.
    2. The logarithm of the p-value.
    3. The penalty for the p-value (if the p-value is larger than a certain
       threshold, the penalty is -1, otherwise it is 0).
    4. The score of the relationship type, as predicted by the model.
    5. The logarithm of the relationship strength.
    6. The score of the predicate, as retrieved from the database.

    Args:
        predicate (str): The predicate of the association.
        n (int): The number of observations.
        p (float): The p-value.
        relationship_strength (float): The strength of the relationship.
        relationship_type (str): The type of the relationship.
        p_correction_method (str): The p-value correction method used.
        method_notes (str): Notes about the method used.
        cursor (object): A database cursor.
        model (object): A machine learning model.
        vectorizer (object): A vectorizer.

    Returns:
        float: The score of the association.
    """
    a = 65
    b = 20
    c = 300
    d = 95
    e = 80
    f = 40
    g = 1000

    # Calculate the logarithm of the number of observations
    n_component = log10(n) if isinstance(n, int) else 0

    # Calculate the logarithm of the p-value
    p_component = p_zip(p)

    # Calculate the penalty for the p-value
    p_penalty = p_pentalty(p)

    # Calculate the score of the relationship type
    methods = list(
        f"{relationship_type} {p_correction_method} {method_notes}")
    method_component = np.mean(model.predict(vectorizer.transform(methods)))

    # Calculate the logarithm of the relationship strength
    strength_component = abslog10(relationship_strength)

    # Retrieve the score of the predicate from the database
    predicate_component = score_predicate(predicate, cursor)

    # Calculate the score
    score = (
        a * n_component +
        b * p_component +
        c * p_penalty +
        d * method_component +
        e * strength_component +
        f * predicate_component +
        g)

    return log10(score)


def put_dataframe_togtherinator(
        section: dict, threshold: float, output_path: str,
        kg2: str, babel: str, override: str, supplement: str,
        predicates: str, timeout: float, model: str,
        vectorizer: str) -> None:
    """
    Processes a DataFrame according to the given section configuration
    and saves the result to the specified output path.

    Args:
        section (dict): Configuration for processing the DataFrame.
        threshold (float): The p-value threshold to enforce.
        output_path (str): The path where the processed DataFrame is saved.
        kg2 (str): Path to the KG2 database.
        babel (str): Path to the Babel database.
        override (str): Path to the override database.
        supplement (str): Path to the supplement database.

    Raises:
        ValueError: If any errors occur during DataFrame processing.
    """
    try:

        global handler_timeout
        handler_timeout = timeout

        # Set pandas display settings for better output visualization
        set_pandas_settings()

        # Create DataFrame from the specified data location
        location = section["data_location"]
        df = create_df(location)

        # Style the DataFrame columns based on the provided style
        df = column_stylinator(df, section["column_style"])

        # Slice the DataFrame if the file is an Excel format
        if toolkit.fast_extension(
                location["path_to_file"]) in ["xlsx", ".xls"]:
            df = dataframe_slicnator(
                df, location["first_line"], location["last_line"])

        # Add provenance and additional metadata to the DataFrame
        df = basic_key_value_column_addinator(df, section["provenance"])
        df = basic_key_value_column_addinator(
            df, {"predicate": section["predicate"]})
        if "sheet_to_use" in location.keys():
            df = basic_key_value_column_addinator(
                df, {"sheet_to_use": location["sheet_to_use"]})
        else:
            df = basic_key_value_column_addinator(
                df, {"sheet_to_use": "not_applicable"})

        # Add 'domain' column
        df = basic_key_value_column_addinator(
            df, {"domain": join_domainsinator(section["domain"])})

        # Process attributes as per configuration
        df = attribute_addinator(df, section["attributes"])

        # Enforce the specified p-value threshold
        df = enforce_p_value_threshold(df, threshold)

        # Process 'subject' and 'object' columns
        for col in ["subject", "object"]:
            df = node_columninator(
                df, section[col], col, kg2, babel,
                override, supplement, section[col].get("expected_taxa"),
                section[col].get("expected_classes"),
                section[col].get("classes_to_avoid"))
            df = post_normalizatinator(df, col)

        # Reindex DataFrame if specified in the section configuration
        if "reindex" in section:
            df = reindexinator(df, section["reindex"])

        # Add method notes to the DataFrame
        df = basic_key_value_column_addinator(
            df, {"method_notes": section["method_notes"]})

        # Transform the 'relationship_strength' columns
        df["relationship_strength"] = df["relationship_strength"].apply(
            lambda x: null_zip(x))

        # Add the 'edge_score' column
        conn = sqlite3.connect(predicates)
        cursor = conn.cursor()
        model = joblib.load(model)
        vectorizer = joblib.load(vectorizer)

        df["edge_score"] = df.apply(
            lambda row: score_zip(
                row["predicate"], row["n"], row["p"],
                row["relationship_strength"], row["relationship_type"],
                row["p_correction_method"], row["method_notes"],
                cursor, model, vectorizer), axis=1)

        cursor.close()
        conn.close()

        # Truncate, drop NaN values, and remove duplicates from DataFrame
        df = column_truncinator(df)
        df = dropnaninator(df)
        df = drop_duplicatesinator(df)

        # Reverse any specific transformations on the 'p' column
        df["p"] = df["p"].apply(lambda x: four_unzip(str(x)))

        # Reverse any transformations on the 'relationship_strength' column
        df["relationship_strength"] = df["relationship_strength"].apply(
            lambda x: null_unzip(str(x)))

        # Save the processed DataFrame to the specified output path
        save_dateframe(df, output_path)

    except ValueError as e:
        # Raise any ValueErrors encountered during processing
        raise ValueError(str(e))
