from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from src.utils import logging, nlp, toolkit
from src import main
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
    return df.iloc[(first_line - 1):last_line]


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
    if start_time is None:
        start_time = time.time()
    elapsed_time = time.time() - start_time
    if elapsed_time > main.handler_timeout:
        return 1
    return 0


def full_map(
        val: str, taxons: list, classes: list,
        cur_kg2: object, cur_babel: object,
        cur_override: object, cur_supplement: object) -> object:
    """
    Maps a given term to a CURIE, preferred name, and category.

    Args:
        val (str): The term to map.
        taxons (list): A list of taxons to restrict the search to.
        classes (list): A list of classes to restrict the search to.
        cur_kg2 (object): A database connection to the KG2 database.
        cur_babel (object): A database connection to the Babel database.
        cur_override (object): A database connection to the override database.
        cur_supplement (object):
            A database connection to the supplement database.

    Returns:
        A list of the mapped CURIE, preferred name, and category if found.
        Otherwise, returns the original term.
    """

    global start_time

    # Construct a query to search for the term in
    # the override and supplement databases
    override_supplement_query = ("""
        WITH name_lookup AS (
            SELECT curie
            FROM name_to_curie
            WHERE name = ?
            UNION
            SELECT curie
            FROM tokenized_name_to_curie
            WHERE tokenized_name = ?
            UNION
            SELECT curie
            FROM hashed_name_to_curie
            WHERE hashed_name = ?
        ),
        preferred_name_lookup AS (
            SELECT
                curie_to_preferred_name.curie,
                curie_to_preferred_name.preferred_name,
                curie_to_class.class
            FROM curie_to_preferred_name
            JOIN curie_to_class
            ON curie_to_preferred_name.curie = curie_to_class.curie
            WHERE curie_to_preferred_name.curie IN (
                SELECT curie FROM name_lookup)
        ),
        ranked_results AS (
            SELECT
                curie,
                preferred_name,
                class,
                ROW_NUMBER() OVER (
                PARTITION BY class ORDER BY curie) AS row_num
            FROM preferred_name_lookup
        )
        SELECT
            curie,
            preferred_name,
            class
        FROM ranked_results
        WHERE row_num = 1;""", (
            val, nlp.tokenize_it(val), nlp.nonword_regex(val)))

    # Execute the query in the override database
    try:
        start_time = time.time()
        cur_override.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_override.execute(
            override_supplement_query[0], override_supplement_query[1])
        result = cur_override.fetchall()
        cur_override.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(val, "override", e)
        result = None

    # Check if result is found in the override database
    if result:
        if classes is None:
            logging.log_mapped_edge(
                val,
                (result[0][0], result[0][1], result[0][2]),
                "override\tfull_map\tnormal")
            return (result[0][0], result[0][1], result[0][2])
        for i, (category) in enumerate((item[2] for item in result)):
            if category in classes:
                logging.log_mapped_edge(
                    val,
                    (result[i][0], result[i][1], result[i][2]),
                    "override\tfull_map\tclassed")
                return (result[i][0], result[i][1], result[i][2])

    # Construct a query to search for the term in the Babel database
    babel_query = """
        WITH synonym_lookup AS (
            SELECT CURIE
            FROM SYNONYMS
            WHERE SYNONYM = ?
            UNION
            SELECT CURIE
            FROM HASHES
            WHERE HASH = ?
        ),
        name_lookup AS (
            SELECT N.NAME, N.CATEGORY, N.TAXON, N.CURIE,
            ROW_NUMBER() OVER (
                PARTITION BY N.CATEGORY ORDER BY N.CATEGORY) AS row_num
            FROM NAMES N
            JOIN synonym_lookup S ON N.CURIE = S.CURIE
            LEFT JOIN MAP M ON S.CURIE = M.ALIAS
            LEFT JOIN NAMES N2 ON M.PREFERRED = N2.CURIE
            WHERE N.CURIE = COALESCE(M.PREFERRED, S.CURIE)
        )
        SELECT NAME, CATEGORY, TAXON, CURIE
        FROM name_lookup
        WHERE row_num = 1;
        """ if not taxons else """
        WITH synonym_lookup AS (
            SELECT CURIE
            FROM SYNONYMS
            WHERE SYNONYM = ?
            UNION
            SELECT CURIE
            FROM HASHES
            WHERE HASH = ?
        ),
        name_lookup AS (
            SELECT N.NAME, N.CATEGORY, N.TAXON, N.CURIE,
            ROW_NUMBER() OVER (
                PARTITION BY N.CATEGORY ORDER BY N.CATEGORY) AS row_num
            FROM NAMES N
            JOIN synonym_lookup S ON N.CURIE = S.CURIE
            LEFT JOIN MAP M ON S.CURIE = M.ALIAS
            LEFT JOIN NAMES N2 ON M.PREFERRED = N2.CURIE
            WHERE N.CURIE = COALESCE(M.PREFERRED, S.CURIE)
        )
        SELECT NAME, CATEGORY, TAXON, CURIE
        FROM name_lookup;
        """
    try:
        start_time = time.time()
        cur_babel.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_babel.execute(
            babel_query, (val, nlp.hash_it(val)))
        result = cur_babel.fetchall()
        cur_babel.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(val, "babel", e)
        result = None

    # Check if result is found in the Babel database
    if result:
        if classes and taxons:
            for i, (category, taxon) in enumerate(
                    (item[1], item[2]) for item in result):
                if (classes and biolink_it(category) in classes) and (
                        taxons and taxon in taxons):
                    logging.log_mapped_edge(
                        val,
                        (result[i][3], result[i][0],
                            biolink_it(result[i][1]), result[i][2]),
                        "babel\tfull_map\tclassed with taxon")
                    return (
                        result[i][3], result[i][0], biolink_it(result[i][1]))
        elif classes and not taxons:
            for i, (category) in enumerate(
                    (item[1]) for item in result):
                if (classes and biolink_it(category) in classes):
                    logging.log_mapped_edge(
                        val,
                        (result[i][3], result[i][0],
                            biolink_it(result[i][1]), result[i][2]),
                        "babel\tfull_map\tclassed")
                    return (
                        result[i][3], result[i][0], biolink_it(result[i][1]))
        elif taxons and not classes:
            if "Gene" in [item[1] for item in result]:
                for i, (taxon) in enumerate(
                        (item[2]) for item in result):
                    if (taxons and taxon in taxons):
                        logging.log_mapped_edge(
                            val,
                            (result[i][3], result[i][0],
                                biolink_it(result[i][1]), result[i][2]),
                            "babel\tfull_map\ttaxon")
                        return (
                            result[i][3], result[i][0], biolink_it(
                                result[i][1]))
            else:
                logging.log_mapped_edge(
                    val,
                    (result[0][3], result[0][0], biolink_it(result[0][1])),
                    "babel\tfull_map\tnormal")
                return (result[0][3], result[0][0], biolink_it(result[0][1]))
        else:
            logging.log_mapped_edge(
                val,
                (result[0][3], result[0][0], biolink_it(result[0][1])),
                "babel\tfull_map\tnormal")
            return (result[0][3], result[0][0], biolink_it(result[0][1]))

    # Construct a query to search for the term in the KG2 database
    kg2_query = """
        WITH node_lookup AS (
            SELECT id, category
            FROM nodes
            WHERE name = ?
            UNION
            SELECT id, category
            FROM nodes
            WHERE name_simplified = ?
        ),
        cluster_lookup AS (
            SELECT id, cluster_id, category
            FROM nodes
            WHERE id IN (SELECT id FROM node_lookup)
        ),
        ranked_results AS (
            SELECT
                cluster_lookup.cluster_id,
                cluster_lookup.id AS node_id,
                cluster_lookup.category AS node_category,
                clusters.name AS cluster_name,
                clusters.category AS cluster_category,
                ROW_NUMBER() OVER (
                PARTITION BY clusters.category
                ORDER BY cluster_lookup.id) AS row_num
            FROM cluster_lookup
            JOIN clusters ON cluster_lookup.cluster_id = clusters.cluster_id
        )
        SELECT
            cluster_id,
            node_id,
            node_category,
            cluster_name,
            cluster_category
        FROM ranked_results
        WHERE row_num = 1;"""
    try:
        start_time = time.time()
        cur_kg2.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_kg2.execute(kg2_query, (val, nlp.nonword_regex(val)))
        result = cur_kg2.fetchall()
        cur_kg2.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(val, "kg2", e)
        result = None

    # Check if result is found in the KG2 database
    if result:
        if classes is None:
            logging.log_mapped_edge(
                val,
                (result[0][0], result[0][3], biolink_it(result[0][4])),
                "kg2\tfull_map\tnormal")
            return (result[0][0], result[0][3], biolink_it(result[0][4]))
        for i, (category) in enumerate((item[4] for item in result)):
            if biolink_it(category) in classes:
                logging.log_mapped_edge(
                    val,
                    (result[0][0], result[0][3], biolink_it(result[0][4])),
                    "kg2\tfull_map\tclassed")
                return (result[i][0], result[i][3], biolink_it(result[i][4]))

    # Execute the query in the supplement database
    try:
        start_time = time.time()
        cur_supplement.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_supplement.execute(
            override_supplement_query[0], override_supplement_query[1])
        result = cur_supplement.fetchall()
        cur_supplement.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(val, "supplement", e)
        result = None

    # Check if result is found in the supplement database
    if result:
        if classes is None:
            logging.log_mapped_edge(
                val,
                (result[0][0], result[0][1], result[0][2]),
                "supplement\tfull_map\tnormal")
            return (result[0][0], result[0][1], result[0][2])
        for i, (category) in enumerate((item[2] for item in result)):
            if category in classes:
                logging.log_mapped_edge(
                    val,
                    (result[i][0], result[i][1], result[i][2]),
                    "supplement\tfull_map\tclassed")
                return (result[i][0], result[i][1], result[i][2])

    # If no result is found, return the original term
    logging.log_dropped_edge(val, "dropped\tfull_map")
    return [val]


def half_map(
        curie: str, cur_kg2: object, cur_babel: object,
        cur_override: object, cur_supplement: object) -> object:
    """
    Maps the given curie to the preferred name, class, and curie in the
    Babel, KG2, and supplement databases.

    Returns:
        tuple: A tuple containing the preferred name, class, and curie of the
            given curie.
    """

    global start_time

    # First, try to find the curie in the override database
    override_supplement_query = ("""
        WITH preferred_name_lookup AS (
            SELECT
                curie_to_preferred_name.curie,
                curie_to_preferred_name.preferred_name,
                curie_to_class.class
            FROM curie_to_preferred_name
            JOIN curie_to_class
            ON curie_to_preferred_name.curie = curie_to_class.curie
            WHERE curie_to_preferred_name.curie = ?
        ),
        ranked_results AS (
            SELECT
                curie,
                preferred_name,
                class,
                ROW_NUMBER() OVER (
                    PARTITION BY class ORDER BY curie
                ) AS row_num
            FROM preferred_name_lookup
        )
        SELECT
            curie,
            preferred_name,
            class
        FROM ranked_results
        WHERE row_num = 1;""", (curie,))

    try:
        start_time = time.time()
        cur_override.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_override.execute(
            override_supplement_query[0], override_supplement_query[1])
        result = cur_override.fetchall()
        cur_override.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(curie, "override", e)
        result = None

    if result:
        logging.log_mapped_edge(
            curie,
            (result[0][0], result[0][1], result[0][2]),
            "override\thalf_map")
        # If the curie is found in the override database, return it
        return (result[0][0], result[0][1], result[0][2])

    # If not, try to find the curie in the Babel database
    try:
        start_time = time.time()
        cur_babel.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_babel.execute("""
            WITH name_lookup AS (
                SELECT N.NAME, N.CATEGORY, M.PREFERRED,
                ROW_NUMBER() OVER (
                    PARTITION BY N.CATEGORY ORDER BY N.CATEGORY) AS row_num
                FROM NAMES N
                LEFT JOIN MAP M ON N.CURIE = COALESCE(M.ALIAS, ?)
                WHERE N.CURIE = COALESCE(M.ALIAS, ?)
            )
            SELECT NAME, CATEGORY, PREFERRED
            FROM name_lookup
            WHERE row_num = 1;
            """, (curie, curie))
        result = cur_babel.fetchall()
        cur_babel.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(curie, "babel", e)
        result = None

    if result:
        logging.log_mapped_edge(
            curie,
            (result[0][2], result[0][0], biolink_it(result[0][1])),
            "kg2\thalf_map")
        # If the curie is found in the Babel database, return it
        return (result[0][2], result[0][0], biolink_it(result[0][1]))

    # If not, try to find the curie in the KG2 database
    try:
        start_time = time.time()
        cur_kg2.connection.set_progress_handler(lambda: progress_handler(), 1)
        cur_kg2.execute("""
            WITH cluster_lookup AS (
                SELECT cluster_id, category
                FROM nodes
                WHERE id = ?  -- Using the parameter to filter by node_id
            ),
            ranked_results AS (
                SELECT
                    cluster_lookup.cluster_id,
                    clusters.name AS cluster_name,
                    clusters.category AS cluster_category,
                    ROW_NUMBER() OVER (
                    PARTITION BY clusters.category
                    ORDER BY cluster_lookup.cluster_id) AS row_num
                FROM cluster_lookup
                JOIN clusters
                    ON cluster_lookup.cluster_id = clusters.cluster_id
            )
            SELECT
                cluster_id,
                cluster_name,
                cluster_category
            FROM ranked_results
            WHERE row_num = 1;""", (curie,))
        result = cur_kg2.fetchall()
        cur_kg2.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(curie, "kg2", e)
        result = None

    if result:
        logging.log_mapped_edge(
            curie,
            (result[0][0], result[0][1], biolink_it(result[0][2])),
            "kg2\thalf_map")
        # If the curie is found in the KG2 database, return it
        return (result[0][0], result[0][1], biolink_it(result[0][2]))

    try:
        start_time = time.time()
        cur_supplement.connection.set_progress_handler(
            lambda: progress_handler(), 1)
        cur_supplement.execute(
            override_supplement_query[0], override_supplement_query[1])
        result = cur_supplement.fetchall()
        cur_supplement.connection.set_progress_handler(None, 1)
    except sqlite3.OperationalError as e:
        logging.log_slow_query(curie, "supplement", e)
        result = None

    if result:
        logging.log_mapped_edge(
            curie,
            (result[0][0], result[0][1], result[0][2]),
            "supplement\thalf_map")
        # If the curie is found in the supplement database, return it
        return (result[0][0], result[0][1], result[0][2])

    # If not, return the original curie
    logging.log_dropped_edge(curie, "dropped\thalf_map")
    return [curie]


def check_that_curie_case(
            curie: object, babel: object, kg2: object,
            override: object, supplement: object) -> None:
    if not isinstance(curie, str):
        raise ValueError(
            f"Invalid value: {curie} should be instance str")
    if not isinstance(half_map(
            curie, kg2, babel, override, supplement), tuple):
        raise ValueError(
            f"Invalid value: {curie} does not map via half_map")


def check_that_value_case(
            value: object, babel: object, kg2: object, override: object,
            supplement: object, taxons: object, classes: object) -> None:
    if not isinstance(value, str):
        raise ValueError(
            f"Invalid value: {value} should be instance str")
    if not isinstance(full_map(
                value, taxons, classes,
                kg2, babel, override, supplement), tuple):
        raise ValueError(
            f"Invalid value: {value} does not map via full_map")


def node_columninator(
        df: object, subconfig: dict, column: str,
        kg2: object, babel: object, override: object,
        supplement: object, taxons: object, classes: object) -> object:
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
        taxons (object): A list of taxons to restrict the search to.
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
                    [half_map(
                        str(subconfig["curie"]), cur_kg2, cur_babel,
                        cur_override, cur_supplement)]
                    * len(df[column].to_list()))
            elif "value" in subconfig.keys():
                check_that_value_case(
                        str(subconfig["value"]), cur_babel, cur_kg2,
                        cur_override, cur_supplement, taxons, classes)
                df[column] = (
                    [full_map(
                        str(subconfig["value"]), taxons, classes, cur_kg2,
                        cur_babel, cur_override, cur_supplement)]
                    * len(df[column].to_list()))
            elif "curie_column_name" in subconfig.keys():
                df = column_renamanator(
                    df, {subconfig["curie_column_name"]: column})
                df[column] = df[column].apply(
                    lambda x, *args: half_map(str(x), *args), args=(
                        cur_kg2, cur_babel, cur_override, cur_supplement))
            elif "value_column_name" in subconfig.keys():
                df = column_renamanator(
                    df, {subconfig["value_column_name"]: column})
                df[column] = df[column].apply(
                    lambda x, *args: full_map(str(x), *args), args=(
                        taxons, classes, cur_kg2, cur_babel,
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
    if str(x).strip().lower() in ["not_applicable", "not applicable"]:
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
    return cursor.fetchone()[0]


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
    a = 0.8
    b = 0.6
    c = 1.3
    d = 1.1
    e = 1.5
    f = 0.7

    # Calculate the logarithm of the number of observations
    n_component = log10(n)

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
        f * predicate_component)

    return score


def put_dataframe_togtherinator(
        section: dict, threshold: float, output_path: str,
        kg2: str, babel: str, override: str, supplement: str) -> None:
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
                override, supplement, section[col].get("taxons"),
                section[col].get("expected_classes"))
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
        conn = sqlite3.connect(main.predicates_sqlite)
        cursor = conn.cursor()
        model = joblib.load(main.confidence_model)
        vectorizer = joblib.load(main.tfidf_vectorizer)

        df["edge_score"] = df.apply(
            lambda row: score_zip(
                row["predicate"], row["n"], row["p"],
                row["relationship_strength"], row["relationship_type"],
                row["p_correction_method"], row["method_notes"],
                cursor, model, vectorizer), axis=1)

        cursor.close()
        conn.close()

        # Reverse any specific transformations on the 'p' column
        df["p"] = df["p"].apply(lambda x: four_unzip(str(x)))

        # Reverse any transformations on the 'relationship_strength' column
        df["relationship_strength"] = df["relationship_strength"].apply(
            lambda x: null_unzip(str(x)))

        # Truncate, drop NaN values, and remove duplicates from DataFrame
        df = column_truncinator(df)
        df = dropnaninator(df)
        df = drop_duplicatesinator(df)

        # Save the processed DataFrame to the specified output path
        save_dateframe(df, output_path)

    except ValueError as e:
        # Raise any ValueErrors encountered during processing
        raise ValueError(str(e))
