from sklearn.preprocessing import MinMaxScaler
from src.utils import toolkit, dataframe
import pandas as pd
import numpy as np
import os


def read_tsv(file_path: str) -> object:
    return pd.read_csv(file_path, sep="\t")


def kg_aggregatinator(kg_path: str) -> None:
    """
    Aggregates all the TSV files in the current directory into a single file.

    Args:
        kg_path (str): The path to the file where the TSV files should be saved

    Raises:
        ValueError: If no TSV files are found in the directory
    """
    # If the file already exists, read it and concatenate it with the new data
    if os.path.isfile(kg_path):
        df = read_tsv(kg_path)
    else:
        df = pd.DataFrame()  # Initialize an empty DataFrame
    # Iterate over all the files in the current directory
    for file in os.listdir("tablassert"):
        # Check if the file has a TSV extension
        if toolkit.fast_extension(file) == ".tsv":
            # Read the file and concatenate it with the existing data
            df = pd.concat([df, read_tsv("tablassert/" + file)])
    # Normalize the 'edge_score' column
    scaler = MinMaxScaler(feature_range=(42.128903, 99.289624))
    df["edge_score"] = scaler.fit_transform(
            np.array(df["edge_score"].values).reshape(-1, 1))
    # Save the concatenated DataFrame to the specified file
    dataframe.save_dateframe(df, kg_path)
    # Check if any TSV files were found in the directory
    if not any(
            toolkit.fast_extension(file) == ".tsv"
            for file in os.listdir("tablassert")):
        raise ValueError("No TSV files found in the directory")


def edge_naminator(name: str, version: str) -> str:
    return str(name) + "_edges_v" + str(version)


def node_naminator(name: str, version: str) -> str:
    return str(name) + "_nodes_v" + str(version)


def kgx_formatinator(kg_path: str, kg_name: str, version: str) -> None:
    """
    Converts a Tablassert Knowledge Graph in TSV format to the KGX format.

    The KGX format is a collection of two TSV files,
    one for nodes and one for edges.

    Args:
        kg_path (str): The path to the Tablassert Knowledge Graph in TSV format
        kg_name (str): The name of the knowledge graph
        version (str): The version of the knowledge graph

    Raises:
        ValueError: If any errors occur during the conversion
    """
    try:
        # Read the TSV file into a DataFrame
        df = read_tsv(kg_path)

        # Construct the paths for the nodes and edges TSV files
        nodes_path = node_naminator(kg_name, version)
        edges_path = edge_naminator(kg_name, version)

        # Deletes the nodes and edges TSV files if they already exist
        if os.path.isfile(nodes_path):
            os.remove(nodes_path)
        if os.path.isfile(edges_path):
            os.remove(edges_path)

        # Select the columns that should be written to the edges TSV file
        edge_columns = [
            "subject", "predicate", "object", "domain", "edge_score", "n",
            "relationship_strength", "p", "relationship_type",
            "p_correction_method", "knowledge_level", "agent_type",
            "publication", "journal", "publication_name",
            "authors", "year_published", "table_url", "sheet_to_use",
            "yaml_curator", "curator_organization", "method_notes"]

        # Extract the edges data
        edges = df[edge_columns]

        # Strip whitespace in 'method_notes' column
        edges["method_notes"] = edges["method_notes"].apply(
            lambda x: str(x).strip())

        # Write the edges DataFrame to the edges TSV file
        edges.to_csv(edges_path, sep="\t", index=False)

        # Extract the nodes data by renaming columns and concatenating
        nodes = pd.concat([
            df[["subject", "subject_name", "subject_category"]].rename(
                columns={
                    "subject": "id", "subject_name": "name",
                    "subject_category": "category"}),
            df[["object", "object_name", "object_category"]].rename(
                columns={
                    "object": "id", "object_name": "name",
                    "object_category": "category"})])

        # Remove duplicate entries in the nodes DataFrame
        nodes = dataframe.drop_duplicatesinator(nodes)
        nodes = dataframe.drop_duplicates_by_columnsinator(
            nodes, ["id", "category"])
        nodes = dataframe.drop_duplicates_by_columnsinator(
            nodes, ["id", "name"])

        # Strip whitespace in nodes
        nodes = nodes.map(lambda x: str(x).strip())

        # Write the nodes DataFrame to the nodes TSV file
        nodes.to_csv(nodes_path, sep="\t", index=False)

        # Gzip the nodes and edges TSV files to save space
        toolkit.gzip_file(nodes_path)
        toolkit.gzip_file(edges_path)
    except Exception as e:
        # Raise a ValueError with the caught exception's message
        raise ValueError(str(e))
