# 2025 Skye Lane Goetz

from tablassert.cfg import load_yaml
import typer


app = typer.Typer()


@app.command()
def check_graph_config(
    path: str = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to GraphConfig.yaml",
        exists=True,
        file_okay=True,
        readable=True,
    )
):
    """Check if GraphConfig is valid"""
    load_yaml(path, "GraphConfig")


@app.command()
def check_table_config(
    path: str = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to TableConfig.yaml",
        exists=True,
        file_okay=True,
        readable=True,
    )
):
    """Check if TableConfig is valid"""
    load_yaml(path, "TableConfig")


if __name__ == "__main__":
    app()
