__author__ = "Skye Lane Goetz"


from tablassert.io import load_sections
from tablassert.cfg import load_yaml
import typer


app = typer.Typer()


@app.command(name="cgc", help='Alias for "check-graph-config"')
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


@app.command(name="ctc", help='Alias for "check-table-config"')
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


@app.command(name="cac", help='Alias for "check-all-configs"')
@app.command()
def check_all_configs(
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
    """Check if GraphConfig and associated TableConfigs are valid"""
    graph_cfg = load_yaml(path, "GraphConfig")
    dirs = graph_cfg["dirs"]
    load_sections(dirs)


if __name__ == "__main__":
    app()
