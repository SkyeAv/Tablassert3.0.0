# 2025 Skye Lane Goetz

from tablassert.cfg import load_yaml
import typer


app = typer.Typer()


@app.command()
def validate_config(
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
    """Check if configuration is valid"""
    load_yaml(path)


if __name__ == "__main__":
    app()
