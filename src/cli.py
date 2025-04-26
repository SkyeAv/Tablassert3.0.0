# 2025 Skye Lane Goetz

from src.cfg import load_yaml
import typer


app = typer.Typer()


@app.command()
def validate_config(
    path: str = typer.Option(
        ...,
        help="Path to GraphConfig.yaml",
        alias="-p",
        exists=True,
        file_okay=True,
        readable=True,
    )
):
    """Check if configuration is valid"""
    load_yaml(path)


if __name__ == "__main__":
    app()
