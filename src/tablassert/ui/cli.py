from src.tablassert.scoring.training import trainscoringmodel
from src.tablassert.core.build import buildgraph
import typer

app = typer.Typer()


@app.command()
def build(
    graphconfig: str = typer.Option(
        ..., "-g", "--graph-config", help="path to the GraphConfig for your build"
    )
) -> None:
    # typer uses docstrings for command descriptions
    """Build a Knowledge Graph with a GraphConfig"""
    buildgraph(graphconfig)
    return None


@app.command()
def train(
    trainingdata: str = typer.Option(
        ...,
        "-t",
        "--training-data",
        help="a jsonlines file containing edge scoring neural net training data",
    ),
    saveto: str = typer.Option(
        "resources/chroma_db",
        "-s",
        "--save-to",
        help="a path to the file you want to save weights to",
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="the number of epochs to train model"
    ),
) -> None:
    """Trains the Neural Network for Edge Scoring with Specified JSONLINES"""
    trainscoringmodel(trainingdata, saveto, epochs)
    return None


# wrapper for poety entrypoint
def cli() -> None:
    app()
    return None
