from src.tablassert.core.verify import verify_graphconfig, verify_tableconfig
from src.tablassert.scoring.training import trainscoringmodel
from src.tablassert.core.build import buildgraph
import warnings
import typer

warnings.filterwarnings(
    "ignore", category=SyntaxWarning
)  # for 3rd party libraries... linting with flake8 catches the rest
warnings.filterwarnings("ignore", category=FutureWarning)
app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def build(
    graphconfig: str = typer.Option(
        ..., "-g", "--graph-config", help="path to the GraphConfig for your build"
    )
) -> None:
    # typer uses docstrings for command descriptions
    """Builds a Knowledge Graph with a GraphConfig"""
    buildgraph(graphconfig)
    return None


@app.command()
def verify(
    config: str = typer.Option(
        ..., "-c", "--config", help="path to the config you want to verify"
    ),
    graphconfig: bool = typer.Option(
        False,
        "-g",
        "--graph-config",
        help="a flag to indicate that the config you're verifying is a GraphConfig",
    ),
    tableconfig: bool = typer.Option(
        False,
        "-c",
        "--table-config",
        help="a flag to indicate that the config you're verifying is a TableConfig",
    ),
) -> None:
    """Verifies that a GraphConfig or TableConfig is Syntactically Valid (Preemptively Downloads Files to DataLake)"""
    if graphconfig == tableconfig:  # matches if both or neither is specified
        raise ValueError(
            "CODE:300 | You must specify exactly one of --graph-config or --table-config"
        )
    if graphconfig:
        verify_graphconfig(config)
    else:  # implicit if tableconfig:
        verify_tableconfig(config)
    return None


@app.command()
def train(
    gold_training_data: str = typer.Option(
        ...,
        "-g",
        "--gold-training-data",
        help="a jsonlines file containing hand scored edge scoring neural net training data",
    ),
    pseudo_labeled_training_data: str = typer.Option(
        ...,
        "-p",
        "--pseudo-labeled-training-data",
        help="a jsonlines file containing pseudo labeled edge scoring neural net training data (weak supervision)",
    ),
    saveto: str = typer.Option(
        "resources/weights",
        "-s",
        "--save-to",
        help="a path to the file you want to save weights to",
    ),
    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="the number of epochs to train model"
    ),
) -> None:
    """Trains the Neural Network for Edge Scoring with Specified JSONLINES"""
    trainscoringmodel(gold_training_data, pseudo_labeled_training_data, saveto, epochs)
    return None


# wrapper for poety entrypoint
def cli() -> None:
    app()
    return None
