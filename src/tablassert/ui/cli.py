from src.tablassert.scoring.training import trainscoringmodel
from src.tablassert.core.build import buildgraph
import typer

app = typer.Typer()


@app.command()
def build(graphconfig: str) -> None:
    # typer uses docstrings for command descriptions
    """
    Build a Knowledge Graph with a GraphConfig
    ---
    graphconfig: path to graphconfig
    """
    buildgraph(graphconfig)
    return None


@app.command()
def train(trainingdata: str, epochs: int = 25, saveto: str = "scoring_model.pt") -> None:
    """
    Trains the Neural Network for Edge Scoring with Specified JSONLINES
    ---
    trainingdata: a jsonlines file with the following schema
    {
        "significant":
        "sample_size":
        "multiple_testing_correction_method":
        "assertion_strength":
        "assertion_method":
        "notes":
        "pmc_file_caption":
        "subject_mapped_with_database":
        "subject_mapped_with_level":
        "object_mapped_with_database":
        "object_mapped_with_level":
        "score":
    }
    epochs: the number of epochs to train model (default=25)
    saveto: a path to the file you want to save weights to (default="scoring_model.pt")
    """
    trainscoringmodel(trainingdata, epochs, saveto)
    return None


# wrapper for poety entrypoint
def cli() -> None:
    app()
    return None
