from src.tablassert.scoring.preprocessing import (
    ScoringRegression,
    encode_data,
    load_data,
)
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import polars as pl
import torch

MODEL = ScoringRegression()


def load_weights(weightspath: Path) -> None:
    MODEL.load_state_dict(torch.load(weightspath, map_location="cpu"))


def score_edges(df: pl.DataFrame, weightspath: Path) -> pl.DataFrame:
    load_weights(weightspath)

    dataset: Dataset = encode_data(df, weightspath, "production")  # type: ignore
    dataloader: DataLoader = load_data(dataset, batch_size=64, shuffle=False)  # type: ignore
    scores: list[float] = []

    MODEL.eval()
    with torch.no_grad():
        for batch in dataloader:

            if isinstance(batch, list):
                batch = torch.cat(batch, dim=0)  # for that one list error
                print(f"[DEBUG] batch type: {type(batch)}")
            
            outputs = MODEL(batch)
            batch_scores = outputs.squeeze()

            # if for some reason batch = 1
            if batch_scores.ndim == 0:
                batch_scores = [batch_scores.item()]
            elif batch_scores.ndim == 1:
                batch_scores = batch_scores.tolist()
            else:
                raise RuntimeError(
                    f"CODE:201 | Unexpected edge_score output shape: {outputs.shape}"
                )

            scores.extend(batch_scores)
    return df.with_columns(pl.Series(name="score", values=scores, dtype=pl.Float64))
