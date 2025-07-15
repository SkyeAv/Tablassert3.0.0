from src.tablassert.scoring.preprocessing import encode_data, load_data
from torch.utils.data import Dataset, DataLoader, random_split
from src.tablassert.scoring.config import SEED, DEVICE
from collections import OrderedDict
from typing import Any, Self
from loguru import logger
from pathlib import Path
from torch import nn
import polars as pl
import numpy as np
import random
import torch

# set seed for reproducability
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# setup training.log
TRAINING_LOG_PATH: Path = Path("TABLASSERT/LOG/training.log").resolve()
TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    TRAINING_LOG_PATH.as_posix(),
    rotation="250 MB",
    compression="xz",
    retention="1 month",
)


def read_jsonl(jsonlpath: Path) -> pl.DataFrame:
    training_data_schema: dict[str, Any] = {
        "significant": pl.String(),
        "sample_size": pl.String(),
        "multiple_testing_correction_method": pl.String(),
        "assertion_strength": pl.String(),
        "assertion_method": pl.String(),
        "notes": pl.String(),
        "pmc_file_caption": pl.String(),
        "subject_mapped_with_database": pl.String(),
        "subject_mapped_with_level": pl.String(),
        "object_mapped_with_database": pl.String(),
        "object_mapped_with_level": pl.String(),
        "score": pl.String(),
    }
    return pl.read_ndjson(
        source=jsonlpath,
        schema=training_data_schema,
    )


def load_training_data(dataset: Dataset) -> tuple[DataLoader, DataLoader]:  # type: ignore
    train_size: int = int(0.8 * len(dataset))  # type: ignore
    test_size: int = len(dataset) - train_size  # type: ignore
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return load_data(train_dataset), load_data(test_dataset, 64, False)


class ScoringRegression(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # this preforms better with the extra layer
            nn.Linear(2312, 32),  # 3079 is the shape of the input
            nn.LeakyReLU(),
            # this was overfitting before
            nn.Dropout(0.2),  # put between densest layers
            nn.Linear(32, 16),
            nn.LeakyReLU(),  # alpha = 0.1 by default
            nn.Linear(16, 1),
        )
        return None

    def forward(self: Self, x: torch.Tensor) -> Any:
        return self.model(x)


MODEL = ScoringRegression().to(DEVICE)
LOSS_FN = nn.SmoothL1Loss(beta=1.0)  # Huber Loss
OPTIMIZER = torch.optim.Adam(
    MODEL.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0,
)


def training_loop(train_dataloader: DataLoader, test_dataloader: DataLoader, epochs: int) -> OrderedDict[str, torch.Tensor]:  # type: ignore
    for epoch in range(epochs):

        MODEL.train()
        training_loss: float = 0.0
        for xb, yb in train_dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            OPTIMIZER.zero_grad()
            preds = MODEL(xb)
            loss = LOSS_FN(preds, yb)
            loss.backward()
            OPTIMIZER.step()
            training_loss += loss.item() * xb.size(0)
        average_training_loss: float = training_loss / len(train_dataloader.dataset)  # type: ignore

        MODEL.eval()
        validation_loss: float = 0.0
        with torch.no_grad():
            for xb, yb in test_dataloader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                preds = MODEL(xb)
                loss = LOSS_FN(preds, yb)
                validation_loss += loss.item() * xb.size(0)
        average_validation_loss: float = validation_loss / len(test_dataloader.dataset)  # type: ignore
        logger.info(
            f"epochs: {epoch}/{epochs} | training loss: {average_training_loss} | validation loss: {average_validation_loss}"
        )
    return MODEL.state_dict()  # type: ignore


def trainscoringmodel(trainingdata: str, saveto: str, epochs: int) -> None:
    trainingdatapath: Path = Path(trainingdata)
    savepath: Path = Path(saveto)
    df: pl.DataFrame = read_jsonl(trainingdatapath)
    dataset: Dataset = encode_data(df)  # type: ignore
    train_dataloader, test_dataloader = load_training_data(dataset)
    weights: OrderedDict[str, torch.Tensor] = training_loop(
        train_dataloader, test_dataloader, epochs
    )
    torch.save(weights, savepath)
    return None
