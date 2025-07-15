from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache
from typing import Self, Any
from pathlib import Path
from os import environ
from torch import nn
import pandas as pd
import numpy as np
import torch

environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")  # set torch to cpu for caching

SEED: int = 87
torch.manual_seed(SEED)  # set seed for reproducability

BIOBERT = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(BIOBERT)  # type:ignore
model = AutoModel.from_pretrained(BIOBERT).to(DEVICE)
model.eval()  # disables dropout for embeddings


@lru_cache(maxsize=16)
def biobertembedding(x: str) -> Any:
    tokens = tokenizer(x, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model(**tokens)

        pooler_output = output.pooler_output
        assert pooler_output is not None, "Missing pooler_output"
        return pooler_output.detach().cpu().float().squeeze()  # shape: (768,)  # type: ignore


def encodedata(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    numericcols = ["sample_size", "assertion_strength"]
    categoricalcols = [
        "significant",
        "assertion_method",
        "subject_mapped_with_database",
        "subject_mapped_with_level",
        "object_mapped_with_database",
        "object_mapped_with_level",
    ]
    embeddablecols = [
        "multiple_testing_correction_method",
        "assertion_method",
        "notes",
        "pmc_file_caption",
    ]

    df[numericcols] = df[numericcols].apply(
        lambda x: pd.to_numeric(x, errors="coerce").fillna(0)
    )

    le = LabelEncoder()
    for col in categoricalcols:
        df[col] = le.fit_transform(df[col].astype(str))
    structured_feats = df[numericcols + categoricalcols].values.astype(np.float32)

    # Collect all embeddings into a list of torch.Tensor
    embeddings_per_row = []
    for _, row in df[embeddablecols].astype(str).iterrows():
        row_embedding_parts = [biobertembedding(row[col]) for col in embeddablecols]
        combined = torch.cat(row_embedding_parts)  # shape: (768 * len(embeddablecols),)
        embeddings_per_row.append(combined)

    # Stack into a 2D tensor
    embedding_tensor = torch.stack(
        embeddings_per_row
    )  # shape: (N, 768*len(embeddablecols))
    structured_tensor = torch.tensor(
        structured_feats, dtype=torch.float32
    )  # shape: (N, structured)

    X = torch.cat([structured_tensor, embedding_tensor], dim=1)
    y = torch.tensor(df["score"].values, dtype=torch.float32).unsqueeze(1)

    return X, y


def prepareinputs(X: torch.Tensor, y: torch.Tensor) -> tuple[DataLoader, DataLoader]:  # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    trainingloader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=32, shuffle=True
    )
    testingloader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)
    return trainingloader, testingloader


NUMFEATURES: int = 3080
NUMOUTPUS: int = 1


# basic linear regression model for the scoring dataset
class ScoringRegression(nn.Module):

    def __init__(self: Self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # this preforms better with the extra layer
            nn.Linear(NUMFEATURES, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),  # put between densest layers
            nn.Linear(32, 16),
            nn.LeakyReLU(),  # alpha = 0.1 by default
            nn.Linear(16, NUMOUTPUS),
        )
        return None

    def forward(self: Self, x: torch.Tensor) -> Any:
        return self.model(x)


MODEL = ScoringRegression().to(DEVICE)
REGRESSIONLOSS = nn.SmoothL1Loss(beta=1.0)
# RANKINGLOSS = nn.HuberLoss(delta=1.0) maybe try later
OPTIMIZER = torch.optim.Adam(
    MODEL.parameters(),
    lr=5e-4,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0,
)


def initiatemodel(
    epochs: int,
    trainingloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    testingloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for epoch in range(epochs):
        MODEL.train()
        total_training_loss: float = 0.0
        for xb, yb in trainingloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            OPTIMIZER.zero_grad()
            preds = MODEL(xb)
            loss = REGRESSIONLOSS(preds, yb)
            loss.backward()
            OPTIMIZER.step()

            total_training_loss += loss.item() * xb.size(0)

        average_training_loss: float = total_training_loss / float(len(trainingloader.dataset))  # type: ignore

        MODEL.eval()
        total_validation_loss: float = 0.0
        with torch.no_grad():
            for xb, yb in testingloader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                preds = MODEL(xb)
                loss = REGRESSIONLOSS(preds, yb)
                total_validation_loss += loss.item() * xb.size(0)

        average_validation_loss: float = total_validation_loss / float(
            len(testingloader.dataset)  # type: ignore
        )

        print(
            f"Epoch {epoch + 1}/{epochs} │ "
            f"Training Loss: {average_training_loss:.4f} │ "
            f"Validation Loss:   {average_validation_loss:.4f}"
        )

    # make sure to update later
    torch.save(MODEL.state_dict(), "scoring_model_epoch50.pt")
    return None


def trainscoringmodel(trainingdata: str, epochs: int) -> None:
    trainingdatapath: Path = Path(trainingdata).resolve()
    df: pd.DataFrame = pd.read_json(trainingdatapath, lines=True)
    X, y = encodedata(df)
    trainingloader, testingloader = prepareinputs(X, y)
    initiatemodel(epochs, trainingloader, testingloader)
    return None
