from src.tablassert.scoring.config import SEED, DEVICE
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache
from typing import Self
import polars as pl
import numpy as np
import random
import torch

# set seed for reproducability
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# load biobert from hugging face
BIOBERT_TRANSFORMER: str = "dmis-lab/biobert-base-cased-v1.1"
TOKENIZER = AutoTokenizer.from_pretrained(BIOBERT_TRANSFORMER)  # type: ignore
BIOBERT_MODEL = AutoModel.from_pretrained(BIOBERT_TRANSFORMER).to(DEVICE)
BIOBERT_MODEL.eval()  # disables dropout for embeddings


@lru_cache(maxsize=32)
def biobert_embedding(x: str) -> torch.Tensor:
    inputs = TOKENIZER(x, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = BIOBERT_MODEL(**inputs)
        # I only need the pooler_output for embeddings
        pooler_output = outputs.pooler_output
        if pooler_output is None or pooler_output.numel() == 0:
            raise RuntimeError(f"CODE:200 | No pooler_output from BioBert {x}")
        return (  # type: ignore
            pooler_output.detach().cpu().float().squeeze()
        )  # shape: (768,)


LABEL_ENCODER: LabelEncoder = LabelEncoder()


def label_encoder(df: pl.DataFrame, column: str) -> np.ndarray:
    x: np.ndarray = df.select(pl.col(column)).to_numpy().ravel()
    return LABEL_ENCODER.fit_transform(x).astype(float)  # type: ignore


STANDARD_SCALER: StandardScaler = StandardScaler()


def cast_to_numeric(df: pl.DataFrame, column: str) -> np.ndarray:
    x: np.ndarray = (
        df.select(pl.col(column).cast(pl.Float64, strict=False).fill_null(0))
        .to_numpy()
    )
    return STANDARD_SCALER.fit_transform(x).ravel().astype(float)  # type: ignore


class EdgeScoringData(Dataset):  # type: ignore
    def __init__(self: Self, X: torch.Tensor, y: np.ndarray) -> None:
        self.X = X.detach().clone().float()
        self.y = torch.tensor(y, dtype=torch.float32)
        return None

    def __len__(self: Self) -> int:
        return len(self.X)

    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def encode_data(df: pl.DataFrame) -> Dataset:  # type: ignore
    numeric: list[str] = ["sample_size", "assertion_strength"]
    categorical: list[str] = [
        "significant",
        "multiple_testing_correction_method",
        "subject_mapped_with_database",
        "subject_mapped_with_level",
        "object_mapped_with_database",
        "object_mapped_with_level",
    ]
    freetext: list[str] = [
        "assertion_method",
        "notes",
        "pmc_file_caption",
    ]

    structured_encodings: list[torch.Tensor] = []
    for column in numeric:
        column_array: np.ndarray = cast_to_numeric(df, column)
        column_tensor: torch.Tensor = torch.tensor(column_array, dtype=torch.float32)
        structured_encodings.append(column_tensor)
    for column in categorical:
        column_array = label_encoder(df, column)
        column_tensor = torch.tensor(column_array, dtype=torch.float32)
        structured_encodings.append(column_tensor)
    structured_tensor: torch.Tensor = torch.stack(structured_encodings, dim=1)

    freetext_embeddings: list[torch.Tensor] = []
    for row in df.select(freetext).rows():
        embedded_row = torch.cat([biobert_embedding(str(cell)) for cell in row])
        freetext_embeddings.append(embedded_row)
    freetext_tensor: torch.Tensor = torch.stack(freetext_embeddings)
    X = torch.cat([structured_tensor, freetext_tensor], dim=1)  # shape: (3079,)
    y = df.select(pl.col("score")).to_numpy().reshape(-1, 1).astype(float)
    return EdgeScoringData(X, y)


def load_data(dataset: Dataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:  # type: ignore
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
