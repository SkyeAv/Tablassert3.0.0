from torch.utils.data import Dataset, DataLoader, TensorDataset
from src.tablassert.scoring.config import SEED, DEVICE
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import OrdinalEncoder
from functools import lru_cache
from typing import Self, Any
from pathlib import Path
from torchdr import UMAP
from torch import nn
import polars as pl
import numpy as np
import joblib
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
    inputs = TOKENIZER(x, return_tensors="pt", truncation=True, max_length=512).to(
        DEVICE
    )
    with torch.no_grad():
        outputs = BIOBERT_MODEL(**inputs)
        # I only need the pooler_output for embeddings
        pooler_output = outputs.pooler_output
        if pooler_output is None or pooler_output.numel() == 0:
            raise RuntimeError(f"CODE:200 | No pooler_output from BioBert {x}")
        return (  # type: ignore
            pooler_output.detach().cpu().float().squeeze()
        )  # shape: (768,)


CACHE: Path = Path("TABLASSERT/CACHE").resolve()
CACHE.mkdir(parents=True, exist_ok=True)


def label_encoder(
    df: pl.DataFrame, column: str, savepath: Path, mode: str
) -> np.ndarray:
    x: np.ndarray = df.select(pl.col(column)).to_numpy()
    encoderpath: Path = CACHE / "ENCODER" / savepath.stem / f"{column}.pkl"
    encoderpath.parent.mkdir(parents=True, exist_ok=True)
    if mode == "training":
        encoder: OrdinalEncoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        encoded_values: np.ndarray = encoder.fit_transform(x).ravel().astype(float)
        joblib.dump(encoder, encoderpath)
        return encoded_values
    elif mode == "production" and not encoderpath.exists():
        raise RuntimeError(
            f"CODE:203 | LabelEncoder {encoderpath.as_posix()} is missing"
        )
    elif mode == "production":
        encoder = joblib.load(encoderpath)
        return encoder.transform(x).ravel().astype(float)  # type: ignore
    else:
        raise RuntimeError(f"CODE:205A | Invalid mode: {mode}")


class EdgeScoringData(Dataset):  # type: ignore
    def __init__(self: Self, X: torch.Tensor, y: np.ndarray) -> None:
        self.X = X.detach().clone().float()
        self.y = torch.tensor(y, dtype=torch.float32)
        return None

    def __len__(self: Self) -> int:
        return len(self.X)

    def __getitem__(self: Self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def encode_data(df: pl.DataFrame, savepath: Path, mode: str) -> Dataset:  # type: ignore
    numeric: list[str] = ["sample_size", "relationship_strength"]
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
        "supplementary_file_caption",
    ]

    structured_encodings: list[torch.Tensor] = []
    for column in numeric:
        column_array: np.ndarray = (
            df.select(pl.col(column).cast(pl.Float64, strict=False).fill_null(0))
            .to_numpy()
            .ravel()
            .astype(float)
        )
        column_tensor: torch.Tensor = torch.tensor(column_array, dtype=torch.float32)
        structured_encodings.append(column_tensor)
    for column in categorical:
        column_array = label_encoder(df, column, savepath, mode)
        column_tensor = torch.tensor(column_array, dtype=torch.float32)
        structured_encodings.append(column_tensor)
    structured_tensor: torch.Tensor = torch.stack(structured_encodings, dim=1)

    freetext_embeddings: list[torch.Tensor] = []
    for row in df.select(freetext).rows():
        embedded_row = torch.cat([biobert_embedding(str(cell)) for cell in row])
        freetext_embeddings.append(embedded_row)
    freetext_tensor: torch.Tensor = torch.stack(freetext_embeddings)
    X = torch.cat([structured_tensor, freetext_tensor], dim=1)  # shape: (2312,)

    umapdrpath: Path = CACHE / "UMAP_DR" / f"{savepath.stem}.pkl"
    umapdrpath.parent.mkdir(parents=True, exist_ok=True)

    if mode == "training":
        umap_dr = UMAP(
            n_neighbors=32,
            n_components=32,
            random_state=SEED,
            backend="torch",
            device=DEVICE,
        ).fit(X)
        X = umap_dr.fit_transform(X)  # shape: (32,)
        joblib.dump(umap_dr, umapdrpath)
        y = df.select(pl.col("score")).to_numpy().reshape(-1, 1).astype(float)
        return EdgeScoringData(X, y)
    elif mode == "production" and not umapdrpath.exists():
        RuntimeError(
            f"CODE:206 | UMAP dimensionality reduction {umapdrpath.as_posix()} is missing"
        )
    elif mode == "production":
        umap_dr = joblib.load(umapdrpath)
        X = umap_dr.fit_transform(
            X
        )  # shape: (32,)  # fit transform is okay because the parameters in the initializaion are fixed
        return TensorDataset(X)
    else:
        raise RuntimeError(f"CODE:205C | Invalid mode: {mode}")


def load_data(dataset: Dataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:  # type: ignore
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ScoringRegression(nn.Module):
    def __init__(
        self: Self,
        in_dim: int = 32,
        hidden1: int = 16,
        hidden2: int = 8,
        out_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm1d(in_dim)  # for 32 dimensions, replaces std-scaler
        self.shortcut = nn.Linear(in_dim, hidden2, bias=False)
        self.block = nn.Sequential(
            # this preforms better with the extra layer
            nn.Linear(in_dim, hidden1),  # 32 is the shape of the UMAP-ed input
            nn.LeakyReLU(),  # alpha = 0.1 by default
            # this was overfitting before
            nn.Dropout(dropout),  # put between densest layers
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden2, out_dim),
            nn.Softplus(),  # for non-negative values
        )
        self._init_weights()
        return None

    def _init_weights(self: Self) -> None:  # zeros biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return None

    def forward(self: Self, x: torch.Tensor) -> Any:
        x = self.bn0(x)
        h = self.block(x)
        skip = self.shortcut(x)
        raw_out = self.head(h + skip)
        return torch.clamp(raw_out, 0.0, 100.0)
