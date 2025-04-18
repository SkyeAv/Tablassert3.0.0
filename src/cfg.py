# 2025 Skye Lane Goetz

from pydantic import (
    field_validator,
    DirectoryPath,
    BaseModel,
    FilePath,
    confloat,
    conlist,
    conint,
)


class GraphConfig(BaseModel):
    name: str
    version: str
    dirs: conlist(
        item_type=DirectoryPath, min_length=1
    )  # min_items=1 doesn't work for some reason?

    workers: conint(ge=1)
    progress_handler: float
    identification_accuracy: confloat(ge=0, le=1)
    extraction_accuracy: confloat(ge=0, le=1)
    cutoff: confloat(ge=0, le=1)

    override: FilePath
    babel: FilePath
    kg2: FilePath
    supplement: FilePath
    pubmed: FilePath
    names: FilePath
    predicates: FilePath

    training_data: FilePath  # ADD TRAINING DATA CLASS LATER, IE ANOTHER BASE MODEL

    @field_validator(
        "override",
        "babel",
        "kg2",
        "supplement",
        "pubmed",
        "names",
        "predicates",
        mode="after",
    )
    def is_sqlite(cls, db: str):
        with open(db, "rb") as f:
            header = f.read(16)
        if header[:16] != b"SQLite format 3\x00":
            msg = "must be a sqlite database"
            raise ValueError(msg)
        return db
