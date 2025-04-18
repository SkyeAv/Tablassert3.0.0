# 2025 Skye Lane Goetz

from pydantic import (
    field_validator,
    DirectoryPath,
    BaseModel,
    FilePath,
    confloat,
    conlist,
    FileUrl,
    conint,
    constr,
)


class GraphConfig(BaseModel):
    name: constr(min_length=1, strip_whitespace=True)
    version: constr(min_length=1, strip_whitespace=True)
    dirs: conlist(
        item_type=DirectoryPath, min_length=1
    )  # min_items=1 doesn't work for some reason?

    workers: conint(ge=1, le=16)
    progress_handler: confloat(ge=0)
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

    @field_validator("progress_handler", mode="before")
    def cast_string(cls, x: object):
        if not isinstance(x, str):
            return str(x)
        return x

    @field_validator("progress_handler", mode="before")
    def cast_float(cls, x: object):
        if not isinstance(x, float):
            try:
                return float(x)
            except ValueError:
                pass
        return x

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


class Location(
    BaseModel
):  # SPLIT INTO SEPARATE CONFIGS DEPENDING ON FILETYPE, Field(discriminator=)
    download: FileUrl
    ext: constr(min_length=1, strip_whitespace=True, to_lower=True)
    # param: Use "|" for all the parm options and add classes for them all


class Provenance(BaseModel):
    publication_id: constr(
        min_length=1, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    )  # regex=r"regex" doesn't work for some reason?
    curator: constr(min_length=1, strip_whitespace=True)
    org: constr(min_length=1, strip_whitespace=True)
