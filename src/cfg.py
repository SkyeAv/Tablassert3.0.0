# 2025 Skye Lane Goetz

from pydantic import (
    field_validator,
    model_validator,
    DirectoryPath,
    BaseModel,
    FilePath,
    confloat,
    conlist,
    FileUrl,
    conint,
    constr,
    Field,
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
    @classmethod
    def cast_string(cls, x: object):
        if not isinstance(x, str):
            return str(x)
        return x

    @field_validator("progress_handler", mode="before")
    @classmethod
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
    @classmethod
    def is_sqlite(cls, db: str):
        with open(db, "rb") as f:
            header = f.read(16)
        if header[:16] != b"SQLite format 3\x00":
            msg = "must be a sqlite database"
            raise ValueError(msg)
        return db


class TextBasedImage(BaseModel):
    pages: (
        constr(
            min_length=1,
            strip_whitespace=True,
            to_lower=True,
            pattern=r"^(\d+|(\d+)(\-(\d+|end))?)(\,(\d+|(\d+)(\-(\d+|end))?))*$",
        )
        | None
    ) = Field(default=None)
    flavor: constr(min_length=6, max_length=7)

    @field_validator("pages", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if x and not isinstance(x, str):
            return str(x)
        return x

    @field_validator("flavor", mode="after")
    @classmethod
    def is_flavor(cls, x: str):
        if str(x) not in ["stream", "lattice"]:
            msg = "must be a valid camelot flavor"
            raise ValueError(msg)
        return x


class DelimitedFile(BaseModel):
    delimeter: constr(min_length=1)


class ExcelSpreadSheet(BaseModel):
    sheet: constr(min_length=1)
    start: conint(ge=1) | None = Field(default=None)
    end: conint(ge=2) | None = Field(default=None)
    rows: conlist(item_type=conint(ge=1), min_length=1) | None = Field(default=None)

    @model_validator(mode="after")
    @classmethod
    def is_valid_slice(self):
        if self.start and self.end:
            if int(self.end - self.start) < 2:
                msg = "use rows to select single rows"
                raise ValueError(msg)
        return self


class Location(
    BaseModel
):  # SPLIT INTO SEPARATE CONFIGS DEPENDING ON FILETYPE, Field(discriminator=)
    download: FileUrl
    ext: constr(
        min_length=1, strip_whitespace=True, to_lower=True, pattern=r"/^\s*(\w+)\s*$/i"
    )
    param: ExcelSpreadSheet | DelimitedFile | TextBasedImage

    @model_validator(mode="after")
    @classmethod
    def validate_extension_params(self):
        ext = self.ext
        param = self.param

        if isinstance(param, ExcelSpreadSheet) and ext not in [
            "xlsx",
            "xls",
            "xlsm",
            "xlsb",
        ]:
            msg = f'Extension "{ext}" is not valid for Excel files'
            raise ValueError(msg)
        if isinstance(param, DelimitedFile) and ext not in ["csv", "tsv", "txt"]:
            msg = f'Extension "{ext}" is not valid for delimited files'
            raise ValueError(msg)
        if isinstance(param, TextBasedImage) and ext != "pdf":
            msg = f'Extension "{ext}" is not valid for text based image files'
            raise ValueError(msg)
        return self


class Provenance(BaseModel):
    publication_id: constr(
        min_length=1, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    )  # regex=r"regex" doesn't work for some reason?
    curator: constr(min_length=1, strip_whitespace=True)
    org: constr(min_length=1, strip_whitespace=True)


class Attribute(BaseModel):
    mode: constr(min_length=6, max_length=10, to_lower=True, strip_whitespace=True)
    value: constr(min_length=1)
    # Include Something For the Math Parameter Here

    @field_validator("mode", mode="after")
    @classmethod
    def is_mode(cls, x: str):
        if str(x) not in ["column", "predefined"]:
            msg = 'must be "column" or "predefined"'
            raise ValueError(msg)
        return x

    @model_validator(mode="after")
    @classmethod
    def validate_value(self):
        mode = self.mode
        value = self.value

        if str(mode) == "column" and not value.isupper():
            msg = "value must follow alphabetical naming convention"
            raise ValueError(msg)
        return self


class Attributes(BaseModel):
    sample: Attribute
    p_value: Attribute
    fdr: Attribute
    strength: Attribute
    statictic: Attribute  # Maybe Baloon into Stats Master
    # Internally predefine knowledge_level and agent_type
