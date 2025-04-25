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
import inspect
import math


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
    params: ExcelSpreadSheet | DelimitedFile | TextBasedImage

    @model_validator(mode="after")
    @classmethod
    def validate_extension_paramss(self):
        ext = self.ext
        params = self.params

        if isinstance(params, ExcelSpreadSheet) and ext not in [
            "xlsx",
            "xls",
            "xlsm",
            "xlsb",
        ]:
            msg = f'Extension "{ext}" is not valid for Excel files'
            raise ValueError(msg)
        if isinstance(params, DelimitedFile) and ext not in ["csv", "tsv", "txt"]:
            msg = f'Extension "{ext}" is not valid for delimited files'
            raise ValueError(msg)
        if isinstance(params, TextBasedImage) and ext != "pdf":
            msg = f'Extension "{ext}" is not valid for text based image files'
            raise ValueError(msg)
        return self


class Provenance(BaseModel):
    publication_id: constr(
        min_length=1, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    )  # regex=r"regex" doesn't work for some reason?
    curator: constr(min_length=1, strip_whitespace=True)
    org: constr(min_length=1, strip_whitespace=True)


class MathParams(BaseModel):
    attr: constr(min_length=1, strip_whitespace=True)
    args: conlist(min_length=1)

    @field_validator("args", mode="before")
    @classmethod
    def cast_float(cls, args: object):
        processed = []
        for x in args:
            if x and not isinstance(x, float):
                try:
                    processed.append(float(x))
                except ValueError:
                    msg = f"{x} must be null or a float"
                    raise ValueError(msg)
            else:
                processed.append(x)
        return processed

    @field_validator("attr", mode="after")
    @classmethod
    def is_attr(cls, x: str):
        if not hasattr(math, x):
            msg = f"{x} must be a valid attr from the math module"
            raise ValueError(msg)
        return x

    @model_validator(mode="after")
    @classmethod
    def check_args_length(self):
        attr = self.attr
        func = math.getattr(math, attr)
        sig = inspect.signature(func)
        params = sig.parameters
        max_args = len(params)
        min_args = sum(
            1
            for param in params.values()
            if param.default == inspect.Parameter.empty
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        )
        specified = len(self.args)
        if max_args < specified or min_args > specified:
            msg = f"{attr} requires {min_args}-{max_args} arguments, {specified} specified"
            raise ValueError(msg)
        return self


class Attribute(BaseModel):
    mode: constr(min_length=6, max_length=10, to_lower=True, strip_whitespace=True)
    value: constr(min_length=1)
    math: conlist(item_type=MathParams, min_length=1) | None = Field(default=None)

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
    sample_size: Attribute | None = Field(default=None)
    p_value: Attribute | None = Field(default=None)
    fdr: Attribute | None = Field(default=None)
    strength: Attribute | None = Field(default=None)
    # Maybe Baloon into Stats Master
    statictic: Attribute | None = Field(default=None)
    notes: constr(min_length=1, strip_whitespace=True) | None = Field(default=None)
    # Internally predefine knowledge_level and agent_type


class Regex(BaseModel):
    pattern: constr(min_length=1)
    replacement: constr(min_length=1)

    @field_validator("pattern", "replacement", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if not isinstance(x, str):
            return str(x)
        return x


class Node(BaseModel):
    mode: constr(min_length=5, max_length=7, to_lower=True, strip_whitespace=True)
    value: constr(min_length=1)
    in_organism: constr(
        min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    )
    prioritize: conlist(
        item_type=constr(
            min_length=10, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
        ),
        min_length=1,
    )
    avoid: conlist(
        item_type=constr(
            min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
        ),
        min_length=1,
    )
    prefix: constr(min_length=1, strip_whitespace=True) | None = Field(default=None)
    suffix: constr(min_length=1, strip_whitespace=True) | None = Field(default=None)
    cfill: (
        constr(
            min_length=1, pattern=r"^[A-Za-z]+$", to_lower=True, strip_whitespace=True
        )
        | None
    ) = Field(default=None)
    remove: conlist(item_type=constr(min_length=1), min_length=1) | None = Field(
        default=None
    )
    regex: conlist(item_type=Regex, min_length=1) | None = Field(default=None)
    dexplode: constr(min_length=1)

    @field_validator("prefix", "suffix", "value", "remove", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if isinstance(x, list):
            return [
                str(item) if item and not isinstance(item, str) else item for item in x
            ]
        return str(x) if x and not isinstance(x, str) else x

    @field_validator("prioritize", "avoid", method="after")
    @classmethod
    def is_biolink(cls, args: list):
        for x in args:
            if "biolink:" not in str(x):
                msg = f"{x} must be a biolink:Class"
                raise ValueError(msg)
        return args

    @field_validator("cfill", method="after")
    @classmethod
    def is_strategy(cls, x: str):
        strategies = [
            "forward",
            "backward",
            "min",
            "max",
            "mean",
            "zero",
            "one",
        ]
        if x and str(x) not in strategies:
            msg = f"{x} must be a polars fill_null strategy"
            raise ValueError(msg)
        return x

    @field_validator("mode", method="after")
    @classmethod
    def is_mode(cls, x: str):
        modes = ["value", "cvalue", "scvalue", "curie", "ccurie", "sccurie"]
        if x not in modes:
            msg = f"must be a valid mode {modes}, {x} provided"
            raise ValueError(msg)
        return x


class Triple(BaseModel):
    subject: Node
    obj: Node
    predicate: constr(
        min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    )


class TableConfig(BaseModel):
    template: dict[str, object] | None = Field(default=None)
    sections: list[dict[str, object]]
