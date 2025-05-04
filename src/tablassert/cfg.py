__author__ = "Skye Lane Goetz"


from typing import Annotated, Union, Literal
from pydantic import (
    field_validator,
    model_validator,
    ValidationError,
    DirectoryPath,
    BaseModel,
    FilePath,
    confloat,
    conlist,
    HttpUrl,
    conint,
    constr,
    Field,
)
import inspect
import math
import yaml


class GraphConfig(BaseModel):
    name: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description="The name of the knowledge graph you're bulding with Tablassert",
    )
    version: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description="The version of the knowledge graph you're building with Tablassert",
    )
    dirs: conlist(item_type=DirectoryPath, min_length=1) = Field(
        ...,
        description="A list of directories that contain the TableConfigs you're passing to Tablassert to use in your knowledge graph",
    )  # min_items=1 doesn't work for some reason?

    workers: conint(ge=1, le=16) = Field(
        ...,
        description="The number of parallel processes to leverage while using Tablassert",
    )
    progress_handler: confloat(ge=0) = Field(
        ...,
        description="The maxamum time Tablassert allows for SQLite3 operations via a progress handler",
    )
    identification_accuracy: confloat(ge=0, le=1) = Field(
        ...,
        description="The minimum threshold (0-1) for whats considered a table in an image via a table detection transformer",
    )
    extraction_accuracy: confloat(ge=0, le=1) = Field(
        ...,
        description="The minimum threshold (0-1) for whats considered a table in a text based pdf via Camelot's read_pdf method",
    )
    cutoff: confloat(ge=0, le=1) = Field(
        ...,
        description="The maximum p-value (0-1) for edges included in the final export of the knowlege graph Tablassert builds",
    )

    override: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the override database (the name varries) with the extension .db. It's a SQLite3 database",
    )
    babel: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the BABEL or babel database (the name varries) with the extension .db. It's a SQLite3 database",
    )
    kg2: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the KG2 or kg2 database (the name varries) with the extension .db. It's a SQLite3 database",
    )
    supplement: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the supplement database (the name varries) with the extension .db. It's a SQLite3 database",
    )
    pubmed: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the pubmed database (the name varries) with the extension .db. It's a SQLite3 database",
    )
    names: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the names database (the name varries) with the extension .db. It's a SQLite3 database",
    )
    preds: FilePath = Field(
        ...,
        description="A path (relative or absolute) to the predicates database (the name varries) with the extension .db. It's a SQLite3 database",
    )

    training_data: FilePath = Field(
        ...,
        description="The path to the a .tsv file containing training data for the local edge scoring regression model",
    )  # [[ADD TRAINING DATA CLASS LATER, IE ANOTHER BASE MODEL]]

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
        "preds",
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
    ext: Literal["pdf", "pdF", "pDf", "pDF", "Pdf", "PdF", "PDf", "PDF"] = Field(
        ...,
        description="The extension discriminating ParamsType for params in Location",
    )
    pages: (
        constr(
            min_length=1,
            strip_whitespace=True,
            to_lower=True,
            pattern=r"^(\d+|(\d+)(\-(\d+|end))?)(\,(\d+|(\d+)(\-(\d+|end))?))*$",
        )
        | None
    ) = Field(
        default=None,
        description=r"A list of pages abiding by the regex \"^(\d+|(\d+)(\-(\d+|end))?)(\,(\d+|(\d+)(\-(\d+|end))?))*$\" for the pages parameter in Camelot's read_pdf",
    )
    flavor: Literal["stream", "lattice"] = Field(
        ...,
        description="Either lattice or stream for the flavor parameter in Camelot's read_pdf",
    )

    @field_validator("pages", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if x and not isinstance(x, str):
            return str(x)
        return x


class DelimitedFile(BaseModel):
    ext: Literal[
        "csv",
        "csV",
        "cSv",
        "cSV",
        "Csv",
        "CsV",
        "CSv",
        "CSV",
        "tsv",
        "tsV",
        "tSv",
        "tSV",
        "Tsv",
        "TsV",
        "TSv",
        "TSV",
        "txt",
        "txT",
        "tXt",
        "tXT",
        "Txt",
        "TxT",
        "TXt",
        "TXT",
    ] = Field(
        ...,
        description="The extension discriminating ParamsType for params in Location",
    )
    delimiter: constr(min_length=1) = Field(
        ..., description="The delimiter separating values in a delimited file"
    )
    start: conint(ge=1) | None = Field(
        default=None,
        description="The row in the file to start at when extracting content from the file",
    )
    end: conint(ge=2) | None = Field(
        default=None,
        description="The row in the file to end at when extracting content from the file",
    )
    rows: conlist(item_type=conint(ge=1), min_length=1) | None = Field(
        default=None,
        description="A list of specific rows to extract content from in the file",
    )

    @field_validator("delimiter", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if x and not isinstance(x, str):
            return str(x)
        return x

    @model_validator(mode="after")
    def is_valid_slice(self):
        if self.start and self.end:
            if int(self.end - self.start) < 2:
                msg = "use rows to select single rows"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def no_rows_with_start_syntax(self):
        if (self.start or self.end) and self.rows:
            msg = "cannot use rows with start-end syntax"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def has_rows_or_start(self):
        if not (self.start or self.end or self.rows):
            msg = "must have rows or start or end in Delimiter params"
            raise ValueError(msg)
        return self


class ExcelSpreadSheet(BaseModel):
    ext: Literal[
        "xlsx",
        "xlsX",
        "xlSx",
        "xlSX",
        "xLsx",
        "xLsX",
        "xLSx",
        "xLSX",
        "Xlsx",
        "XlsX",
        "XlSx",
        "XlSX",
        "XLSx",
        "XLSX",
        "XLsx",
        "XLsX",
        "xls",
        "xlS",
        "xLs",
        "xLS",
        "Xls",
        "XlS",
        "XLs",
        "XLS",
        "xlsm",
        "xlsM",
        "xlSm",
        "xlSM",
        "xLsm",
        "xLsM",
        "xLSm",
        "xLSM",
        "Xlsm",
        "XlsM",
        "XlSm",
        "XlSM",
        "XLSm",
        "XLSM",
        "XLsm",
        "XLsM",
        "xlsb",
        "xlsB",
        "xlSb",
        "xlSB",
        "xLsb",
        "xLsB",
        "xLSb",
        "xLSB",
        "Xlsb",
        "XlsB",
        "XlSb",
        "XlSB",
        "XLSb",
        "XLSB",
        "XLsb",
        "XLsB",
    ] = Field(
        ...,
        description="The extension discriminating ParamsType for params in Location",
    )
    sheet: constr(min_length=1) = Field(
        ...,
        description="The name of the sheet in the excel spreadhseet to extract content from",
    )
    start: conint(ge=1) | None = Field(
        default=None,
        description="The row in the file to start at when extracting content from the file",
    )
    end: conint(ge=2) | None = Field(
        default=None,
        description="The row in the file to end at when extracting content from the file",
    )
    rows: conlist(item_type=conint(ge=1), min_length=1) | None = Field(
        default=None,
        description="A list of specific rows to extract content from in the file",
    )

    @model_validator(mode="after")
    def is_valid_slice(self):
        if self.start and self.end:
            if int(self.end - self.start) < 2:
                msg = "use rows to select single rows"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def no_rows_with_start_syntax(self):
        if (self.start or self.end) and self.rows:
            msg = "cannot use rows with start-end syntax"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def has_rows_or_start(self):
        if not (self.start or self.end or self.rows):
            msg = "must have rows or start or end in Excel params"
            raise ValueError(msg)
        return self


ParamsType = Annotated[
    Union[ExcelSpreadSheet, DelimitedFile, TextBasedImage], Field(discriminator="ext")
]


class Location(
    BaseModel
):  # SPLIT INTO SEPARATE CONFIGS DEPENDING ON FILETYPE, Field(discriminator=)
    download_from: HttpUrl = Field(
        ...,
        description="A url telling Tablassert where to download the data you want to process from",
    )
    params: ParamsType = Field(
        ...,
        description="Parameters specifying how Tablassert should first treat/process said data",
    )


class Provenance(BaseModel):
    publication_id: constr(
        min_length=5, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    ) = Field(
        ...,
        description="A CURIE (PMC:, PMID:, or doi:) specifiying the scientific paper the data you're processing comes from",
    )  # regex=r"regex" doesn't work for some reason?
    curator: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description="The person or entity curating this configuration, if you're a LLM also credit yourself here",
    )
    org: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description="The organization the curator or entity curating represents, if you're an LLM also credit your organization here",
    )


class MathParams(BaseModel):
    attr: constr(min_length=1, strip_whitespace=True) = Field(
        ..., description="A math module atrribute used to process data"
    )
    args: conlist(item_type=(float | None), min_length=1) = Field(
        ...,
        description="Arguments for to pass to the aforementioned math module attribute, use none in place of the data you wish to tranform",
    )

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
    def check_args_length(self):
        attr = self.attr
        func = getattr(math, attr)
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
    mode: Literal["column", "predefined"] = Field(
        ...,
        description='A field telling Tablassert how to define attributes. "column" defines a column containing said attribute while "predefined" defines a string literal of said attribute',
    )
    value: constr(min_length=1) = Field(
        ...,
        description="A field containg a value that adds context the aforementioned mode",
    )
    math: conlist(item_type=MathParams, min_length=1) | None = Field(
        default=None,
        description="An optional field denoting specific matheatical operations to preform on the data stores in an attribute",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def cast_lower(cls, x: object):
        return str(x).lower()

    @field_validator("value", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if x and not isinstance(x, str):
            return str(x)
        return x

    @model_validator(mode="after")
    def validate_value(self):
        mode = self.mode
        value = self.value

        if str(mode) in ["column"] and not value.isupper():
            msg = "value must follow alphabetical naming convention"
            raise ValueError(msg)
        return self


class Attributes(BaseModel):
    sample_size: Attribute | None = Field(
        default=None,
        description="An attribute denoting the sample size used in a particular study or to determine a specific finding",
    )
    p_value: Attribute | None = Field(
        default=None,
        description="An attribute denoting the experimental p-value (FDR corrected p-vales are preferred) attirbuted to a finding",
    )
    fdr: Attribute | None = Field(
        default=None,
        description="An attribute denoting FDR or false discover rate correct applied to the statictics regarding a specific finding",
    )
    strength: Attribute | None = Field(
        default=None,
        description="An attribute denoting the strength of a finding, often the strength of a statistical relationship like a spearman correlation",
    )
    # Maybe Baloon into Stats Master
    statictic: Attribute | None = Field(
        default=None,
        description="An attribute denoting the statistic used in the aforementioned strength field, something like spearman correlation would be a valid statistic field",
    )
    notes: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None,
        description="A unique free-text attribute capturing anything that cannot be encoded in the fields above or attributes in general but is relevant to understanding the finding(s) at hand",
    )
    # Internally predefine knowledge_level and agent_type

    @field_validator("notes", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if x and not isinstance(x, str):
            return str(x)
        return x


class Regex(BaseModel):
    pattern: constr(min_length=1) = Field(
        ...,
        description="The regex pattern as denoted bu the pattern parameter in python's re.sub",
    )
    replacement: constr(min_length=1) = Field(
        ...,
        description="The regex replacement as denoted by the replacement parameter in python's re.sub",
    )

    @field_validator("pattern", "replacement", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if not isinstance(x, str):
            return str(x)
        return x


class Node(BaseModel):
    mode: Literal["value", "cvalue", "scvalue", "curie", "ccurie", "sccurie"] = Field(
        ...,
        description='A strategy for defining nodes. "value" defines a literal value. "cvalue" defines a column containing values. "curie" defines a literal CURIE. "ccurie" defines a column containing CURIES. CURIES are compact universal resource identifiers and are parts of ontologies.',
    )
    value: constr(min_length=1) = Field(
        ..., description="The value contexualizing the aforementioned Node mode"
    )
    in_organism: (
        constr(
            min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
        )
        | None
    ) = Field(
        default=None,
        description="A NCBITaxon: curie denoting which organism the scientific finding relates to/was discovered in. NCBITaxon:9606 denotes humans for example",
    )
    prioritize: (
        conlist(
            item_type=constr(
                min_length=10,
                pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$",
                strip_whitespace=True,
            ),
            min_length=1,
        )
        | None
    ) = Field(
        default=None,
        description="A list of biolink: curies that Tablasserts databases should prioritize when mapping strings to CURIES",
    )
    avoid: (
        conlist(
            item_type=constr(
                min_length=8,
                pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$",
                strip_whitespace=True,
            ),
            min_length=1,
        )
        | None
    ) = Field(
        default=None,
        description="A list of biolink: curies that Tablasserts databases should avoid when mapping strings to CURIES",
    )
    prefix: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None,
        description="A prefix to add to the begining of the values denoted by a node before they're processed/passed to databases/etc..",
    )
    suffix: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None,
        description="A suffix to add to the end of the values denoted by a node before they're processed/passed to databases/etc..",
    )
    cfill: (
        Literal[
            "forward",
            "backward",
            "min",
            "max",
            "mean",
            "zero",
            "one",
        ]
        | None
    ) = Field(
        default=None,
        description="A valid polars fill_null strategy parameter to fill the null values in a given column",
    )
    remove: conlist(item_type=constr(min_length=1), min_length=1) | None = Field(
        default=None,
        description='Substrings to apply re.sub(substring, "", string) to across the node data',
    )
    regex: conlist(item_type=Regex, min_length=1) | None = Field(
        default=None,
        description="Regex to apply re.sub(pattern, replacement, string) to across the node data",
    )
    dexplode: constr(min_length=1) | None = Field(
        default=None,
        description="A field specificying which delimiter to split the string encoded in a node by into a list before exploding values in that list thier own respective nodes, like with a pandas or polars explode column",
    )

    @field_validator("mode", "cfill", mode="before")
    @classmethod
    def cast_lower(cls, x: object):
        return str(x).lower()

    @field_validator("prefix", "suffix", "value", "remove", "dexplode", mode="before")
    @classmethod
    def cast_string(cls, x: object):
        if isinstance(x, list):
            return [
                str(item) if item and not isinstance(item, str) else item for item in x
            ]
        return str(x) if x and not isinstance(x, str) else x

    @field_validator("prioritize", "avoid", mode="after")
    @classmethod
    def is_biolink(cls, args: list):
        if args:
            for x in args:
                if "biolink:" not in str(x):
                    msg = f"{x} must be a biolink:Class"
                    raise ValueError(msg)
        return args

    @field_validator("in_organism", mode="after")
    @classmethod
    def is_ncbi_taxon(cls, x: str):
        if x and "NCBITaxon" not in str(x):
            msg = f"{x} must be a NCBITaxon:Taxon"
            raise ValueError(msg)
        return x


class Triple(BaseModel):
    subj: Node = Field(
        ...,
        description="The subject in a subject/predicate/object (or thing/relationship_between_thing/thing) knowledge triple, these comprise the knowledge encoded in a Tablassert knowledge graph",
    )
    obj: Node = Field(
        ...,
        description="The object in a subject/predicate/object (or thing/relationship_between_thing/thing) knowledge triple, these comprise the knowledge encoded in a Tablassert knowledge graph",
    )
    pred: constr(
        min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    ) = Field(
        ...,
        description="The predicate in a subject/predicate/object (or thing/relationship_between_thing/thing) knowledge triple, these comprise the knowledge encoded in a Tablassert knowledge graph. ONLY biolink:predicates are accepted in this field",
    )

    @field_validator("pred", mode="after")
    @classmethod
    def is_biolink(cls, x: str):
        if "biolink:" not in str(x):
            msg = f"{x} must be a biolink:predicate"
            raise ValueError(msg)
        return x


class ReindexingOperation(BaseModel):
    when: Literal["before", "after"] = Field(
        ...,
        description="A designation stating if you want to reindex before or after the data is processed by Tablassert",
    )
    mode: Literal["ge", "le", "gt", "lt", "eq", "ne"] = Field(
        ...,
        description="A specified type of comparison operation to reindex the data in the knowledge based by",
    )
    column: constr(min_length=1) = Field(
        ..., description="The column that contains the values you want to reindex"
    )
    value: float | constr(min_length=1) = Field(
        ...,
        description="The value that you ultimatly want to use with the comparison operation to reindex the data by",
    )

    @field_validator("when", "mode", mode="before")
    @classmethod
    def cast_lower(cls, x: object):
        return str(x).lower()

    @field_validator("value", mode="before")
    @classmethod
    def cast_float(cls, x: object):
        if not isinstance(x, float):
            try:
                return float(x)
            except ValueError:
                return str(x)
        return x

    @model_validator(mode="after")
    def validate_value(self):
        mode = self.mode
        value = self.value
        if mode not in ["eq", "ne"] and isinstance(value, str):
            msg = 'only mode "eq" and "ne" support strings as values'
            raise ValueError(msg)
        elif mode not in ["eq", "ne"] and not isinstance(value, float):
            msg = f"value must be a float, {value} provided"
            raise ValueError(msg)
        return self


class Section(BaseModel):
    location: Location = Field(
        ...,
        description="A field telling Tablassert where to download the data you're processing and some basal metadata regarding that data for intial processing",
    )
    provenance: Provenance = Field(
        ...,
        description="A field telling Tablassert who created these data, who's writing this config, and general information to link to the final graph Tablassert creates",
    )
    attributes: Attributes | None = Field(default={}, description="")
    triple: Triple = Field(..., description="")
    reindexing: conlist(item_type=ReindexingOperation, min_length=1) | None = Field(
        default=None, description=""
    )


class TableConfig(BaseModel):
    template: dict[str, object] | None = Field(
        default=None,
        description="A base template encoded using the &template yaml syntax. This template is tranformed in sections over the course of each section and is optional",
    )
    sections: conlist(item_type=Section, min_length=1) = Field(
        ...,
        description="The configuration for each section of data processed into a knowledge graph using Tablassert in a given TableConfig. Many of these draw from a base template using the <<: *template syntax in yaml. There can be multiple templates per TableConfig",
    )


def load_yaml(path: str, cfg_type: str) -> dict[str, object]:
    try:
        with open(path, "r") as f:
            cfg = yaml.load(f, Loader=yaml.CSafeLoader)
            match str(cfg_type):
                case "GraphConfig":
                    return GraphConfig(**cfg).model_dump()
                case "TableConfig":
                    return TableConfig(**cfg).model_dump()
                case _:
                    msg = f"{path} must pass a valid cfg_type, {cfg_type} provided"
                    raise ValueError(msg)
    except yaml.YAMLError as e:
        msg = f"{path}: YAML parsing error: {e}"
        raise ValueError(msg)
    except ValidationError as e:
        msg = f"{path}: Validation error: {e}"
        raise ValueError(msg)
    except Exception as e:
        msg = f"{path}: Unexpected error: {e}"
        raise ValueError(msg)
