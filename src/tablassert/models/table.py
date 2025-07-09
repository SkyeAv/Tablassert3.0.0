from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    FilePath,
    HttpUrl,
    Field,
)
from typing import Any, Self, Optional, Literal, Annotated, Union, TypeAlias
from playwright.async_api import async_playwright
from urllib.parse import urlparse
from pathlib import Path
import requests
import asyncio
import math


class Reindexing(BaseModel):
    mode: Literal["before", "after"] = Field(default="after")
    column: str = Field(...)
    comparison: Literal["ge", "le", "gt", "lt", "eq", "ne"] = Field(...)
    value_for_comparison: Union[str, float] = Field(...)

    @model_validator(mode="after")
    def mode_value_type_checking(self: Self) -> Self:
        mode = self.mode
        value = self.value_for_comparison
        string_modes = ["eq", "ne"]
        float_modes = ["ge", "le", "gt", "lt"]
        if mode in string_modes and not isinstance(value, str):
            self.value_for_comparison = str(value)
        if mode in float_modes and not isinstance(value, float):
            if isinstance(value, int):
                self.value_for_comparison = float(value)
            else:
                raise ValueError(
                    f"CODE:103 | You must specify a numeric type for {mode}"
                )
        return self


class RegularExpression(BaseModel):
    pattern: str = Field(...)
    replacement: str = Field(...)

    @field_validator("replacement", "pattern", mode="before")
    @classmethod
    def cast_string(cls, x: Any) -> str:
        return str(x)


def biolink_fallback(x: str) -> str:
    if "biolink:" not in x:
        return "biolink:" + x
    return x


class MappingHyperparameters(BaseModel):
    in_this_organism: Optional[str] = Field(default=None)
    classes_to_prioritize: Optional[list[str]] = Field(default=None)
    classes_to_avoid: Optional[list[str]] = Field(default=None)
    prefix: Optional[str] = Field(default=None)
    suffix: Optional[str] = Field(default=None)
    how_to_fill_column: Optional[
        Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
    ] = Field(default=None)
    substrings_to_remove: Optional[list[str]] = Field(default=None)
    regular_expressions: Optional[list[RegularExpression]] = Field(default=None)
    explode_by_delimiter: Optional[str] = Field(default=None)

    @field_validator("classes_to_prioritize", "classes_to_avoid", mode="after")
    @classmethod
    def biolink_priorities_and_avoid(cls, biolink_list: list[str]) -> list[str]:
        if biolink_list:
            return [biolink_fallback(thing) for thing in biolink_list]
        return biolink_list

    @field_validator("in_this_organism", mode="after")
    @classmethod
    def ncbi_taxon_fallback(cls, ncbi_taxon: str) -> str:
        if ncbi_taxon and "NCBITaxon:" not in ncbi_taxon:
            return "NCBITaxon:" + ncbi_taxon
        return ncbi_taxon


class GraphVertex(BaseModel):
    encoding_method: Literal["value", "column_of_values"] = Field(default="value")
    value_for_encoding: str = Field(...)
    mapping_hyperparameters: MappingHyperparameters = Field(
        default_factory=MappingHyperparameters
    )

    @model_validator(mode="after")
    def column_name_fix(self: Self) -> Self:
        encoding = self.encoding_method
        value = self.value_for_encoding
        if encoding in {"column_of_values"}:
            self.value_for_encoding = column_name_fallback(value)
        return self


class Triple(BaseModel):
    triple_subject: GraphVertex = Field(...)
    triple_object: GraphVertex = Field(...)
    triple_predicate: str = Field(default="biolink:associated_with")

    @field_validator("triple_predicate", mode="after")
    @classmethod
    def biolink_predicate(cls, predicate: str) -> str:
        return biolink_fallback(predicate)


def column_name_fallback(column_name: str) -> str:
    return column_name.upper()


class MathModuleTransformation(BaseModel):
    attribute: str = Field(...)
    arguments: list[Optional[float]] = Field(...)

    @field_validator("attribute", mode="after")
    @classmethod
    def is_math_module_attribute(cls, attribute: str) -> str:
        if not hasattr(math, attribute):
            raise ValueError(
                "CODE:101 | Transformation must include a valid math module atribute"
            )
        return attribute

    @field_validator("arguments", mode="after")
    @classmethod
    def arguments_contains_nonetype(
        cls, arguments: list[Optional[float]]
    ) -> list[Optional[float]]:
        if not any(argument is None for argument in arguments):
            raise ValueError("CODE:100 | At least one argument must be nonetype")
        return arguments

    @field_validator("arguments", mode="before")
    @classmethod
    def arguments_fallback(
        cls, arguments: list[Optional[float]]
    ) -> list[Optional[float]]:
        return [
            float(argument) if isinstance(argument, int) else argument
            for argument in arguments
        ]


class Attribute(BaseModel):
    encoding_method: Literal["value", "column_of_values"] = Field(default="value")
    value_for_encoding: Optional[Union[float, str, int]] = Field(
        default="Not applicable"
    )
    math_module_transformations: Optional[list[MathModuleTransformation]] = Field(
        default=None
    )

    @field_validator("value_for_encoding", mode="after")
    @classmethod
    def cast_string_with_nulls(cls, x: Any) -> Any:
        if x and not isinstance(x, str):
            return str(x)
        return x

    @model_validator(mode="after")
    def column_name_fix(self: Self) -> Self:
        encoding = self.encoding_method
        value = self.value_for_encoding
        if encoding == "column_of_values":
            self.value_for_encoding = column_name_fallback(str(value))
        return self


class Attributes(BaseModel):
    sample_size: Attribute = Field(default_factory=Attribute)
    p_value: Attribute = Field(default_factory=Attribute)
    multiple_testing_correction_method: Attribute = Field(default_factory=Attribute)
    assertion_strength: Attribute = Field(default_factory=Attribute)
    assertion_method: Attribute = Field(default_factory=Attribute)
    notes: Optional[str] = Field(default=None)


class Provenance(BaseModel):
    article_curie: str = Field(...)
    config_curator_name: str = Field(...)
    config_curator_organization: str = Field(...)

    @field_validator("article_curie", mode="after")
    @classmethod
    def is_article_curie(cls, article_curie: str) -> str:
        accepted_domains = {"PMC:", "PMID:", "doi:"}
        if all(domain not in article_curie for domain in accepted_domains):
            if "PMC" in article_curie:
                return "PMC:" + article_curie
            elif "/" in article_curie:
                return "doi:" + article_curie
            else:
                return "PMID:" + article_curie
        return article_curie


def start_end_rows_fallback(
    start: Optional[int], end: Optional[int], rows: Optional[list[int]]
) -> tuple[Optional[int], Optional[int], Optional[list[int]]]:
    fallback = 1
    if (start or end) and rows:
        raise ValueError(
            "CODE:102 | You cannot define start/end_at_line_number AND use_row_numbers"
        )
    if (start or end) and not (start and end):
        if not start:
            return fallback, end, rows
        if not end:
            return start, fallback, rows
    return start, end, rows


class ExcelHyperparameters(BaseModel):
    extension: Literal["xlsx", "xls"] = Field(...)
    which_excel_sheet_to_use: str = Field(default="Sheet1")
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[list[int]] = Field(default=None)

    @model_validator(mode="after")
    def start_end_rows_fix(self: Self) -> Self:
        start, end, rows = start_end_rows_fallback(
            self.start_at_line_number, self.end_at_line_number, self.use_row_numbers
        )
        self.start_at_line_number = start
        self.end_at_line_number = end
        self.use_row_numbers = rows
        return self


class CsvHyperparameters(BaseModel):
    extension: Literal["csv", "tsv", "txt"] = Field(...)
    file_delimiter: str = Field(default=",")
    start_at_line_number: Optional[int] = Field(default=None)
    end_at_line_number: Optional[int] = Field(default=None)
    use_row_numbers: Optional[list[int]] = Field(default=None)

    @model_validator(mode="after")
    def start_end_rows_fix(self: Self) -> Self:
        start, end, rows = start_end_rows_fallback(
            self.start_at_line_number, self.end_at_line_number, self.use_row_numbers
        )
        self.start_at_line_number = start
        self.end_at_line_number = end
        self.use_row_numbers = rows
        return self


class PdfHyperparameters(BaseModel):
    extension: Literal["pdf"] = Field(...)
    pages_table_is_on: Optional[list[int]] = Field(default=None)
    camelot_flavor: Literal["stream", "lattice"] = Field(default="lattice")


DownloadHyperparameters: TypeAlias = Annotated[
    Union[ExcelHyperparameters, CsvHyperparameters, PdfHyperparameters],
    Field(discriminator="extension"),
]


class Location(BaseModel):
    where_to_download_data_from: HttpUrl = Field(...)
    download_hyperparameters: DownloadHyperparameters = Field(...)


DATALAKE_INTERNAL: Path = Path("TABLASSERT/DATALAKE")
DATALAKE_INTERNAL.mkdir(parents=True, exist_ok=True)


# made download fallback because it takes longer to get the filepath like this
async def downloadfallback(link: str, storagepath: Path) -> Path:

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        async with page.expect_download(
            timeout=60_000  # or 1 minute
        ) as download_information:  # quadrupled timeout because it wouldn't work sometimes
            try:
                await page.goto(link, wait_until="load")
            except Exception as e:
                if "net::ERR_ABORTED" not in str(e):
                    raise RuntimeError(
                        f"CODE:104 | Unanticipated playright error: {str(e)}"
                    )

        config = await download_information.value
        filepath: Path = storagepath / config.suggested_filename
        posix_filepath: str = filepath.as_posix()

        if not filepath.exists():
            await config.save_as(posix_filepath)
            await context.close()
            await browser.close()

        return filepath


def download(link: str, storagepath: Path) -> Path:
    storagepath.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(link)
    name: str = Path(parsed.path).name or "not_applicable.ext"
    filepath: Path = storagepath / name

    if not filepath.exists():

        try:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            headers = {"User-Agent": user_agent}
            resp = requests.get(
                link, headers=headers, stream=True, timeout=30
            )  # or 30 seconds
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"CODE:105 | Error downloading file with requests: {str(e)}"
            )

        try:
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        except OSError as e:
            raise RuntimeError(f"CODE:106 | Error saving downloaded file: {str(e)}")

    return filepath


class Section(BaseModel):
    location: Location = Field(...)
    provenance: Provenance = Field(...)
    attributes: Attributes = Field(...)
    triple: Triple = Field(...)
    reindexing: Optional[list[Reindexing]] = Field(default=None)

    # FilePath inherits from pathlib apparently...
    posix_filepath: Optional[FilePath] = Field(default=None)

    @model_validator(mode="after")
    def file_downloader_and_path_generator(self: Self) -> Self:

        if not self.posix_filepath:
            # THIS ALSO DOWNLOADS THE FILE
            article_curie: str = self.provenance.article_curie

            storagepath: Path = DATALAKE_INTERNAL / article_curie
            storagepath.mkdir(parents=True, exist_ok=True)

            link: str = str(self.location.where_to_download_data_from)
            try:
                self.posix_filepath = download(link, storagepath)
            except Exception:
                self.posix_filepath = asyncio.run(downloadfallback(link, storagepath))
        return self


class TableConfig(BaseModel):
    sections: list[Section] = Field(...)
    template: Optional[Any] = Field(default=None)
