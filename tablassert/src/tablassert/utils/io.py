from tablassert.src.tablassert.models.table_config import TableConfig, Section
from playwright.async_api import async_playwright
from pydantic import ValidationError, BaseModel
from typing import Any, Type, TypeVar, Optional
from ruamel.yaml.error import YAMLError
from pydantic import HttpUrl, FilePath
from functools import lru_cache
from ruamel.yaml import YAML
from pathlib import Path
import asyncio


@lru_cache(maxsize=None)
def project_root(
    io_utility_path: Path = Path(__file__).resolve(),
    configuration_at_root: str = "pyproject.toml",
) -> Path:
    for parent in [io_utility_path] + list(io_utility_path.parents):
        if (parent / configuration_at_root).exists():
            return parent.resolve()
    raise FileNotFoundError(
        "Couldn't locate " + configuration_at_root + " in IO utility parents"
    )


yaml = YAML()


def load_yaml(filename: Path) -> Any:
    try:
        with filename.open("r") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise RuntimeError(filename.as_posix() + " not found")
    except PermissionError:
        raise RuntimeError("Permission denied: " + filename.as_posix())
    except YAMLError:
        raise RuntimeError("YAML parsing error in " + filename.as_posix())


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


def load_model(parsed_yaml: Any, model: Type[PydanticModel]) -> PydanticModel:
    try:
        return model.model_validate(parsed_yaml)
    except ValidationError as e:
        raise RuntimeError(model.__name__ + ": " + str(e))


async def download_from_link(url: HttpUrl, filepath: Path) -> None:

    if filepath.exists():
        return  # exits if the file is already downloaded

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        async with page.expect_download() as download_information:
            await page.goto(str(url))

        download = await download_information.value
        await download.save_as(filepath.as_posix())
        await browser.close()

TABLE_CONFIG_EXTENSION: str = ".yaml"

def get_sections(dirs: set[FilePath]) -> set[Optional[Section]]:
    sections: set[Optional[Section]] = set()
    for d in dirs:
        for path in Path(str(d)).rglob("*"):
            if path.suffix.lower() == TABLE_CONFIG_EXTENSION:
                table_yaml: Any = load_yaml(path)
                Table: TableConfig = load_model(table_yaml, TableConfig)
                for section in Table.sections:
                    Sections.add(section)
    return sections
