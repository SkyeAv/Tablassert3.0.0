from pydantic import ValidationError, BaseModel, DirectoryPath
from playwright.async_api import async_playwright
from src.tablassert.models.table import Section
from ruamel.yaml.error import YAMLError
from typing import Any, Type, TypeVar
from deepmerge.merger import Merger
from urllib.parse import urlparse
from functools import lru_cache
from ruamel.yaml import YAML
from copy import deepcopy
from pathlib import Path
import requests


@lru_cache(maxsize=1)
def project_root(
    io_utility_path: Path = Path(__file__).resolve(),
    configuration_at_root: str = "pyproject.toml",
) -> Path:
    for parent in [io_utility_path] + list(io_utility_path.parents):
        if (parent / configuration_at_root).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"CODE:110 | Couldn't locate {configuration_at_root} in IO utility parents"
    )


# from ruamel.yaml
yaml = YAML()


def load_yaml(filename: Path) -> Any:
    try:
        with filename.open("r") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"CODE:111 | {filename.as_posix()} not found")
    except PermissionError:
        raise RuntimeError(f"CODE:112 | Permission denied: {filename.as_posix()}")
    except YAMLError as e:
        raise RuntimeError(
            f"CODE:113 | YAML parsing error in {filename.as_posix()} {str(e)}"
        )


PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


# checks for validations errors
def load_model(parsed_yaml: Any, model: Type[PydanticModel]) -> PydanticModel:
    try:
        return model.model_validate(parsed_yaml)
    except ValidationError as e:
        raise RuntimeError(f"CODE:114 | {model.__name__}: {str(e)}")


TABLE_CONFIG_EXTENSION: str = ".yaml"
# merger because default yaml merging doesn't work
merger: Merger = Merger(
    [(dict, ["merge"]), (list, ["append"]), (set, ["union"])],
    ["override"],
    ["override"],
)


def build_sections(
    graphmodel: dict[str, Any],
) -> list[tuple[Section, dict[str, Any], int]]:
    directories: list[DirectoryPath] = graphmodel["location"][
        "table_config_containing_directories"
    ]
    sections: list[tuple[Section, dict[str, Any], int]] = []
    for d in directories:
        for path in Path(str(d)).rglob("*"):
            if path.suffix.lower() == TABLE_CONFIG_EXTENSION:
                table_yaml: Any = load_yaml(path)
                template = table_yaml.get("template", {})
                subsections = table_yaml.get("sections", [])
                if subsections != []:
                    for idx, section in enumerate(subsections, start=1):
                        merged_section = merger.merge(
                            deepcopy(template),
                            (
                                deepcopy(section)
                                if section and isinstance(section, dict)
                                else {}
                            ),
                        )
                        SubSection: Section = load_model(merged_section, Section)
                        sections.append((SubSection, graphmodel, idx))
    return sections


def filepathgen(link: str, storagepath: Path) -> Path:
    parsed = urlparse(link)
    name: str = Path(parsed.path).name or "not_applicable.ext"
    return storagepath / name


def download(link: str, storagepath: Path) -> Path:
    storagepath.mkdir(parents=True, exist_ok=True)

    filepath: Path = filepathgen(link, storagepath)
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
