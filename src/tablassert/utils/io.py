from pydantic import ValidationError, BaseModel, DirectoryPath
from src.tablassert.models.table import Section
from ruamel.yaml.error import YAMLError
from typing import Any, Type, TypeVar
from deepmerge.merger import Merger
from functools import lru_cache
from ruamel.yaml import YAML
from pathlib import Path


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
    [(dict, ["merge"]), (list, ["append"])], ["override"], ["override"]
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
                        merged_section = merger.merge(template.copy(), section if section else {})
                        SubSection: Section = load_model(merged_section, Section)
                        sections.append((SubSection, graphmodel, idx))
    return sections
