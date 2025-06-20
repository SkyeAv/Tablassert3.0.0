from pydantic import ValidationError, BaseModel
from ruamel.yaml.error import YAMLError
from typing import Any, Type, TypeVar
from functools import lru_cache
from ruamel.yaml import YAML
from pathlib import Path

@lru_cache(maxsize=None)
def project_root(io_utility_path: Path = Path(__file__).resolve(), configuration_at_root: str = "pyproject.toml") -> Path:
    for parent in [io_utility_path] + list(io_utility_path.parents):
        if (parent / configuration_at_root).exists():
            return parent.resolve()
    raise FileNotFoundError("Couldn't locate " + configuration_at_root + " in IO utility parents")

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

def load_model(dictonary: Any, model: Type[PydanticModel]) -> PydanticModel:
    try:
        return model.model_validate(dictonary)
    except ValidationError as e:
        raise RuntimeError(model.__name__ + ": " + str(e))