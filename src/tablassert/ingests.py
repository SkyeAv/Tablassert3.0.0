from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import lazy_loader as Lazy
from yaml import CLoader

if TYPE_CHECKING:
    import yaml
else:
    yaml = Lazy.load("yaml")


def fastmerge(a: Union[list[Any], dict[str, Any]], b: Union[list[Any], dict[str, Any]]) -> Any:
    # ? Streamlined (Fast) Implementation Of Deepmerge Config
    if isinstance(a, dict) and isinstance(b, dict):
        for k, v in b.items():
            if k in a:
                av: Any = a[k]
                if isinstance(av, dict) and isinstance(v, dict):
                    fastmerge(av, v)
                elif isinstance(av, list) and isinstance(v, list):
                    av.extend(v)
                else:
                    a[k] = v
            else:
                a[k] = v
        return a

    elif isinstance(a, list) and isinstance(b, list):
        a.extend(b)
        return a
    else:
        return b


def from_yaml(p: Path) -> object:
    # ? Reads YAML Config To Dict
    with p.open("r") as f:
        return yaml.load(f, Loader=CLoader)


def to_sections(instructions: dict[str, Any], table: Path) -> list[list[dict[str, Any]]]:
    # ? Converts Dict To Sections
    template: dict[str, Any] = instructions.get("template", {})
    template["config"] = table
    sections: list[dict[str, Any]] = instructions.get("sections", [{}])
    return [fastmerge(deepcopy(template), x) for x in sections]
