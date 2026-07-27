from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Union

import yaml
from yaml import CLoader


def fastmerge(a: Union[list[Any], dict[str, Any]], b: Union[list[Any], dict[str, Any]]) -> Any:
    """Recursively merge ``b`` into ``a`` in place and return the result.

    Mirrors the legacy deepmerge config but with the slow generic walk stripped
    out: dict-on-dict merges recurse, list-on-list extends, and any other
    scalar-vs-anything collision falls back to ``b`` overwriting ``a``.

    Args:
        a: Left operand (mutated in place when it is a container).
        b: Right operand; wins on scalar collisions.

    Returns:
        The merged value (same object as ``a`` when types aligned, else ``b``).
    """
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
    """Read a YAML config file into a Python object.

    Args:
        p: Path to the YAML file.

    Returns:
        Parsed YAML content (typically a ``dict``).
    """
    with p.open("r") as f:
        return yaml.load(f, Loader=CLoader)


def to_yaml(p: Path, data: object) -> None:
    """Write dict-like data to YAML preserving declared key order.

    Args:
        p: Destination path.
        data: Object to serialize.
    """
    with p.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def to_sections(instructions: dict[str, Any], table: Path) -> list[list[dict[str, Any]]]:
    """Expand a table config dict into a list of section dicts.

    Each section is the deep merge of the table-level ``template`` over a
    per-section entry. The originating ``table`` path is stamped into each
    section as ``config``.

    Args:
        instructions: Parsed YAML config with optional ``template`` and
            ``sections`` keys.
        table: Path of the source table file (recorded on each section).

    Returns:
        One merged dict per section.
    """
    template: dict[str, Any] = instructions.get("template", {})
    template["config"] = table
    sections: list[dict[str, Any]] = instructions.get("sections", [{}])
    return [fastmerge(deepcopy(template), x) for x in sections]
