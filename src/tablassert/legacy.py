"""Load legacy YAML configs without silently losing data to duplicate mapping keys.

WHY: PyYAML's default mapping construction keeps only the LAST occurrence of a duplicate
key, silently dropping everything earlier occurrences contributed. Legacy human-curated
table configs accumulated such duplicates over years, and a plain ``yaml.safe_load`` would
ingest them with data lost and no signal. This loader DETECTS every duplicate, MERGES the
occurrences (dict+dict deep-merged, list+list extended, later value wins otherwise), and
fires :class:`~tablassert.errors.LegacyDuplicateKeyWarning` per duplicate so curators can
fix the source file.

Safety stance mirrors :func:`tablassert.ingests.from_yaml` — configs are untrusted, so
parsing runs on a ``yaml.SafeLoader`` subclass: only the safe tags are resolved and no
arbitrary Python objects are constructed. The pure-Python loader is used because the
duplicate hook replaces the mapping constructor; legacy files are small, so the C loader's
speed buys nothing here.
"""

from __future__ import annotations

import collections.abc
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from yaml import SafeLoader
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

from tablassert.errors import LegacyDuplicateKeyWarning
from tablassert.ingests import fastmerge


class _LegacyDuplicateMergingSafeLoader(SafeLoader):
    """SafeLoader that merges duplicate mapping keys instead of dropping them.

    See the module docstring for WHY. Merge semantics are delegated to
    :func:`tablassert.ingests.fastmerge` so this loader and the production
    section-merge path can never drift apart.
    """

    def construct_mapping(self, node: Node, deep: bool = False) -> dict[Any, Any]:
        """Build one mapping, warning on and merging every duplicated key.

        Args:
            node: The mapping node being constructed.
            deep: Accepted for ``SafeConstructor`` signature compatibility; values are
                always constructed eagerly (see below), which subsumes it.

        Returns:
            The mapping with duplicate keys merged instead of overwritten.

        Note:
            Values are constructed with ``deep=True``: PyYAML's lazy two-phase
            construction returns EMPTY placeholders for nested containers whose fills are
            deferred, so merging them (or deep-copying them) would freeze the emptiness and
            lose every item once the deferred fills land on the orphaned originals.
        """
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, f"expected a mapping node, but found {node.tag}", node.start_mark)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key: Any = self.construct_object(key_node, deep=True)
            if not isinstance(key, collections.abc.Hashable):
                raise ConstructorError("while constructing a mapping", node.start_mark, "found unhashable key", key_node.start_mark)
            value: Any = self.construct_object(value_node, deep=True)
            if key in mapping:
                warnings.warn(
                    f"Duplicate key `{key}` at {key_node.start_mark.name}:{key_node.start_mark.line + 1} — merged with the "
                    "earlier value instead of dropping it (PyYAML silently keeps only the last occurrence of a duplicate key).",
                    LegacyDuplicateKeyWarning,
                    stacklevel=2,
                )
                # deepcopy: the earlier value may be shared with another node through a YAML alias;
                # fastmerge mutates in place, so merging into the shared object would silently
                # rewrite the aliased original too.
                mapping[key] = fastmerge(deepcopy(mapping[key]), value)
            else:
                mapping[key] = value
        return mapping


def load_legacy_yaml(p: Path) -> object:
    """Read a legacy YAML config, merging duplicate mapping keys instead of dropping them.

    Duplicate-key occurrences fire :class:`~tablassert.errors.LegacyDuplicateKeyWarning`
    naming the key and its line; the merged result is returned regardless.

    Args:
        p: Path to the legacy YAML file.

    Returns:
        Parsed YAML content (typically a ``dict``).
    """
    with p.open("r") as f:
        # SafeLoader (never Loader/CLoader): legacy configs are untrusted; SafeLoader resolves
        # only the safe tags and refuses arbitrary-object construction (same stance as
        # ingests.from_yaml's CSafeLoader).
        return yaml.load(f, Loader=_LegacyDuplicateMergingSafeLoader)
