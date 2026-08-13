"""Tablassert configuration enums (non-Biolink).

These enums describe Tablassert's own configuration vocabulary (source kinds,
comparison operators, encoding/fill methods, contribution labels, and so on).

The Biolink Model vocabulary -- ``Categories``, ``Predicates``, ``Qualifiers``,
``KnowledgeLevels``, ``AgentTypes``, ``EdgeCategories``, and
``ALLOWED_EDGE_FIELDS`` -- is derived from the ``biolink-model`` package and lives
in :mod:`tablassert.biolink`. Import those from there.
"""

from __future__ import annotations

from enum import Enum


class Tokens(str, Enum):
    AUTO = "auto"
    VALUES = "values"


class Repositories(str, Enum):
    PUBMED_CENTRAL = "PMC"
    PUBMED = "PMID"


class InformationResources(str, Enum):
    PUBMED = "infores:pubmed"
    PUBMED_CENTRAL = "infores:pubmed-central"


class Contributions(str, Enum):
    CURATION = "curation"
    VALIDATION = "validation"
    TOOL = "tool"


class Comparisons(str, Enum):
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    EQ = "eq"
    NE = "ne"


class Functions(str, Enum):
    COPYSIGN = "copysign"
    POW = "pow"


class Files(str, Enum):
    TEXT = "text"
    EXCEL = "excel"


class EncodingMethods(str, Enum):
    VALUE = "value"
    COLUMN = "column"


class FillMethods(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    ZERO = "zero"
    ONE = "one"
