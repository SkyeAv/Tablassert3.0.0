from tablassert.enums import EncodingMethods
from tablassert.enums import Contributions
from tablassert.enums import Repositories
from tablassert.enums import Comparisons
from tablassert.enums import FillMethods
from tablassert.enums import Predicates
from tablassert.enums import Qualifiers
from tablassert.enums import Categories
from tablassert.enums import Functions
from tablassert.enums import Statuses
from tablassert.enums import Syntaxes
from tablassert.enums import Tokens
from pydantic import NonNegativeInt
from tablassert.enums import Files
from pydantic import PositiveInt
from pydantic import ConfigDict
from pydantic import BaseModel
from pydantic import HttpUrl
from typing import Optional
from pydantic import Field
from typing import Literal
from typing import Union
from pathlib import Path

class TablaBase(BaseModel):
  model_config: ConfigDict = ConfigDict(
    str_strip_whitespace=True,
    validate_assignment=True,
    use_enum_values=True,
    extra="forbid",
    populate_by_name=True,
  )

class Reindex(TablaBase):
  column: str = Field(...)
  comparison: Comparisons = Field(Comparisons.NE)
  comparator: Union[str, int, float] = Field(...)

class BaseSource(TablaBase):
  local: Path = Field(...)
  url: HttpUrl = Field(...)
  rows: Optional[list[NonNegativeInt]] = Field(None)
  row_slice: Optional[list[Union[NonNegativeInt, Literal[Tokens.AUTO]]]] = Field(None)
  reindex: Optional[list[Reindex]] = Field(None)

class Excel(BaseSource):
  kind: Literal[Files.EXCEL] = Field(Files.EXCEL)
  sheet: Optional[str] = Field("Sheet1")

class Text(BaseSource):
  kind: Literal[Files.TEXT] = Field(Files.TEXT)
  delimiter: Optional[str] = Field(",")

class Regex(TablaBase):
  pattern: str = Field(...)
  replacement: str = Field(...)

class Math(TablaBase):
  function: Functions = Field(...)
  arguments: list[Union[Literal[Tokens.VALUES], float, int]] = Field(...)

class Encoding(TablaBase):
  method: EncodingMethods = Field(EncodingMethods.VALUE)
  encoding: Union[str, int, float] = Field(...)
  regex: Optional[list[Regex]] = Field(None)
  fill: Optional[FillMethods] = Field(None)
  remove: Optional[list[str]] = Field(None)
  prefix: Optional[str] = Field(None)
  suffix: Optional[str] = Field(None)
  explode_by: Optional[str] = Field(None)
  transformations: Optional[list[Math]] = Field(None)

class NodeEncoding(Encoding):
  taxon: Optional[PositiveInt] = Field(None)
  prioritize: Optional[list[Categories]] = Field(None)
  avoid: Optional[list[Categories]] = Field(None)

class Qualifier(NodeEncoding):
  qualifier: Qualifiers = Field(...)

class Statement(TablaBase):
  subject: NodeEncoding = Field(...)
  object: NodeEncoding = Field(...)
  predicate: Predicates = Field(Predicates.RELATED_TO)
  qualifiers: Optional[list[Qualifier]] = Field(None)

class Contributor(TablaBase):
  kind: Contributions = Field(Contributions.CURATION)
  name: str = Field(...)
  date: str = Field(...)
  organizations: Optional[list[str]] = Field(None)
  comment: Optional[str] = Field(None)

class Provenance(TablaBase):
  repo: Repositories = Field(Repositories.PUBMED_CENTRAL)
  publication: str = Field(...)
  contributors: list[Contributor] = Field(...)

class Annotation(Encoding):
  annotation: str = Field(...)

class Section(TablaBase):
  # ? Pydantic "Section" Model And Coercion
  syntax: Syntaxes = Field(Syntaxes.TC3)
  status: Statuses = Field(Statuses.ALPHA)
  source: Union[Excel, Text] = Field(...)
  statement: Statement = Field(...)
  provenance: Provenance = Field(...)
  annotations: Optional[list[Annotation]] = Field(None)

class Graph(TablaBase):
  # ? Pydantic "Graph" Configuration
  syntax: Syntaxes = Field(Syntaxes.GC2)
  name: str = Field(...)
  version: str = Field(...)
  tables: list[Path] = Field(...)
  dbssert: Path = Field(...)
  pubmed_db: Path = Field(...)
  pmc_db: Path = Field(...)
