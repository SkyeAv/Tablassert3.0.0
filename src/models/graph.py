from pydantic import field_validator, BaseModel, DirectoryPath, FilePath, Field
from typing import Optional, Any


class Metadata(BaseModel):
    knowledge_graph_name: str = Field(...)
    graph_version: str = Field(default="0.0.0")
    graph_description: Optional[str] = Field(default=None)


class SqliteDatabases(BaseModel):
    babel: FilePath = Field(...)
    kg2: FilePath = Field(...)
    pubmed: FilePath = Field(...)
    pmc: FilePath = Field(...)


class Location(BaseModel):
    table_config_containing_directories: list[DirectoryPath] = Field(...)
    sqlite_databases: SqliteDatabases = Field(...)


class Hyperparameters(BaseModel):
    number_of_parallel_processes_to_run: int = Field(default=1)
    maximum_p_value_in_graph: float = Field(default=1.0)

    @field_validator("maximum_p_value_in_graph", mode="before")
    @classmethod
    def int_to_float(cls, possible_int: Any) -> Any:
        if possible_int and isinstance(possible_int, int):
            return float(possible_int)
        return possible_int


class GraphConfig(BaseModel):
    metadata: Metadata = Field(...)
    location: Location = Field(...)
    hyperparameters: Optional[Hyperparameters] = Field(default=None)
