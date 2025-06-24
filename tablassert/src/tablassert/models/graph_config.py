from pydantic import field_validator, BaseModel, FilePath, Field
from typing import Optional, Union


class Metadata(BaseModel):
    knowledge_graph_name: str = Field(...)
    graph_version: str = Field(default="0.0.0")
    graph_description: Optional[str] = Field(default=None)


class SqliteDatabases(BaseModel):
    babel: FilePath = Field(...)
    kg2: FilePath = Field(...)
    mapping_patch: FilePath = Field(...)
    pubmed: FilePath = Field(...)


class Location(BaseModel):
    table_config_containing_directories: set[FilePath] = Field(...)
    sqlite_databases: SqliteDatabases = Field(...)


class Hyperparameters(BaseModel):
    number_of_parallel_processes_to_run: int = Field(default=1)
    sql_progess_handler_time: float = Field(default=1.5)
    maximum_p_value_in_graph: float = Field(default=1.0)
    byte_size_of_mapping_cache: int = Field(default=1e9)

    @field_validator(
        "maximum_p_value_in_graph", "sql_progess_handler_time", mode="before"
    )
    @classmethod
    def convert_int_to_float(
        cls, possible_int: Optional[Union[int, float]]
    ) -> Optional[Union[int, float]]:
        if possible_int and isinstance(possible_int, int):
            return float(possible_int)
        return possible_int


class GraphConfig(BaseModel):
    metadata: Metadata = Field(...)
    location: Location = Field(...)
    hyperparameters: Optional[Hyperparameters] = Field(default=None)
