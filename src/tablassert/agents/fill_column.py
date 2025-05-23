__author__ = "Skye Lane Goetz"
__status__ = "Development"


from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field, conlist, constr, model_validator
from tablassert.agents.toolkit import (
    AgentInvocationError,
    get_system_prompt,
    ollama_client,
    UserInput,
    LLMAgent,
)
from typing import Literal


class FillColumnOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    fill_column: A valid polars fill_null strategy parameter to fill the null values in a given column

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)

    OPTIONS FOR fill_column:

    forward
    backward
    min
    max
    mean
    zero
    one
    """

    fill_column: Literal[
        "forward",
        "backward",
        "min",
        "max",
        "mean",
        "zero",
        "one",
    ] = Field(
        ...,
        description="A valid polars fill_null strategy parameter to fill the null values in a given column",
    )
    suggested_questions: (
        conlist(
            item_type=constr(strip_whitespace=True),
            min_length=0,
            max_length=3,
        )
        | None
    ) = Field(
        default=None,
        description="An optional list of suggested follow up questions to clarify any JSON output you're still unsure about",
    )

    @model_validator(mode="before")
    def print_self(self):
        print(self)
        return self


class FillColumnAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "FillColumnOutput"),
            input_schema=UserInput,
            output_schema=FillColumnOutput,
        )


def run_fill_column_agent(user_input: dict[str, object]) -> FillColumnOutput:
    agent = FillColumnAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
