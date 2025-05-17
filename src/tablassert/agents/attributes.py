__author__ = "Skye Lane Goetz"
__status__ = "Development"


from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field, conlist, constr, model_validator
from tablassert.agents.toolkit import (
    AgentInvocationError,
    get_system_prompt,
    get_user_input,
    ollama_client,
    UserInput,
    LLMAgent,
)
from typing import Literal


class AttributesFromOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    mode: A literal field telling Tablassert how to define attributes. "column" explicitly defines a dataframe column containing values for said attribute while "predefined" defines a predefined value for said attribute
    value: A field containg a value that adds context the aforementioned mode

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT CAVEATS:

    ignore the name of the attribute... it DOES NOT BELONG HERE
    if a "column" isn't EXPLICITLY defined default to "mode" == "predefined"
    if "mode" == "column" then "value" MUST follow the A-ZZ naming convention like with microsoft excel columns, if it doesn't, DONT USE "mode" == "column" as it must be a "predefined"
    if a "predefined" "value" contains numeric characters ONLY INCLUDE THE NUMERIC CHARACTERS in your output otherwise output a string
    """

    mode: Literal["column", "predefined"] = Field(
        ...,
        description='A field telling Tablassert how to define attributes. "column" explicitly defines a column containing values for said attribute while "predefined" defines a predefined value for said attribute',
    )
    value: constr(min_length=1) = Field(
        ...,
        description="A field containg a value that adds context the aforementioned mode",
    )
    suggested_questions: (
        conlist(
            item_type=constr(min_length=1, strip_whitespace=True),
            min_length=1,
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


class AttributesAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "AttributesFromOutput"),
            input_schema=UserInput,
            output_schema=AttributesFromOutput,
        )


def run_attributes_agent():
    agent = AttributesAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_attributes_agent()
