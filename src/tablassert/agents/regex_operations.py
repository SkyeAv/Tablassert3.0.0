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
from tablassert.cfg import Regex


class RegexOperationsOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    regex: A list of regex operations to apply re.sub(pattern, replacement, string) to
        pattern: The regex pattern as denoted bu the pattern parameter in python's re.sub
        replacement: The regex replacement as denoted by the replacement parameter in python's re.sub

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)
    """

    regex: conlist(item_type=Regex, min_length=1) = Field(
        ...,
        description="A list of regex operations to apply re.sub(pattern, replacement, string) to",
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


class RegexOperationsAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "RegexOperationsOutput"),
            input_schema=UserInput,
            output_schema=RegexOperationsOutput,
        )


def run_regex_operations_agent(user_input: dict[str, object]) -> RegexOperationsOutput:
    agent = RegexOperationsAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
