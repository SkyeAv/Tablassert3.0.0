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


class SuffixOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    suffix: A suffix to add to the end of the values denoted by a node before they're processed/passed to databases/etc.. (this can be a string of literally anything and usually is the same as chat_msg)

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)

    IMPORTANT CLARIFICATIONS:

    INCLUDE ALL SPECIAL CHARACTERS ESPECIALLY ":" IN SUFFIX IF THEYRE IN A SUFFIX
    """

    suffix: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description="A suffix to add to the end of the values denoted by a node before they're processed/passed to databases/etc..",
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


class SuffixAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "SuffixOutput"),
            input_schema=UserInput,
            output_schema=SuffixOutput,
        )


def run_suffix_agent():
    agent = SuffixAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_suffix_agent()
