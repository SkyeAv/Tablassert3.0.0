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


class SplitExplodeOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    split_explode: A field specificying which delimiter to split the string encoded in a node by into a list before exploding values in that list thier own respective nodes, like with a pandas or polars explode column (this can be a string of literally anything and usually is the same as chat_msg)

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)
    """

    split_explode: constr(min_length=1) = Field(
        ...,
        description="A field specificying which delimiter to split the string encoded in a node by into a list before exploding values in that list thier own respective nodes, like with a pandas or polars explode column",
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


class SplitExplodeAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "SplitExplodeOutput"),
            input_schema=UserInput,
            output_schema=SplitExplodeOutput,
        )


def run_split_explode_agent(user_input: dict[str, object]) -> SplitExplodeOutput:
    agent = SplitExplodeAgent()
    try:
        return(agent.invoke(user_input))
    except AgentInvocationError as e:
        raise e
