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


class IsMathOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    has_math: A boolean denoting if the "math" attribute (An optional field denoting specific matheatical operations to preform on the data stores in an attribute) is implied in the message
    math_portions: An optional list of text pertaining to each individual mathematical operation specified in a message (e.g., "First I want to square root the p-value then negative log10 the p_value" -> ["First I want to square root the p-value", "then negative log10 the p_value"])

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about
    """

    has_math: bool = Field(
        ...,
        description='A boolean denoting if the "math" attribute (An optional field denoting specific matheatical operations to preform on the data stores in an attribute) is implied in the message',
    )
    math_portions: (
        conlist(
            item_type=constr(min_length=1, strip_whitespace=True),
            min_length=1,
        )
        | None
    ) = Field(
        default=None,
        description='An optional list of text pertaining to each individual mathematical operation specified in a message (e.g., "First I want to square root the p-value then negative log10 the p_value" -> ["First I want to square root the p-value", "then negative log10 the p_value"])',
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


class IsMathAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("boolean_split", "IsMathOutput"),
            input_schema=UserInput,
            output_schema=IsMathOutput,
        )


def run_is_math_agent(user_input: dict[str, object]) -> IsMathOutput:
    agent = IsMathAgent()
    try:
        return(agent.invoke(user_input))
    except AgentInvocationError as e:
        raise e
