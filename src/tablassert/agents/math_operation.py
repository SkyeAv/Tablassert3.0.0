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
from tablassert.cfg import MathParams


class MathOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    math: A field denoting specific matheatical operations to preform on the data stores in an attribute
        attr: A string of a valid python math module atrribute used to process data (DONT INCLUDE "math." ONLY INCLUDE THE ATTR LIKE "sqrt")
        args: A list[Union[str, None]] of arguments for to pass to the aforementioned math module attribute, use null in place of the data you wish to tranform 

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT CAVEATS:
    
    "args" MUST contain at least one None value
    "args" MUST have a length equal to the # of arguments the math module attribute requires
    ABSOLUTELY DO NOT CHANGE THE ATTRIBUTE OR DEFAULT TO "sqrt"
    ABSOLUTELY DO NOT return a list of type list for "args"
    NEVER RETURN "\"None\"" and return "None" INSTEAD

    SUPER IMPORTANT CAVEATS:
    
    For "args" if there is only one agrument return [None] otherwise return something like [2, None] for "log 2 the values"
    """

    math: MathParams = Field(
        ...,
        description="A field denoting specific matheatical operations to preform on the data stores in an attribute",
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


class MathAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "MathOutput"),
            input_schema=UserInput,
            output_schema=MathOutput,
        )


def run_math_agent():
    agent = MathAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_math_agent()
