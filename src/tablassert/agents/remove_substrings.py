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


class RemoveSubstringsOutput(BaseIOSchema):
    """
    From the following message, extract a list of *all* whitespace-separated substrings into the `remove` field. 
    Keep them exactly as they appear — including symbols like `"`, `_`, etc. 
    Do **not** deduplicate, clean, or ignore punctuation.

    Also return an optional `suggested_questions` field, or `null` if there are no clarifying questions.
    """

    remove: conlist(item_type=constr(min_length=1), min_length=1) | None = Field(
        default=None,
        description='A list of substrings to remove by applying re.sub(substring, "", string) to',
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


class RemoveSubstringsAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "RemoveSubstringsOutput"),
            input_schema=UserInput,
            output_schema=RemoveSubstringsOutput,
        )


def run_remove_substrings_agent():
    agent = RemoveSubstringsAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_remove_substrings_agent()
