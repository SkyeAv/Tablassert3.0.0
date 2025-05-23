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


class AvoidOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    avoid: A list of biolink:<Class> curies (e.g., biolink:Gene or biolink:OrganismTaxon) that Tablasserts databases should Avoid when mapping strings to CURIES

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)

    IMPORTANT CLARIFICATIONS:

    ONLY ACCEPT A biolink:<Class> curie FOR avoid
    """

    avoid: conlist(
        item_type=constr(
            min_length=10,
            pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$",
            strip_whitespace=True,
        ),
        min_length=1,
    ) = Field(
        ...,
        description="A list of biolink:<Class> curies that Tablasserts databases should avoid when mapping strings to CURIES",
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


class AvoidAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "AvoidOutput"),
            input_schema=UserInput,
            output_schema=AvoidOutput,
        )


def run_avoid_agent(user_input: dict[str, object]) -> AvoidOutput:
    agent = AvoidAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
