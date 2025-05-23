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


class InOrganismOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    in_organism: A NCBITaxon: curie denoting which organism the scientific finding relates to/was discovered in. NCBITaxon:9606 denotes humans for example

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)

    IMPORTANT CLARIFICATIONS:

    ONLY ACCEPT A NCBITaxon: curie FOR in_organism
    """

    in_organism: constr(
        min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9./-]+$", strip_whitespace=True
    ) = Field(
        ...,
        description="A NCBITaxon: curie denoting which organism the scientific finding relates to/was discovered in. NCBITaxon:9606 denotes humans for example",
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


class InOrganismAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "InOrganismOutput"),
            input_schema=UserInput,
            output_schema=InOrganismOutput,
        )


def run_in_organism_agent(user_input: dict[str, object]) -> InOrganismOutput:
    agent = InOrganismAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
