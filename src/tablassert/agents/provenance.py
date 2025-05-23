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
from tablassert.cfg import Provenance


class ProvenanceOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    provenance: A field denoting the provenance of edges in a Tablassert knowledge graph (i.e., where the knowledge came from, whos curating, from what organization, etc..)
        publication_id: A CURIE (STRICTLY PMC:, PMID:, or doi:) specifiying the scientific paper the data you're processing comes from
        - THIS IS NEVER A URL AND NEVER INCLUDES "https://"
        - publication_id MUST INCLUDE "PMC:" or "PMID:" or "doi:" IN YOUR OUTPUT, DONT MAKE UP CURIES IF THEY DONT EXIST
        curator: The person or entity curating this configuration, if you're a LLM also credit yourself here (USUALLY STARTS WITH "I am" or "My name is")
        org: The organization the curator or entity curating represents, if you're an LLM also credit your organization here (USUALLY STARTS WITH "I work at" or "My organization is")

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about
    """

    provenance: Provenance = Field(
        ...,
        description="A field denoting the provenance of edges in a Tablassert knowledge graph (i.e., where the knowledge came from, whos curating, from what organization, etc..)",
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


class ProvenanceAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "ProvenanceOutput"),
            input_schema=UserInput,
            output_schema=ProvenanceOutput,
        )


def run_provenance_agent(user_input: dict[str, object]) -> ProvenanceOutput:
    agent = ProvenanceAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
