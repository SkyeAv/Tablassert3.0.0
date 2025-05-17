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


class NodeParamsOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    mode: A strategy for defining nodes. "value" defines a literal value. "cvalue" defines a column containing values. "scvalue" defines a shared column containing values and something else. "curie" defines a literal CURIE. "ccurie" defines a column containing CURIES. "sccurie" defines a shared column containing CURIES and something else CURIES are compact universal resource identifiers and are parts of ontologies.
    value: The value contexualizing the aforementioned Node mode (e.g., the CURIE specified or the name of the shared column of values)

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about (do NOT use empty strings here)

    SUPER IMPORTANT CLARIFICATIONS:

    if the word "shared" is specified then "mode" MUST start with "sc" otherwise it CANNOT start with "sc"
    some form of "curie" MUST be specified for "mode" to end with "curie" OR THERE MUST BE AN ACUTAL CURIE (e.g., NCBIGene:9303, KEGG.PATHWAY:10380328)
    when specifying a "[...]c[...]" for column in "mode" it MUST strictly be an uppercase Excel-style label (e.g., "A", "B", ..., "Z", "AA", "ZZ"); DO NOT INCLUDE "column" before the label
    """

    mode: Literal["value", "cvalue", "scvalue", "curie", "ccurie", "sccurie"] = Field(
        ...,
        description='A strategy for defining nodes. "value" defines a literal value. "cvalue" defines a column containing values. "curie" defines a literal CURIE. "ccurie" defines a column containing CURIES. CURIES are compact universal resource identifiers and are parts of ontologies.',
    )
    value: constr(min_length=1) = Field(
        ..., description="The value contexualizing the aforementioned Node mode"
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


class NodeParamsAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "NodeParamsOutput"),
            input_schema=UserInput,
            output_schema=NodeParamsOutput,
        )


def run_node_params_agent():
    agent = NodeParamsAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_node_params_agent()
