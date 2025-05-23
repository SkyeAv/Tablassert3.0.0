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
from tablassert.cfg import ReindexingOperation


class ReindexingOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    reindexing: A field containing any useful reindexing operations for Tablassert to Preform
        when: A Literal["before", "after"] designation stating if you want to reindex before or after the data is processed by Tablassert
        mode: A Literal["ge", "le", "gt", "lt", "eq", "ne"] specifying type of comparison operation to reindex the data in the knowledge based by (eq means equal to and ne means not equal to)
        column: The column that contains the values you want to reindex (this should be explicitly states like "p-value" or "AA")
        value: The value that you ultimatly want to use with the comparison operation to reindex the data by (like 2 when reindexing by "greater than 2")

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT CAVEATS:

    only mode "eq" and "ne" support strings as values for reindexing operations

    CRITICAL RULE FOR when:

    If the column name is strictly an uppercase Excel-style label (e.g., "A", "B", ..., "Z", "AA", "ZZ"), then when must be "before".
    If the column name is any other format (e.g., words like "p-value", "height", or "score"), then when must be "after".
    """

    reindexing: ReindexingOperation = Field(
        ...,
        description="A field containing any useful reindexing operations for Tablassert to Preform",
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


class ReindexingAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "ReindexingOutput"),
            input_schema=UserInput,
            output_schema=ReindexingOutput,
        )


def run_reindexing_agent(user_input: dict[str, object]) -> ReindexingOutput:
    agent = ReindexingAgent()
    try:
        return(agent.invoke(user_input))
    except AgentInvocationError as e:
        raise e
