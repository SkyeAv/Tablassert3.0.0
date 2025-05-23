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


class IsReindexingOperationOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    has_reindexing: A boolean denoting if the "reindexing" operation (A field containing any useful reindexing operations for Tablassert to Preform) is implied in the message
    reindexing_operations: An optional list of text pertaining to each individual reindexing operation specified in a message (e.g., "First I want to filter the p-value less than 0.05 then remove anything the p-value column that equals the 0.00" -> ["First I want to filter the p-value less than 0.05", "then remove anything the p-value column that equals the 0.00"])
    - IF has_reindexing == True THEN reindexing_operations IS REQUIRED

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about
    """

    has_reindexing: bool = Field(
        ...,
        description='A boolean denoting if the "reindexing" operation (A field containing any useful reindexing operations for Tablassert to Preform) is implied in the message',
    )
    reindexing_operations: (
        conlist(
            item_type=constr(min_length=1, strip_whitespace=True),
            min_length=1,
        )
        | None
    ) = Field(
        default=None,
        description='An optional list of text pertaining to each individual reindexing operation specified in a message (e.g., "First I want to filter the p-value less than 0.05 then remove anything the p-value column that equals the 0.00" -> ["First I want to filter the p-value less than 0.05", "then remove anything the p-value column that equals the 0.00"])',
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


class IsReindexingOperationAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt(
                "boolean_split", "IsReindexingOperationOutput"
            ),
            input_schema=UserInput,
            output_schema=IsReindexingOperationOutput,
        )


def run_is_reindexing_operation_agent(
    user_input: dict[str, object],
) -> IsReindexingOperationOutput:
    agent = IsReindexingOperationAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
