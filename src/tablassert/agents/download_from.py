__author__ = "Skye Lane Goetz"
__status__ = "Development"


from pydantic import Field, conlist, constr, model_validator, HttpUrl
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from tablassert.agents.toolkit import (
    AgentInvocationError,
    get_system_prompt,
    get_user_input,
    ollama_client,
    UserInput,
    LLMAgent,
)


class DownloadFromOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    download_from: A url telling Tablassert where to download the data you want to process from

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about
    """

    download_from: HttpUrl = Field(
        ...,
        description="A url telling Tablassert where to download the data you want to process from",
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


class DownloadFromAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "DownloadFromOutput"),
            input_schema=UserInput,
            output_schema=DownloadFromOutput,
        )


def run_download_from_agent(user_input: dict[str, object]) -> DownloadFromOutput:
    agent = DownloadFromAgent()
    try:
        return(agent.invoke(user_input))
    except AgentInvocationError as e:
        raise e
