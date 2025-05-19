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


class IsAttributeOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    has_sample_size: A boolean denoting if the "sample_size" attribute (An attribute denoting the sample size used in a particular study or to determine a specific finding) is implied in the message
    sample_size_portion: <See Important Instructions>

    has_p_value: A boolean denoting if the "p_value" attribute (An attribute denoting the experimental p-value (FDR corrected p-vales are preferred) attirbuted to a finding) is implied in the message
    p_value_portion: <See Important Instructions>

    has_fdr: A boolean denoting if the "fdr" attribute (An attribute denoting FDR or false discover rate correct applied to the statictics regarding a specific finding) is implied in the message
    fdr_portion: <See Important Instructions>

    has_strength: A boolean denoting if the "strength" attribute (An attribute denoting the strength of a finding, often the strength of a statistical relationship like a spearman correlation) is implied in the message
    strength_portion: <See Important Instructions>

    has_statistic: A boolean denoting if the "statistic" attribute (An attribute denoting the statistic used in the aforementioned strength field, something like spearman correlation would be a valid statistic field) is implied in the message
    statistic_portion: <See Important Instructions>

    notes: An optional unique free-text attribute capturing anything that cannot be encoded in the fields above or attributes in general but is relevant to understanding the finding(s) at hand

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT INSTRUCTIONS:

    Any "<FACTOR>_portion" Key is string of text related to the factor specified in a message (e.g., "I want to use Mono as the factor and focus on Alzheimer's Disease as the other factor" -> "I want to use Mono as the factor").
    """

    has_sample_size: bool = Field(
        ...,
        description='A boolean denoting if the "sample_size" attribute (An attribute denoting the sample size used in a particular study or to determine a specific finding) is implied in the message',
    )
    sample_size_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_p_value: bool = Field(
        ...,
        description='A boolean denoting if the "p_value" attribute (An attribute denoting the experimental p-value (FDR corrected p-vales are preferred) attirbuted to a finding) is implied in the message',
    )
    p_value_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_fdr: bool = Field(
        ...,
        description='A boolean denoting if the "fdr" attribute (An attribute denoting FDR or false discover rate correct applied to the statictics regarding a specific finding) is implied in the message',
    )
    fdr_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_strength: bool = Field(
        ...,
        description='A boolean denoting if the "strength" attribute (An attribute denoting the strength of a finding, often the strength of a statistical relationship like a spearman correlation) is implied in the message',
    )
    strength_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_statistic: bool = Field(
        ...,
        description='A boolean denoting if the "statistic" attribute (An attribute denoting the statistic used in the aforementioned strength field, something like spearman correlation would be a valid statistic field) is implied in the message',
    )
    statistic_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    notes: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None,
        description="A unique free-text attribute capturing anything that cannot be encoded in the fields above or attributes in general but is relevant to understanding the finding(s) at hand",
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


class IsAttributeAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("boolean_split", "IsAttributeOutput"),
            input_schema=UserInput,
            output_schema=IsAttributeOutput,
        )


def run_is_attribute_agent(user_input: dict[str, object]) -> IsAttributeOutput:
    agent = IsAttributeAgent()
    try:
        return(agent.invoke(user_input))
    except AgentInvocationError as e:
        raise e
