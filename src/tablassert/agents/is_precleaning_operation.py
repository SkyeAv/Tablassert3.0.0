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


class IsPrecleaningOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    has_in_organism: A boolean denoting if the "in_organism" attribute (A NCBITaxon: curie denoting which organism the scientific finding relates to/was discovered in. NCBITaxon:9606 denotes humans for example) is implied in the message
    in_organism_portion: <See Important Instructions>

    has_prioritize: A boolean denoting if the "prioritize" attribute (A list of biolink: curies that Tablasserts databases should prioritize when mapping strings to CURIES) is implied in the message
    prioritize_portion: <See Important Instructions>

    has_avoid: A boolean denoting if the "avoid" attribute (A list of biolink: curies that Tablasserts databases should avoid when mapping strings to CURIES) is implied in the message
    avoid_portion: <See Important Instructions>

    has_prefix: A boolean denoting if the "prefix" attribute (A prefix to add to the begining of the values denoted by a node before they're processed/passed to databases/etc..) is implied in the message
    prefix_portion: <See Important Instructions>

    has_suffix: A boolean denoting if the "suffix" attribute (A suffix to add to the end of the values denoted by a node before they're processed/passed to databases/etc..) is implied in the message
    suffix_portion: <See Important Instructions>

    has_fill_column: A boolean denoting if the "fill_column" attribute (A valid polars fill_null strategy parameter to fill the null values in a given column) is implied in the message
    fill_column_portion: <See Important Instructions>

    has_remove: A boolean denoting if the "remove" attribute (Substrings to apply re.sub(substring, "", string) to across the node data) is implied in the message
    remove_portion: <See Important Instructions>

    has_regex: A boolean denoting if the "regex" attribute (Regex to apply re.sub(pattern, replacement, string) to across the node data) is implied in the message
    regex_portion: <See Important Instructions>

    has_split_explode: A boolean denoting if the "split_explode" attribute (A field specificying which delimiter to split the string encoded in a node by into a list before exploding values in that list thier own respective nodes, like with a pandas or polars explode column) is implied in the message
    split_explode_portion: <See Important Instructions>

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT INSTRUCTIONS:

    Any "<FACTOR>_portion" Key is string of text related to the factor specified in a message (e.g., "I want to use Mono as the factor and focus on Alzheimer's Disease as the other factor" -> "I want to use Mono as the factor").
    """

    has_in_organism: bool = Field(
        ...,
        description='A boolean denoting if the "in_organism" attribute (A NCBITaxon: curie denoting which organism the scientific finding relates to/was discovered in. NCBITaxon:9606 denotes humans for example) is implied in the message',
    )
    in_organism_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_prioritize: bool = Field(
        ...,
        description='A boolean denoting if the "prioritize" attribute (A list of biolink: curies that Tablasserts databases should prioritize when mapping strings to CURIES) is implied in the message',
    )
    prioritize_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_avoid: bool = Field(
        ...,
        description='A boolean denoting if the "avoid" attribute (A list of biolink: curies that Tablasserts databases should avoid when mapping strings to CURIES) is implied in the message',
    )
    avoid_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_prefix: bool = Field(
        ...,
        description='A boolean denoting if the "prefix" attribute (A prefix to add to the begining of the values denoted by a node before they\'re processed/passed to databases/etc..) is implied in the message',
    )
    prefix_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_suffix: bool = Field(
        ...,
        description='A boolean denoting if the "suffix" attribute (A suffix to add to the end of the values denoted by a node before they\'re processed/passed to databases/etc..) is implied in the message',
    )
    suffix_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_fill_column: bool = Field(
        ...,
        description='A boolean denoting if the "fill_column" attribute (A valid polars fill_null strategy parameter to fill the null values in a given column) is implied in the message',
    )
    fill_column_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_remove: bool = Field(
        ...,
        description='A boolean denoting if the "remove" attribute (Substrings to apply re.sub(substring, "", string) to across the node data) is implied in the message',
    )
    remove_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_regex: bool = Field(
        ...,
        description='A boolean denoting if the "regex" attribute (Regex to apply re.sub(pattern, replacement, string) to across the node data) is implied in the message',
    )
    regex_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
    )
    has_split_explode: bool = Field(
        ...,
        description='A boolean denoting if the "split_explode" attribute (A field specificying which delimiter to split the string encoded in a node by into a list before exploding values in that list thier own respective nodes, like with a pandas or polars explode column) is implied in the message',
    )
    split_explode_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None, description="<See Important Instructions>"
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


class IsPrecleaningAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("boolean_split", "IsPrecleaningOutput"),
            input_schema=UserInput,
            output_schema=IsPrecleaningOutput,
        )


def run_is_precleaning_agent():
    agent = IsPrecleaningAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_is_precleaning_agent()
