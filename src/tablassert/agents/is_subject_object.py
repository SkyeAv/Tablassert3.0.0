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


class IsSubjectObjectOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    subject_portion: Extract the full portion of the message related to the subject in the subject/predicate/object triple.
        This must include:
            - The object/column/value identifier (e.g., NCBITaxon:92310, "where the column is AB", "Alzheimers Disease")
            - Any relevant modifiers, conditions, or column-level instructions (e.g., "in the column CD", "explode by ','")
        Example:
            Input: "Now I want the subject to be MONDO:2260 and to split_explode the column by ',' and in the column CD, then the object to be NCBITaxon:92310..."
            Output: "MONDO:2260 and to split_explode the column by ',' and in the column CD"

    object_portion: Extract the full portion of the message related to the object in the subject/predicate/object triple.
        This must include:
            - The object/column/value identifier (e.g., NCBITaxon:92310, "where the column is AB", "Alzheimers Disease")
            - All relevant details, qualifiers, or transformations related to the object (e.g., "regex = ("", " ", str)" and "replacements = prink")
        Example:
            Input: "...then the object to be NCBITaxon:92310 where the column is AB and the replacements = prink"
            Output: "NCBITaxon:92310 where the column is AB and the replacements = prink"

    suggested_questions: An optional list of follow-up questions the model can ask to clarify any parts of the JSON output it is uncertain about.

    SUPER IMPORTANT CAVEAT:

    Do NOT shorten or simplify these portions. Preserve all information that modifies, qualifies, or constrains the subject/object.
    """

    subject_portion: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description='A string of text pertaining to the subject part of the subject/predicate/object triple specified in a message (e.g., "Other thing I\'m talking about. I want Mono as the subject with column AA and expected taxa = NCBITaxon:9606 and Alzhemiers Disease as the object with column AB and expected taxa = NCBITaxon:9606 -> "I want Mono as the subject with column AA and expected taxa = NCBITaxon:9606")',
    )
    object_portion: constr(min_length=1, strip_whitespace=True) = Field(
        ...,
        description='A string of text pertaining to the object part of the subject/predicate/object triple specified in a message (e.g., "Other thing I\'m talking about. I want Mono as the subject with column AA and expected taxa = NCBITaxon:9606 and Alzhemiers Disease as the object with column AB and expected taxa = NCBITaxon:9606 -> "Alzhemiers Disease as the object with column AB and expected taxa = NCBITaxon:9606")',
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


class IsSubjectObjectAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("boolean_split", "IsSubjectObjectOutput"),
            input_schema=UserInput,
            output_schema=IsSubjectObjectOutput,
        )


def run_is_subject_object_agent(user_input: dict[str, object]) -> IsSubjectObjectOutput:
    agent = IsSubjectObjectAgent()
    try:
        return(agent.invoke(user_input))
    except AgentInvocationError as e:
        raise e
