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


class IsSectionsTemplateOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    has_template: A boolean denoting if the user intends to build a template (if the user is building a new section the value here should be "False")
    template_portion: An optional string of text pertaining to the template specified in a message (e.g., I want a template using Mono as the subject and the first section to use Alzhemiers Disease as a subject -> "I want a template using Mono as the subject")

    has_section: A boolean denoting if the user intends to build a section (if the user is building a template the value here should be "False")
    sections_portions: An optional list of text pertaining to each individual section specified in a message (e.g., I want the first section to use Alzhemiers Disease as a subject and the second section to use Mono as a subject -> ["I want the first section to use Alzhemiers Disease as a subject", "and the second section to use Mono as a subject"])

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT KEYWORDS:

    Keywords that indicate a section include (but aren't limited to) the mono/bi/trigrams:
        section
        sections
        new section
        new sections
        new part
        new parts
        augment
        change template
        change Templates
        not template
        not templates

    Keywords that indicate a template include (but aren't limited to) the mono/bi/trigrams:
        template
        templates
        base
        base config
        base configuration
        template configuration
        yaml template
        &template
        table config template
        table configs template
        tableconfig template
        tableconfigs template
        not section
        not sections

    SUPER IMPORTANT CLARIFICATION:

    IF NEITHER A template NOT section IS SPECIFIED TREAT THE MESSAGE AS A SECTION
    """

    has_template: bool = Field(
        ...,
        description='A boolean denoting if the user intends to build a template (if the user is building a new section the value here should be "False")',
    )
    template_portion: constr(min_length=1, strip_whitespace=True) | None = Field(
        default=None,
        description='An string of text pertaining to the template specified in a message (e.g., "I want a template using Mono as the subject and the first section to use Alzhemiers Disease as a subject" -> "I want a template using Mono as the subject")',
    )
    has_section: bool = Field(
        ...,
        description='A boolean denoting if the user intends to build a section (if the user is building a template the value here should be "False")',
    )
    sections_portion: (
        conlist(
            item_type=constr(min_length=1, strip_whitespace=True),
            min_length=1,
        )
        | None
    ) = Field(
        default=None,
        description='An optional list of text pertaining to each individual section specified in a message (e.g., "I want the first section to use Alzhemiers Disease as a subject and the second section to use Mono as a subject" -> ["I want the first section to use Alzhemiers Disease as a subject", "and the second section to use Mono as a subject"])',
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


class IsSectionsTemplateAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt(
                "boolean_split", "IsSectionsTemplateOutput"
            ),
            input_schema=UserInput,
            output_schema=IsSectionsTemplateOutput,
        )


def run_is_sections_template_agent():
    agent = IsSectionsTemplateAgent()
    user_input = get_user_input()
    try:
        print(agent.invoke(user_input))
    except AgentInvocationError as e:
        print(e)


run_is_sections_template_agent()
