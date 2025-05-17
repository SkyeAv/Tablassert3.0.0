__author__ = "Skye Lane Goetz"
__status__ = "Development"


from tablassert.cfg import TextBasedImage, DelimitedFile, ExcelSpreadSheet
from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptContextProviderBase,
)
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field, conlist, constr, model_validator
from tablassert.agents.ext import run_ext_engine
from tablassert.agents.toolkit import (
    AgentInvocationError,
    get_system_prompt,
    get_user_input,
    ollama_client,
    UserInput,
    LLMAgent,
)


class TextBasedPDFOutput(BaseIOSchema):
    r"""
    SCHEMA ANNOTATIONS:

    params: Parameters specifying how Tablassert should first treat/process said data
        ext: The current file extension
        pages: A comma delimited list (ENSURE THIS VALUE IS PYTHON TYPE STRING) of pages abiding by the regex \"^(\d+|(\d+)(\-(\d+|end))?)(\,(\d+|(\d+)(\-(\d+|end))?))*$\" for the pages parameter in Camelot's read_pdf (DO NOT UNDER ANY CIRCUMSTANCES ADD "START-" BEFORE or "-END" AFTER ANY PAGES IN THIS FIELD)
        flavor: Either lattice or stream for the flavor parameter in Camelot's read_pdf

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about
    """

    params: TextBasedImage = Field(
        ...,
        description="Parameters specifying how Tablassert should first treat/process said data",
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


class DelimitedFileOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    params: Parameters specifying how Tablassert should first treat/process said data
        ext: The current file extension
        delimiter: The delimiter separating values in a delimited file
        start: The row in the file to start at when extracting content from the file
        end: The row in the file to end at when extracting content from the file
        rows: A list of specific rows to extract content from in the file

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT CAVEAT:

    Use at least one of the following keys in "params": "start", "end", or "rows"
    You may not use both "rows" and "start/"end" together
    If both "start" and "end" are used, the range must be at least 2 rows
    """

    params: DelimitedFile = Field(
        ...,
        description="Parameters specifying how Tablassert should first treat/process said data",
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


class ExcelSpreadSheetOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    params: Parameters specifying how Tablassert should first treat/process said data
        ext: The current file extension
        sheet: The name of the sheet in the excel spreadhseet to extract content from
        start: The row in the file to start at when extracting content from the file
        end: The row in the file to end at when extracting content from the file
        rows: A list of specific rows to extract content from in the file

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    IMPORTANT CAVEAT:

    Use at least one of the following keys in "params": "start", "end", or "rows"
    You may not use both "rows" and "start/"end" together
    If both "start" and "end" are used, the range must be at least 2 rows
    """

    params: ExcelSpreadSheet = Field(
        ...,
        description="Parameters specifying how Tablassert should first treat/process said data",
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


USER_INPUT: str = get_user_input()
FILE_EXTENSION: str = run_ext_engine(USER_INPUT.get("chat_msg"))


class ExtensionProvider(SystemPromptContextProviderBase):
    def __init__(self, title: str = "File Extension"):
        super().__init__(title)
        self.extension = FILE_EXTENSION

    def get_info(self) -> str:
        return f'You\'re writing "params" for a {self.extension} file'


def get_output_type() -> BaseIOSchema:
    ext = FILE_EXTENSION
    match ext:
        case "xlsx" | "xls" | "xlsb" | "xlsm":
            return ExcelSpreadSheetOutput
        case "csv" | "tsv" | "txt":
            return DelimitedFileOutput
        case "pdf":
            return TextBasedPDFOutput


class ParamsAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", get_output_type().__name__),
            input_schema=UserInput,
            output_schema=get_output_type(),
        )

    def register_extention_context(self):
        self.agent.register_context_provider("File Extension", ExtensionProvider())


def run_params_agent():
    agent = ParamsAgent()
    try:
        agent.register_extention_context()
        print(agent.invoke(USER_INPUT))
    except AgentInvocationError as e:
        print(e)


run_params_agent()
