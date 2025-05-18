__author__ = "Skye Lane Goetz"
__status__ = "Development"


from tablassert.agents.is_sections_template import run_is_sections_template_agent
from tablassert.agents.download_from import run_download_from_agent
from tablassert.agents.params import run_params_agent
from tablassert.agents.toolkit import UserInput
from tablassert.cfg import Section, TableConfig


class TableConfigAgent:

    @staticmethod
    def msg_to_user_input(x: object) -> UserInput:
        return UserInput(chat_msg=str(x))
    
    @staticmethod
    def section_intepreter(user_input: dict[str, object]) -> tuple[Section, list[str]]:
        suggested_questions = []
        section = {}
        DownloadFromOutput = run_download_from_agent(user_input)
        download_from = DownloadFromOutput.download_from
        download_from_questions = DownloadFromOutput.suggested_questions
        if download_from_questions:
            suggested_questions.append(download_from_questions)
        Output: object = run_params_agent(user_input)
        params = Output.params
        params_questions = Output.suggested_questions
        if params_questions:
            suggested_questions.append(params_questions)
        location = {"location": {"download_from": download_from, "params": params}}
        section.update(location)

    @staticmethod
    def invoke(user_input: dict[str, object]) -> TableConfig:
        template = {}
        sections = []
        suggested_questions = []
        IsSectionsTemplateOutput = run_is_sections_template_agent(user_input)
        has_template = IsSectionsTemplateOutput.has_template
        has_sections = IsSectionsTemplateOutput.has_sections
        is_sections_template_questions = IsSectionsTemplateOutput.suggested_questions
        if is_sections_template_questions:
            suggested_questions.append(is_sections_template_questions)
        if has_template:
            section, questions = section_intepreter(msg_to_user_input(IsSectionsTemplateOutput.template_portion))
            template.update(section)
            suggested_questions.append(questions)
        if has_sections:
            for portion in IsSectionsTemplateOutput.sections_portions:
                section, questions = section_intepreter((msg_to_user_input(portion)))
                sections.append(section)
                suggested_questions.append(questions)
