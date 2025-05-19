__author__ = "Skye Lane Goetz"
__status__ = "Development"


from tablassert.agents.is_sections_template import run_is_sections_template_agent
from tablassert.agents.math_operation import run_math_operation_agent
from tablassert.agents.download_from import run_download_from_agent
from tablassert.agents.is_attribute import run_is_attribute_agent
from tablassert.agents.provenance import run_provenance_agent
from tablassert.agents.attributes import run_attributes_agent
from tablassert.agents.is_math import run_is_math_agent
from tablassert.agents.params import run_params_agent
from tablassert.agents.toolkit import UserInput
from tablassert.cfg import Section, TableConfig


class TableConfigAgent:

    @classmethod
    def msg_to_user_input(cls, x: object) -> UserInput:
        return UserInput(chat_msg=str(x))

    @classmethod
    def extend_questions(cls, x: object, suggested_questions: list[str]) -> list[str]:
        if x:
            suggested_questions.extend(x)
        return suggested_questions

    @classmethod
    def attribute_interpreter(
        cls, attribute: str, IsAttributeOutput: object, suggested_questions: list[str]
    ) -> tuple[dict[str, object] | None, list[str]]:
        has_attribute: str = getattr(IsAttributeOutput, (f"has_{attribute}"))
        if has_attribute:
            attribute_block = {}
            portion: str = getattr(IsAttributeOutput, (f"{attribute}_portion"))
            attribute_input = cls.msg_to_user_input(portion)
            IsMathOutput = run_is_math_agent(attribute_input)
            is_math_questions = IsMathOutput.suggested_questions
            suggested_questions = cls.extend_questions(
                is_math_questions, suggested_questions
            )
            has_math = IsMathOutput.has_math
            if has_math:
                math_block = []
                math_portions: list[str] = IsMathOutput.math_portions
                for math_portion in math_portions:
                    math_input = cls.msg_to_user_input(math_portion)
                    MathOutput = run_math_operation_agent(math_input)
                    math_params = MathOutput.math
                    math_questions = MathOutput.suggested_questions
                    suggested_questions = cls.extend_questions(
                        math_questions, suggested_questions
                    )
                    math_block.append(math_params.model_dump())
                math = {"math": math_block}
                attribute_block.update(math)
            AttributeOutput = run_attributes_agent(attribute_input)
            mode = AttributeOutput.mode
            value = AttributeOutput.value
            attributes_question = AttributeOutput.suggested_questions
            suggested_questions = cls.extend_questions(
                attributes_question, suggested_questions
            )
            attribute_params = {"mode": mode, "value": value}
            attribute_block.update(attribute_params)
            output = {attribute: attribute_block}
            return (output, suggested_questions)
        return (None, suggested_questions)

    @classmethod
    def section_interpreter(
        cls, user_input: dict[str, object]
    ) -> tuple[Section, list[str]]:
        suggested_questions = []
        section = {}
        DownloadFromOutput = run_download_from_agent(user_input)
        download_from = DownloadFromOutput.download_from
        download_from_questions = DownloadFromOutput.suggested_questions
        suggested_questions = cls.extend_questions(
            download_from_questions, suggested_questions
        )
        Output: object = run_params_agent(user_input)
        params = Output.params
        params_questions = Output.suggested_questions
        suggested_questions = cls.extend_questions(
            params_questions, suggested_questions
        )
        location = {
            "location": {"download_from": download_from, "params": params.model_dump()}
        }
        section.update(location)
        ProvenanceOutput = run_provenance_agent(user_input)
        provenance = ProvenanceOutput.provenance
        provenance_questions = ProvenanceOutput.suggested_questions
        suggested_questions = cls.extend_questions(
            provenance_questions, suggested_questions
        )
        provenance = {"provenance": provenance.model_dump()}
        section.update(provenance)
        IsAttributeOutput = run_is_attribute_agent(user_input)
        notes = IsAttributeOutput.notes
        is_attribute_questions = IsAttributeOutput.suggested_questions
        suggested_questions = cls.extend_questions(
            is_attribute_questions, suggested_questions
        )
        sample_size, sample_size_questions = cls.attribute_interpreter(
            "sample_size", IsAttributeOutput, suggested_questions
        )
        suggested_questions = cls.extend_questions(
            sample_size_questions, suggested_questions
        )
        p_value, p_value_questions = cls.attribute_interpreter(
            "p_value", IsAttributeOutput, suggested_questions
        )
        suggested_questions = cls.extend_questions(
            p_value_questions, suggested_questions
        )
        fdr, fdr_questions = cls.attribute_interpreter(
            "fdr", IsAttributeOutput, suggested_questions
        )
        suggested_questions = cls.extend_questions(fdr_questions, suggested_questions)
        strength, strength_questions = cls.attribute_interpreter(
            "strength", IsAttributeOutput, suggested_questions
        )
        suggested_questions = cls.extend_questions(
            strength_questions, suggested_questions
        )
        statistic, statistic_questions = cls.attribute_interpreter(
            "statistic", IsAttributeOutput, suggested_questions
        )
        suggested_questions = cls.extend_questions(
            statistic_questions, suggested_questions
        )
        attributes = {
            "attributes": {
                "sample_size": sample_size,
                "p_value": p_value,
                "fdr": fdr,
                "strength": strength,
                "statistic": statistic,
                "notes": notes,
            }
        }
        section.update(attributes)
        return section

    @classmethod
    def invoke(cls, user_input: dict[str, object]) -> TableConfig:
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
            section, questions = cls.section_interpreter(
                cls.msg_to_user_input(IsSectionsTemplateOutput.template_portion)
            )
            template.update(section)
            suggested_questions.append(questions)
        if has_sections:
            for portion in IsSectionsTemplateOutput.sections_portions:
                section, questions = cls.section_interpreter(
                    (cls.msg_to_user_input(portion))
                )
                sections.append(section)
                suggested_questions.append(questions)
