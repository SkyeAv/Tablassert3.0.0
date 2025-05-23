__author__ = "Skye Lane Goetz"
__status__ = "Development"


from tablassert.agents.is_precleaning_operation import (
    run_is_precleaning_operation_agent,
)
from tablassert.agents.is_reindexing_operation import run_is_reindexing_operation_agent
from tablassert.agents.is_sections_template import run_is_sections_template_agent
from tablassert.agents.is_subject_object import run_is_subject_object_agent
from tablassert.agents.remove_substrings import run_remove_substrings_agent
from tablassert.agents.regex_operations import run_regex_operations_agent
from tablassert.agents.math_operation import run_math_operation_agent
from tablassert.agents.download_from import run_download_from_agent
from tablassert.agents.split_explode import run_split_explode_agent
from tablassert.agents.is_attribute import run_is_attribute_agent
from tablassert.agents.in_organism import run_in_organism_agent
from tablassert.agents.fill_column import run_fill_column_agent
from tablassert.agents.node_params import run_node_params_agent
from tablassert.agents.reindexing import run_reindexing_agent
from tablassert.agents.provenance import run_provenance_agent
from tablassert.agents.prioritize import run_prioritize_agent
from tablassert.agents.attributes import run_attributes_agent
from tablassert.agents.is_math import run_is_math_agent
from tablassert.agents.params import run_params_agent
from tablassert.agents.prefix import run_prefix_agent
from tablassert.agents.suffix import run_suffix_agent
from tablassert.agents.avoid import run_avoid_agent
from tablassert.agents.pred import run_pred_agent

# UserInput is obsolete now
from tablassert.agents.toolkit import get_user_input
from tablassert.cfg import Section, TableConfig


class TableConfigAgent:

    @classmethod
    def msg_to_user_input(cls, x: object) -> dict[str, str]:
        return {"chat_msg": str(x)}

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
        IsSubjectObjectOutput = run_is_subject_object_agent(user_input)
        subject_portion = IsSubjectObjectOutput.subject_portion
        object_portion = IsSubjectObjectOutput.object_portion
        is_subject_object_questions = IsSubjectObjectOutput
        suggested_questions = cls.extend_questions(
            is_subject_object_questions, suggested_questions
        )
        node_portions = {"subj": subject_portion, "obj": object_portion}
        nodes = {}
        for node, portion in node_portions.items():
            inner_node_params = {}
            node_input = cls.msg_to_user_input(portion)
            NodeParamsOutput = run_node_params_agent(node_input)
            mode = NodeParamsOutput.mode
            value = NodeParamsOutput.value
            node_params_questions = NodeParamsOutput.suggested_questions
            suggested_questions = cls.extend_questions(
                node_params_questions, suggested_questions
            )
            inner_node_params.update({"mode": mode, "value": value})
            IsPrecleaningOutput = run_is_precleaning_operation_agent(node_input)
            is_precleaning_questions = IsPrecleaningOutput.suggested_questions
            suggested_questions = cls.extend_questions(
                is_precleaning_questions, suggested_questions
            )
            has_in_organism = IsPrecleaningOutput.has_in_organism
            if has_in_organism:
                in_organism_portion = IsPrecleaningOutput.in_organism_portion
                in_organism_input = cls.msg_to_user_input(in_organism_portion)
                InOrganismOutput = run_in_organism_agent(in_organism_input)
                in_organism = InOrganismOutput.in_organism
                inner_node_params.update({"in_organism": in_organism})
                in_organism_questions = InOrganismOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    in_organism_questions, suggested_questions
                )
            has_prioritize = IsPrecleaningOutput.has_prioritize
            if has_prioritize:
                prioritize_portion = IsPrecleaningOutput.prioritize_portion
                prioritize_input = cls.msg_to_user_input(prioritize_portion)
                PrioritizeOutput = run_prioritize_agent(prioritize_input)
                prioritize = PrioritizeOutput.prioritize
                inner_node_params.update({"prioritize": prioritize})
                prioritize_questions = PrioritizeOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    prioritize_questions, suggested_questions
                )
            has_avoid = IsPrecleaningOutput.has_avoid
            if has_avoid:
                avoid_portion = IsPrecleaningOutput.avoid_portion
                avoid_input = cls.msg_to_user_input(avoid_portion)
                AvoidOutput = run_avoid_agent(avoid_input)
                avoid = AvoidOutput.avoid
                inner_node_params.update({"avoid": avoid})
                avoid_questions = AvoidOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    avoid_questions, suggested_questions
                )
            has_prefix = IsPrecleaningOutput.has_prefix
            if has_prefix:
                prefix_portion = IsPrecleaningOutput.prefix_portion
                prefix_input = cls.msg_to_user_input(prefix_portion)
                PrefixOutput = run_prefix_agent(prefix_input)
                prefix = PrefixOutput.prefix
                inner_node_params.update({"prefix": prefix})
                prefix_questions = PrefixOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    prefix_questions, suggested_questions
                )
            has_suffix = IsPrecleaningOutput.has_suffix
            if has_suffix:
                suffix_portion = IsPrecleaningOutput.suffix_portion
                suffix_input = cls.msg_to_user_input(suffix_portion)
                SuffixOutput = run_suffix_agent(suffix_input)
                suffix = SuffixOutput.suffix
                inner_node_params.update({"suffix": suffix})
                suffix_questions = SuffixOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    suffix_questions, suggested_questions
                )
            has_fill_column = IsPrecleaningOutput.has_fill_column
            if has_fill_column:
                fill_column_portion = IsPrecleaningOutput.fill_column_portion
                fill_column_input = cls.msg_to_user_input(fill_column_portion)
                FillColumnOutput = run_fill_column_agent(fill_column_input)
                fill_column = FillColumnOutput.fill_column
                inner_node_params.update({"fill_column": fill_column})
                fill_column_questions = FillColumnOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    fill_column_questions, suggested_questions
                )
            has_remove = IsPrecleaningOutput.has_remove
            if has_remove:
                remove_portion = IsPrecleaningOutput.remove_portion
                remove_input = cls.msg_to_user_input(remove_portion)
                RemoveSubstringsOutput = run_remove_substrings_agent(remove_input)
                remove = RemoveSubstringsOutput.remove
                inner_node_params.update({"remove": remove})
                remove_substrings_questions = RemoveSubstringsOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    remove_substrings_questions, suggested_questions
                )
            has_regex = IsPrecleaningOutput.has_regex
            if has_regex:
                regex_portion = IsPrecleaningOutput.regex_portion
                regex_input = cls.msg_to_user_input(regex_portion)
                RegexOperationsOutput = run_regex_operations_agent(regex_input)
                regex = RegexOperationsOutput.regex
                inner_node_params.update({"regex": regex})
                regex_operations_questions = RegexOperationsOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    regex_operations_questions, suggested_questions
                )
            has_split_explode = IsPrecleaningOutput.has_split_explode
            if has_split_explode:
                split_explode_portion = IsPrecleaningOutput.split_explode_portion
                split_explode_input = cls.msg_to_user_input(split_explode_portion)
                SplitExplodeOutput = run_split_explode_agent(split_explode_input)
                split_explode = SplitExplodeOutput.split_explode
                inner_node_params.update({"split_explode": split_explode})
                split_explode_questions = SplitExplodeOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    split_explode_questions, suggested_questions
                )
            nodes.update({node: inner_node_params})
        PredOutput = run_pred_agent(user_input)
        pred_questions = PredOutput.suggested_questions
        suggested_questions = cls.extend_questions(pred_questions, suggested_questions)
        pred = PredOutput.pred
        nodes.update({"pred": pred})
        triple = {"triple": nodes}
        section.update(triple)
        IsReindexingOperationOutput = run_is_reindexing_operation_agent(user_input)
        is_reindexing_operation_questions = (
            IsReindexingOperationOutput.suggested_questions
        )
        has_reindexing = IsReindexingOperationOutput.has_reindexing
        suggested_questions = cls.extend_questions(
            is_reindexing_operation_questions, suggested_questions
        )
        if has_reindexing:
            reindexing_operations: list[str] = (
                IsReindexingOperationOutput.reindexing_operations
            )
            reindexing_portion = []
            for reindexing_operation in reindexing_operations:
                reindexing_input = cls.msg_to_user_input(reindexing_operation)
                ReindexingOutput = run_reindexing_agent(reindexing_input)
                reindexing_questions = ReindexingOutput.suggested_questions
                suggested_questions = cls.extend_questions(
                    reindexing_questions, suggested_questions
                )
                reindexing_intermediate = ReindexingOutput.reindexing
                reindexing_portion.append(reindexing_intermediate.model_dump())
            reindexing = {"reindexing": reindexing_portion}
            section.update(reindexing)
        return section, suggested_questions

    @classmethod
    def invoke(cls, user_input: dict[str, object]) -> TableConfig:
        table_config = {}
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
            table_config.update({"template": template})
        if has_sections:
            for portion in IsSectionsTemplateOutput.sections_portions:
                section, questions = cls.section_interpreter(
                    (cls.msg_to_user_input(portion))
                )
                sections.append(section)
                suggested_questions.append(questions)
            table_config.update({"sections": sections})
        print("\n")
        print(table_config)
        print("\n")
        print(suggested_questions)


if __name__ == "__main__":
    agent = TableConfigAgent
    user_input = get_user_input()
    TableConfigAgent.invoke(user_input)
