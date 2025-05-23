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


class PredOutput(BaseIOSchema):
    """
    SCHEMA ANNOTATIONS:

    pred: The predicate in a subject/predicate/object (or thing/relationship_between_thing/thing) knowledge triple, these comprise the knowledge encoded in a Tablassert knowledge graph. ONLY biolink:predicates are accepted in this field

    suggested_questions: An optional list of suggested follow up questions to clarify any JSON output you're still unsure about

    LIST OF pred: (include "biolink:" before all bellow)

        overlaps
        binds
        same_as
        catalyzes
        publisher
        expresses
        editor
        closely_related_to
        coexpressed_with
        exact_match
        related_condition
        translates_to
        similar_to
        physically_interacts_with
        interacts_with
        provider
        related_to
        coexists_with
        affects
        affects_response_to
        location_of
        produces
        causes
        located_in
        member_of
        in_complex_with
        expressed_in
        target_for
        associated_with
        has_part
        in_taxon
        treats
        treated_by
        treated_in_studies_by
        studied_to_treat
        ameliorates_condition
        preventative_for_condition
        promotes_condition
        exacerbates_condition
        related_to_at_concept_level
        related_to_at_instance_level
        has_biomarker
        biomarker_for
        associated_with_resistance_to
        associated_with_sensitivity_to
        resistance_associated_with
        sensitivity_associated_with
        consumed_by
        consumes
        has_metabolite
        is_metabolite_of
        increased_amount_of
        decreased_amount_in
        has_decreased_amount
        has_increased_amount
        has_gene_product
        gene_product_of
        participates_in
        has_participant
        contributes_to
        contribution_from
        capable_of
        can_be_carried_out_by
        is_active_ingredient_of
        has_active_ingredient
        has_active_component
        is_substrate_of
        has_substrate
        subclass_of
        superclass_of
        variant_part_of
        has_variant_part
        is_sequence_variant_of
        has_sequence_variant
        in_linkage_disequilibrium_with
        genetic_association
        genetically_associated_with
        gene_associated_with_condition
        condition_associated_with_gene
        phenotype_of
        has_phenotype
        location_of_disease
        occurs_in_disease
        condition_promoted_by
        condition_ameliorated_by
        condition_prevented_by
        condition_predisposed_by
        condition_exacerbated_by
        associated_with_decreased_likelihood_of
        associated_with_increased_likelihood_of
        associated_with_likelihood_of
        likelihood_associated_with
        affects_likelihood_of
        likelihood_affected_by
        response_increased_by
        response_decreased_by
        decreases_response_to
        increases_response_to
        has_output
        is_output_of
        has_input
        is_input_of
    """  # List of Preds is a List of the 100 Least Redundant

    pred: constr(
        min_length=8, pattern=r"^[A-Za-z]+:[A-Za-z0-9._/-]+$", strip_whitespace=True
    ) = Field(
        ...,
        description="The predicate in a subject/predicate/object (or thing/relationship_between_thing/thing) knowledge triple, these comprise the knowledge encoded in a Tablassert knowledge graph. ONLY biolink:predicates are accepted in this field",
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


class PredAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            llm_client=ollama_client(),
            llm="mistral",
            system_prompt=get_system_prompt("default", "PredOutput"),
            input_schema=UserInput,
            output_schema=PredOutput,
        )


def run_pred_agent(user_input: dict[str, object]) -> PredOutput:
    agent = PredAgent()
    try:
        return agent.invoke(user_input)
    except AgentInvocationError as e:
        raise e
