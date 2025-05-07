__author__ = "Skye Lane Goetz"


from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from transformers import AutoModelForCausalLM, AutoTokenizer

# from tablassert.agents.wrapper import TransformersLLM
from pydantic import Field, conlist, constr
from tablassert.cfg import Provenance
from typing import Optional, Type
import torch


class ProvenanceOutput(BaseIOSchema):
    """
    Provenance for edges in tablassert
    """

    provenance: Provenance = Field(
        ...,
        description="A field denoting the provenance of edges in a Tablassert knowledge graph (i.e., where the knowledge came from, whos curating, from what organization, etc..)",
    )
    suggested_questions: conlist(
        item_type=constr(min_length=1, strip_whitespace=True), min_length=1
    ) = Field(..., description="A list of suggested follow up questions")


class TransformersLLM:

    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map=None
        ).to(self.device)

    def invoke(
        self, prompt: str, output_schema: Optional[Type[ProvenanceOutput]] = None
    ) -> str | ProvenanceOutput:
        if output_schema:
            return self.client(prompt, schema=output_schema)
        hyperparameters = {
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.3,  # Low randomness for structured generation
            "top_p": 0.6,  # Focuses on the most likely outputs, low creativity
            "repetition_penalty": 1.2,  # Prevents loops in structured blocks
        }
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, **hyperparameters)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are a structured configuration agent",
        "You generate strictly valid JSON that conforms to a complex schema for transforming tabular data into knowledge graphs",
        "This JSON will be parsed using Pydantic models, so it must respect all type, format, and validation rules",
    ],
    steps=[
        "1. Generate a JSON object containing the 'provenance' block",
        "2. The provenance block has three required fields: 'publication_id', 'curator', and 'org'.",
        "3. 'publication_id' must be a valid CURIE — it should begin with one of: 'PMID:', 'PMC:', or 'doi:'.",
        "4. 'curator' is the name of the person or agent curating the data (e.g., 'Dr. Smith', 'phi-2').",
        "5. 'org' is the name of the institution or organization associated with the curator (e.g., 'microsoft', 'NIH').",
    ],
    output_instructions=[
        "Only output a valid JSON object representing the 'provenance' block",
        "Do not include explanations or comments",
        "Do not leave any field blank or use placeholder values like 'string'",
        "Ensure that 'publication_id' matches the expected CURIE pattern (e.g., 'PMID:12345678')",
        "Strip whitespace and ensure values are syntactically valid",
    ],
)


llm_client = TransformersLLM()


agent = BaseAgent(
    config=BaseAgentConfig(
        client=llm_client.invoke,
        model="microsoft/phi-2",
        system_prompt_generator=system_prompt_generator,
        memory=AgentMemory(),
        output_schema=ProvenanceOutput,
    )
)
