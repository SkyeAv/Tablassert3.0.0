__author__ = "Skye Lane Goetz"
__status__ = "Development"


from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from tablassert.io import get_root, read_lines
from pydantic import Field, constr
from functools import lru_cache
from openai import OpenAI
from re import split
import instructor


SYSTEM_PROMPTS: str = get_root() + "/src/tablassert/agents/system_prompts/"
KEYWORDS: str = get_root() + "/src/tablassert/agents/keywords/"


@lru_cache(maxsize=None)
def get_system_prompt(subdir: str, SCHEMA: str | None = None) -> dict[str, list[str]]:
    root = SYSTEM_PROMPTS + subdir + "/"
    background = root + "background.txt"
    steps = root + "steps.txt"
    output_instructions = root + "output_instructions.txt"
    if SCHEMA:
        system_prompt = {
            "background": [x.replace("SCHEMA", SCHEMA) for x in read_lines(background)],
            "steps": [x.replace("SCHEMA", SCHEMA) for x in read_lines(steps)],
            "output_instructions": [
                x.replace("SCHEMA", SCHEMA) for x in read_lines(output_instructions)
            ],
        }
    else:
        system_prompt = {
            "background": read_lines(background),
            "steps": read_lines(steps),
            "output_instructions": read_lines(output_instructions),
        }
    return SystemPromptGenerator(**system_prompt)


@lru_cache(maxsize=None)
def ollama_client() -> OpenAI:
    return instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",  # don't change... its constant across machines
            api_key="ollama",  # don't remove... this does nothing but is required
        ),
        mode=instructor.Mode.JSON,
    )


class UserInput(BaseIOSchema):
    """
    An unstructured text-based user chat message/input for LLMs to interpret into a strict JSON structure
    """

    chat_msg: constr(min_length=1, strip_whitespace=True) = Field(
        ..., description="An unprocessed string containing user messaging/input"
    )


def get_user_input() -> dict[str, str]:
    user_input: str = input("Enter a message: ")
    return {"chat_msg": user_input}


class AgentInvocationError(Exception):
    """
    Raised when an agent fails to process user input
    """

    pass


class LLMAgent:

    def __init__(
        self,
        llm_client: OpenAI,
        llm: str,
        system_prompt: SystemPromptGenerator,
        input_schema: BaseIOSchema,
        output_schema: BaseIOSchema,
    ):
        self.llm_client = llm_client
        self.llm = llm
        self.system_prompt = system_prompt
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.agent = BaseAgent(
            config=BaseAgentConfig(
                client=self.llm_client,
                model=self.llm,
                system_prompt_generator=self.system_prompt,
                memory=AgentMemory(),
                output_schema=self.output_schema,
            )
        )

    def invoke(self, user_input: dict[str, object]):
        try:
            parsed_input = self.input_schema(**user_input)
            output = self.agent.run(parsed_input)
            return output
        except Exception as e:
            raise AgentInvocationError(e)


class AdaptTokenizer:

    def invoke(self, text) -> str:
        tokens = [t for t in split(r"\W+", text) if t]
        return " ".join(tokens)
