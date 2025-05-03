__author__ = "Skye Lane Goetz"


from adapt.engine import IntentDeterminationEngine  # , DomainIntentDeterminationEngine
from adapt.intent import IntentBuilder
from tablassert.io import get_root
from functools import lru_cache
from numpy import mean
import sys


KEYWORDS = get_root() + "/src/tablassert/nl2cfg_keywords/"


@lru_cache(maxsize=None)
def get_keywords(text_file: str) -> list[str]:
    with open(text_file, "r") as f:
        return [line.rstrip("\n") for line in f]


@lru_cache(maxsize=None)
def template_section_parsers() -> (
    tuple[IntentDeterminationEngine, IntentDeterminationEngine]
):
    engine = IntentDeterminationEngine()
    root = KEYWORDS
    sections: str = root + "sections_keywords.txt"
    template: str = root + "template_keywords.txt"
    for kw in get_keywords(sections):
        engine.register_entity(kw, "sections")
    for kw in get_keywords(template):
        engine.register_entity(kw, "template")
    sections_intent = IntentBuilder("Sections").require("sections").build()
    template_intent = IntentBuilder("Template").require("template").build()
    return (sections_intent, template_intent)


def template_section_split(user_input: str) -> bool:
    """
    returns TRUE if "Sections" and FALSE if "Template"
    """
    engine = IntentDeterminationEngine()
    sections_intent, template_intent = template_section_parsers()
    engine.register_intent_parser(sections_intent)
    engine.register_intent_parser(template_intent)
    intents: list[dict[str, object]] = engine.determine_intent(user_input)
    if intents:
        results = [
            1 if intent["intent_type"] == "Sections" else 0 for intent in intents
        ]
        if mean(results) >= 0.5:
            return True
        else:
            return False


if __name__ == "__main__":
    user_input: str = " ".join(sys.argv[1:])
    print(template_section_split(user_input))
