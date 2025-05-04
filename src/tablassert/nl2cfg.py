__author__ = "Skye Lane Goetz"


from adapt.tools.text.tokenizer import EnglishTokenizer
from adapt.engine import IntentDeterminationEngine  # , DomainIntentDeterminationEngine
from adapt.intent import IntentBuilder
from tablassert.io import get_root
from functools import lru_cache
from numpy import mean
import sys


class CustomTokenizer(EnglishTokenizer):

    def tokenize(self, text):
        tokens = super().tokenize(text)
        return " ".join([t for t in tokens if t.isalpha()])


KEYWORDS = get_root() + "/src/tablassert/nl2cfg_keywords/"
CUSTOM_TOKENIZER = CustomTokenizer()


@lru_cache(maxsize=8)
def tokenize_input(x: str) -> str:
    tokenizer = CUSTOM_TOKENIZER
    return tokenizer.tokenize(str(x))


@lru_cache(maxsize=None)
def get_keywords(text_file: str) -> list[str]:
    with open(text_file, "r") as f:
        return [line.rstrip("\n") for line in f]


@lru_cache(maxsize=None)
def template_section_parsers() -> IntentDeterminationEngine:
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
    engine.register_intent_parser(sections_intent)
    engine.register_intent_parser(template_intent)
    return engine


def template_section_split(user_input: str) -> bool:
    """
    returns TRUE if "Sections" and FALSE if "Template" and NONE if else
    """
    engine = template_section_parsers()
    user_input: str = tokenize_input(user_input)
    intents: list[dict[str, object]] = list(engine.determine_intent(user_input))
    if intents:
        results = [
            1 if intent["intent_type"] == "Sections" else 0 for intent in intents
        ]
        if mean(results) >= 0.5:
            return True
    return False


@lru_cache(maxsize=None)
def attribute_parser() -> IntentDeterminationEngine:
    engine = IntentDeterminationEngine()
    root = KEYWORDS
    attributes: str = root + "attributes_keywords.txt"
    for kw in get_keywords(attributes):
        engine.register_entity(kw, "attributes")
    attributes_intent = IntentBuilder("Attributes").require("attributes").build()
    engine.register_intent_parser(attributes_intent)
    return engine


def attributes_split(user_input: str) -> bool:
    """
    returns TRUE if "Atrributes" and FALSE if else
    """
    engine = attribute_parser()
    user_input: str = tokenize_input(user_input)
    intents: list[dict[str, object]] = list(engine.determine_intent(user_input))
    if intents:
        return True
    return False


if __name__ == "__main__":
    user_input: str = " ".join(sys.argv[1:])
    print("\n")
    print(user_input)
    print("\n")
    print(
        f"Template == False, Section == True --> {template_section_split(user_input)}"
    )
    print(f"Attributes == True --> {attributes_split(user_input)}")
    print("\n")
