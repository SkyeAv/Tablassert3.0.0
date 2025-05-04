__author__ = "Skye Lane Goetz"


from adapt.tools.text.tokenizer import EnglishTokenizer
from adapt.engine import IntentDeterminationEngine  # , DomainIntentDeterminationEngine
from adapt.intent import IntentBuilder
from tablassert.io import get_root
from functools import lru_cache
from collections import Counter
from numpy import mean
import sys
import re


class CustomTokenizer(EnglishTokenizer):

    def tokenize(self, text):
        url_pattern = re.compile(r"https?://[^\s]+")
        urls = url_pattern.findall(text)
        cleaned_text = url_pattern.sub("", text)
        tokens = super().tokenize(cleaned_text)
        clean_tokens = [t for t in tokens if t.isalpha()]
        return " ".join(clean_tokens + urls)


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
def attributes_parser() -> IntentDeterminationEngine:
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
    engine = attributes_parser()
    user_input: str = tokenize_input(user_input)
    intents: list[dict[str, object]] = list(engine.determine_intent(user_input))
    if intents:
        return True
    return False


@lru_cache(maxsize=None)
def filetype_parser() -> IntentDeterminationEngine:
    engine = IntentDeterminationEngine()
    root = KEYWORDS
    text_based_image: str = root + "text_based_image_keywords.txt"
    excel_spreadsheet: str = root + "excel_spreadsheet_keywords.txt"
    delimited_file: str = root + "delimited_file_keywords.txt"
    for kw in get_keywords(text_based_image):
        engine.register_entity(kw, "text_based_image")
    for kw in get_keywords(excel_spreadsheet):
        engine.register_entity(kw, "excel_spreadsheet")
    for kw in get_keywords(delimited_file):
        engine.register_entity(kw, "delimited_file")
    filetype_intent = (
        IntentBuilder("Filetype")
        .optionally("text_based_image")
        .optionally("excel_spreadsheet")
        .optionally("delimited_file")
        .build()
    )
    engine.register_intent_parser(filetype_intent)
    return engine


def get_mode(x: list[int]) -> int:
    counter = Counter(x)
    return max(counter.items(), key=lambda x: x[1])[0]


@lru_cache(maxsize=None)
def filetype_encoder(x: str) -> int:
    match str(x):
        case "text_based_image":
            return 0
        case "excel_spreadsheet":
            return 1
        case "delimited_file":
            return 2
        case _:
            raise ValueError(f"Filetype {x} cannot be encoded")


def filetype_split(user_input: str) -> int:
    """
    returns 0 if "TextBasedImage" and 1 if "ExcelSpreadSheet" and 2 if "DelimitedFile"
    """
    engine = filetype_parser()
    user_input: str = tokenize_input(user_input)
    intents: list[dict[str, object]] = list(engine.determine_intent(user_input))
    if intents:
        results = [filetype_encoder(list(intent.keys())[1]) for intent in intents]
        return get_mode(results)


@lru_cache(maxsize=None)
def location_parser() -> IntentDeterminationEngine:
    engine = IntentDeterminationEngine()
    root = KEYWORDS
    extensions: str = root + "extensions_keywords.txt"
    for kw in get_keywords(extensions):
        engine.register_entity(kw, "ext")
    # engine.register_regex_entity(r"(?P<download>https?://[^\s]+)")
    extensions_intent = (
        IntentBuilder("Location").require("ext").build()
    )  # .optionally("download").build()
    engine.register_intent_parser(extensions_intent)
    return engine


def get_location(user_input: str) -> tuple[str, str]:
    engine = location_parser()
    user_input: str = tokenize_input(user_input)
    intents: list[dict[str, object]] = list(engine.determine_intent(user_input))
    print(intents)
    if intents:
        best_intent = max(intents, key=lambda i: i.get("confidence", 0))
        # download = best_intent.get("download")
        ext = best_intent.get("ext")
        url_pattern = re.compile(r"https?://[^\s]+")
        downloads = list(url_pattern.findall(user_input))
        if downloads:
            return downloads[0], ext


if __name__ == "__main__":
    user_input: str = " ".join(sys.argv[1:])
    print("\n")
    print("INPUT:", user_input)
    print("TOKENIZED:", tokenize_input(user_input))
    print("\n")
    print(
        f"Template == False, Section == True --> {template_section_split(user_input)}"
    )
    print(f"Attributes == True --> {attributes_split(user_input)}")
    print(
        f"Filetype: :0 = text based image, :1 = excel, :2 = delimited file --> {filetype_split(user_input)}"
    )
    print(f"download, ext --> {get_location(user_input)}")
    print("\n")
