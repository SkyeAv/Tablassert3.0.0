__author__ = "Skye Lane Goetz"


from tablassert.agents.toolkit import AdaptTokenizer, KEYWORDS
from adapt.engine import IntentDeterminationEngine
from adapt.intent import IntentBuilder
from tablassert.io import read_lines
from functools import lru_cache
import sys


@lru_cache(maxsize=None)
def ext_engine() -> IntentDeterminationEngine:
    engine = IntentDeterminationEngine()
    root: str = KEYWORDS
    keyword_file = KEYWORDS + "ext_keywords.txt"
    ext_keywords = read_lines(keyword_file)
    for kw in ext_keywords:
        engine.register_entity(kw, "extension")
    ext_parser = IntentBuilder("Ext").require("extension").build()
    engine.register_intent_parser(ext_parser)
    return engine


def run_ext_engine(user_input: str):
    engine = ext_engine()
    tokenizer = AdaptTokenizer()
    cleaned_input: str = tokenizer.invoke(user_input)
    intents: list[dict[str, object]] = list(engine.determine_intent(cleaned_input))
    if not intents:
        msg = "You must specify a supported file extension to use the [ConfigBuilder]"
        raise ValueError(msg)
    best_intent = max(intents, key=lambda i: i.get("confidence", 0))
    ext = best_intent.get("extension")
    return ext
