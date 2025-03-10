from nltk.tokenize import word_tokenize
import hashlib
import re


def hash_it(thing_to_hash: str) -> str:
    thing_to_hash = nonword_regex(thing_to_hash)
    return hashlib.sha1(thing_to_hash.encode("utf-8")).hexdigest()


def nonword_regex(thing_to_regex: str) -> str:
    return re.sub(r"\W+", "", str(thing_to_regex)).lower()


def tokenize_it(table_value: str) -> list:
    """
    Tokenizes a given table value into a list of strings, using NLTK's
    word_tokenize function. The result is sorted alphabetically.
    """
    try:
        return " ".join(sorted(word_tokenize(table_value))).lower()
    except TypeError:
        return ""
