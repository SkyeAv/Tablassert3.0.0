from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import hashlib
import spacy
import re


stop_words = set(stopwords.words("english"))
lemmatizer = spacy.load("en_core_web_sm")


def hash_it(thing_to_hash: str) -> str:
    thing_to_hash = nonword_regex(thing_to_hash)
    return hashlib.sha1(thing_to_hash.encode("utf-8")).hexdigest()


def nonword_regex(thing_to_regex: str) -> str:
    return re.sub(r"\W+", "", str(thing_to_regex)).lower()


def get_tokens(sentence: str) -> list:
    return word_tokenize(sentence)


def tokenize_it(table_value: str) -> str:
    """
    Tokenizes a given table value into a list of strings, using NLTK's
    word_tokenize function. The result is sorted alphabetically.
    """
    try:
        return " ".join(sorted(get_tokens(table_value))).lower()
    except TypeError:
        return ""


def lemmatize_it(table_value: str) -> str:
    tokens = lemmatizer(table_value)
    lemmatized_tokens = [token.lemma_ for token in tokens]
    return " ".join(lemmatized_tokens)


def remove_stopwords(table_value) -> str:
    tokens = get_tokens(table_value)
    cleaned_text = [
            token.strip() for token in tokens
            if token.lower() not in stop_words]
    return " ".join(cleaned_text)
