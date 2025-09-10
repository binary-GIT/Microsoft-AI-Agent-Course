import json
from langchain.tools import tool
from agent.llm import safe_invoke
import wikipedia
import requests

# -------------------------------
# Load dictionary.json once
# -------------------------------
with open("data/dictionary.json", "r") as f:
    DICTIONARY = json.load(f)


# -------------------------------
# Local Dictionary Lookup
# -------------------------------
@tool
def lookup_dictionary(word: str) -> str:
    """Look up a word in the local dictionary.json file."""
    word_lower = word.lower()
    return DICTIONARY.get(word_lower, "Not found in dictionary.")


# -------------------------------
# LLM-based Tools
# -------------------------------
@tool
def llm_define(word: str) -> str:
    prompt = f"Provide a simple dictionary-style definition of '{word}'."
    return safe_invoke(prompt)


@tool
def llm_synonyms(word: str) -> str:
    prompt = f"List 5 simple synonyms for the word '{word}'."
    return safe_invoke(prompt)


@tool
def llm_antonyms(word: str) -> str:
    prompt = f"List 5 simple antonyms for the word '{word}'."
    return safe_invoke(prompt)


@tool
def llm_examples(word: str) -> str:
    prompt = f"Write 3 simple example sentences using the word '{word}'."
    return safe_invoke(prompt)


@tool
def wiki_usage(word: str) -> str:
    try:
        return wikipedia.summary(word, sentences=2)
    except Exception:
        return "No Wikipedia info found."


@tool
def llm_event_info(word: str) -> str:
    prompt = f"Is '{word}' the name of an event? If yes, explain what it is and give the date (if known). If not, say 'Not an event'."
    return safe_invoke(prompt)


@tool
def llm_etymology(word: str) -> str:
    prompt = f"Explain the origin and etymology of the word '{word}' in simple terms."
    return safe_invoke(prompt)


# -------------------------------
# Advanced Tools
# -------------------------------

@tool
def llm_pronunciation(word: str) -> str:
    """Return the phonetic pronunciation (IPA) of the word."""
    prompt = f"Provide the phonetic pronunciation (IPA) and a simple pronunciation guide for the word '{word}'."
    return safe_invoke(prompt)


@tool
def llm_frequency(word: str) -> str:
    """Estimate how common or frequent this word is in English."""
    prompt = f"Explain how common the word '{word}' is in English, e.g., rare, medium, frequent, and its usage context."
    return safe_invoke(prompt)


@tool
def llm_translate(word: str) -> str:
    """Translate the word into French, Spanish, and German."""
    prompt = f"Translate the word '{word}' into French, Spanish, and German. Provide only the translations."
    return safe_invoke(prompt)
