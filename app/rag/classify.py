"""Step 1 of adaptive RAG: what kind of question is this?

A fixed pipeline does the same work for every message: rewrite it, embed it,
search, and stuff three chunks into the prompt. But "hi" needs no documents at
all, "what is the WPS-365 leave code?" needs the exact wording found, and
"compare the leave rules for staff and students" needs several passages from
different parts of the document.

Classifying first is what lets the rest of the pipeline spend accordingly. The
classifier itself has to be cheap or it eats the saving it makes, so:

  1. plain rules answer the obvious cases for free,
  2. only what the rules cannot place costs one small LLM call,
  3. and that answer is cached, so a repeated question is free again.
"""

import re
from typing import NamedTuple

from app.config import env_int
from app.llm import ROUTER_MODEL, content_of, openai

# The five kinds of question the pipeline knows how to serve.
QUERY_TYPES = [
    "chitchat",  #       "hi", "thanks" — no documents involved
    "conversational",  # about the user or this chat — the history already has it
    "factual",  #        one specific fact, findable in one passage
    "keyword",  #        an exact code, id or quoted phrase to match verbatim
    "complex",  #        several facts, a comparison, or a summary
]

# Ordered: the first pattern that matches wins, so the narrow rules come first.
RULES = [
    # A whole message that is nothing but a greeting or an acknowledgement.
    (
        re.compile(
            r"^\s*(hi|hey|hello|yo|thanks|thank you|thx|ok|okay|cool|bye|"
            r"good (morning|afternoon|evening|night))\b[\s!.?]*$",
            re.IGNORECASE,
        ),
        "chitchat",
    ),
    # Explicitly about the user or the conversation, which the history answers
    # and the documents cannot.
    (
        re.compile(
            r"\b(my name|who am i|what did i (just )?(say|ask)|as i said|earlier i)\b",
            re.IGNORECASE,
        ),
        "conversational",
    ),
    # A quoted phrase, or a code-shaped token like "WPS 365" or "ABC-12": the
    # user wants those characters found, not something semantically nearby.
    (re.compile(r'"[^"]{3,}"|\b[A-Z]{2,}[-_ ]?\d+\b'), "keyword"),
    # Asks for more than one thing, or for something assembled out of several.
    (
        re.compile(
            r"\b(compare|comparison|difference between|summar(y|ise|ize)|list all|"
            r"all the|both|pros and cons|step by step|walk me through)\b",
            re.IGNORECASE,
        ),
        "complex",
    ),
]


def classify_by_rules(question: str) -> str | None:
    """Everything the rules can decide, for free. Returns None when unsure."""
    for pattern, query_type in RULES:
        if pattern.search(question):
            return query_type

    return None


LLM_PROMPT = """
  You label a user's message with exactly one of these types:

  chitchat        a greeting, thanks or small talk. Needs no documents.
  conversational  about the user or about this chat itself ("what's my name?").
  factual         one specific fact to look up in the documents.
  keyword         asks for an exact code, id, name or quoted phrase.
  complex         needs several facts: a comparison, a summary, or a list.

  Reply with the single label and nothing else.
"""

# When even the LLM gives an answer we do not recognise, retrieving a little is
# the safe mistake: an unused passage costs tokens, a missing one costs the
# answer.
FALLBACK = "factual"

# Classifications are keyed by the question text, so the same question asked
# twice is only ever paid for once. Bounded, so a busy day cannot grow it
# without limit; the oldest entry goes first.
MAX_CACHE = env_int("CLASSIFY_CACHE", 500)

_cache: dict[str, str] = {}


def _remember(key: str, query_type: str) -> None:
    _cache[key] = query_type

    if len(_cache) > MAX_CACHE:
        del _cache[next(iter(_cache))]


class Classification(NamedTuple):
    type: str
    # Who decided: 'cache', 'rules', 'llm' or 'fallback'. Worth logging, because
    # it is how you see how often the free paths spared you a call.
    by: str


async def classify_query(question: str) -> Classification:
    """What kind of question is this?

    Cache, then rules, then — only if both come up empty — one small LLM call.
    """
    key = question.strip().lower()

    if key in _cache:
        return Classification(_cache[key], "cache")

    by_rules = classify_by_rules(question)

    if by_rules:
        _remember(key, by_rules)

        return Classification(by_rules, "rules")

    try:
        response = await openai.chat.completions.create(
            model=ROUTER_MODEL,
            # A label, not an opinion: no room for creativity, and no room to
            # ramble either — one word is the whole expected answer.
            temperature=0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": question},
            ],
        )

        label = content_of(response).lower()

        if label in QUERY_TYPES:
            _remember(key, label)

            return Classification(label, "llm")
    except Exception as error:
        # Routing is an optimisation, never a reason to fail the request: an
        # unclassified question is simply served the safe, default way.
        print(f"Classification failed, using default route: {error}")

    return Classification(FALLBACK, "fallback")
