"""Adaptive RAG: classify the question first, then decide how much work to do.

A fixed pipeline does the same thing for every message — rewrite it, embed it,
search, put three chunks in the prompt. But "hi" needs no documents at all, and
"compare the two teachers and list everything" needs more than three chunks
from more than one place.

Four decisions come out of the question's type, and each is a place the fixed
pipeline was overspending or underserving:

  retrieval routing   whether to search at all. Chit-chat and questions about
                      the conversation skip the embedding call, the database
                      round trip and the context tokens entirely.
  dynamic top_k       how many chunks to pull, and how many of those are worth
                      keeping once their scores are in.
  search strategy     which search to run: meaning, exact words, or several
                      rephrasings. See search.py.
  condensing          whether to spend an LLM call rewriting the follow-up.
                      Pointless if nothing is going to be searched.
"""

import time
from typing import Any, Sequence

from app.config import env_int
from app.rag.classify import classify_query
from app.rag.condense import condense_question
from app.rag.search import (
    hybrid_search,
    multi_query_search,
    to_context,
    to_sources,
    vector_search,
)

# ---------------------------------------------------------------------------
# The routing table
# ---------------------------------------------------------------------------
# One row per query type, and the whole of the policy. This is the first place
# to look when an answer came back with too little or too much context.

PLANS = {
    "chitchat": {"retrieve": False, "topK": 0, "strategy": "none", "condense": False},
    "conversational": {"retrieve": False, "topK": 0, "strategy": "none", "condense": False},
    "factual": {"retrieve": True, "topK": 3, "strategy": "vector", "condense": True},
    "keyword": {"retrieve": True, "topK": 5, "strategy": "hybrid", "condense": True},
    "complex": {"retrieve": True, "topK": 8, "strategy": "multi_query", "condense": True},
}

SEARCHES = {
    "vector": vector_search,
    "hybrid": hybrid_search,
    "multi_query": multi_query_search,
}

# Whatever the plan and the adjustment below work out to, the result stays
# inside these bounds — a runaway top_k is paid for on every following turn too,
# because the answer goes into the history.
MIN_TOP_K = env_int("ADAPTIVE_MIN_TOP_K", 2)
MAX_TOP_K = env_int("ADAPTIVE_MAX_TOP_K", 10)


def dynamic_top_k(base: int, question: str) -> int:
    """The plan's top_k, nudged by how much the question is actually asking for.

    The type sets the ballpark; the length distinguishes "who teaches maths?"
    from a question with three clauses in it.
    """
    words = len(question.strip().split())

    top_k = base

    if words > 25:
        top_k += 2  # several clauses, likely several answers to find
    elif words < 6:
        top_k -= 1  # a short, pointed question: one good chunk usually does it

    return min(MAX_TOP_K, max(MIN_TOP_K, top_k))


async def adaptive_retrieve(history: Sequence[dict[str, Any]], question: str) -> dict[str, Any]:
    """Classify, route, search.

    Returns the context and its sources, plus a `routing` object saying what was
    decided and what it cost — the same shape fixed_retrieve() returns, so the
    endpoint can await either one.
    """
    started_at = time.monotonic()

    query_type, by = await classify_query(question)

    plan = PLANS.get(query_type, PLANS["factual"])

    # Route 1: no documents needed. This is the whole saving — no rewrite call,
    # no embedding call, no database query, and no context in the prompt.
    if not plan["retrieve"]:
        print(f"Route: {query_type} ({by}) — no retrieval")

        return {
            "context": "",
            "sources": [],
            "matches": [],
            "routing": {
                "type": query_type,
                "retrieved": False,
                "topK": plan["topK"],  # 0 — nothing was asked of the store
                "strategy": plan["strategy"],  # 'none'
                "classifiedBy": by,
                "ms": _elapsed_ms(started_at),
            },
        }

    # Route 2: retrieve. Only now is it worth rewriting a follow-up into a
    # standalone question, because only a search cares about the wording.
    standalone_question = (
        await condense_question(history, question) if plan["condense"] else question
    )

    top_k = dynamic_top_k(plan["topK"], standalone_question)

    search = SEARCHES[plan["strategy"]]

    matches = await search(standalone_question, top_k)

    routing = {
        "type": query_type,
        "retrieved": True,
        "topK": top_k,
        "strategy": plan["strategy"],
        "classifiedBy": by,
        # What was asked for against what survived the relevance filter: a
        # steady gap means top_k is set higher than this corpus can use.
        "kept": len(matches),
        "ms": _elapsed_ms(started_at),
    }

    print(
        f'Route: {query_type} ({by}) — {plan["strategy"]}, topK {top_k}, '
        f'kept {len(matches)}, {routing["ms"]}ms'
    )

    return {
        "context": to_context(matches),
        "sources": to_sources(matches),
        # The rows behind that context. /chat never looks at them; the
        # corrective layer grades them, and re-derives its own context from
        # what survives.
        "matches": matches,
        "standaloneQuestion": standalone_question,
        "routing": routing,
    }


def _elapsed_ms(started_at: float) -> int:
    return round((time.monotonic() - started_at) * 1000)
