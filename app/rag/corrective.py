"""Corrective RAG: check the retrieval before trusting it, and retry when it is
not good enough.

Adaptive RAG decides how hard to search *before* searching. It has no idea
whether the search worked, because nothing looks at what came back — a top_k of
8 returns 8 rows against a corpus that holds nothing on the subject, and those 8
rows go into the prompt exactly as if they answered the question.

This layer closes that loop:

  1. retrieve            whatever the pipeline underneath does (adaptive or fixed)
  2. evaluate + grade    is each chunk relevant, partial or irrelevant?  (grade.py)
  3. verdict             correct / ambiguous / incorrect
  4. correct             rewrite the query (rewrite.py) and search again, with a
                         different strategy: hybrid first, then multi-query
  5. merge               keep what earned its place, from either round

A `correct` verdict short-circuits at step 3, which is the usual case, so the
cost of the loop on a working retrieval is one small grading call. The expensive
path only runs when the alternative was a wrong answer.

What it will not do is invent context. If both rounds come back irrelevant, the
prompt gets no context at all and the model says it does not know — having
*checked* that it does not, rather than having failed to notice.
"""

import time
from typing import Any, Awaitable, Callable, Sequence

from app.config import env_int, env_str
from app.rag.grade import grade_matches
from app.rag.rewrite import rewrite_query
from app.rag.search import Match, hybrid_search, key_of, multi_query_search, to_context, to_sources

# How many correction rounds a question may cost, at one rewrite call plus one
# search each. One is nearly always enough: the second round is for a query
# whose vocabulary was doubly wrong.
MAX_CORRECTIONS = env_int("CRAG_MAX_CORRECTIONS", 1)

# The searches a correction round falls back to, in order. Hybrid first, because
# a vector search that found nothing has already shown that meaning alone is not
# landing, and the keyword half is what catches a term the rewriter has just
# supplied. Multi-query second: broader, and a whole extra LLM call, so it is
# the last thing tried and not the first.
FALLBACK_STRATEGIES = [
    name.strip()
    for name in env_str("CRAG_FALLBACK_STRATEGIES", "hybrid,multi_query").split(",")
    if name.strip()
]

SEARCHES = {
    "hybrid": hybrid_search,
    "multi_query": multi_query_search,
}

# A fallback search asks for a few more rows than the first one did: it is
# running because the first pass came back thin, and the grader drops whatever
# does not deserve to be in the prompt anyway.
FALLBACK_TOP_K = env_int("CRAG_FALLBACK_TOP_K", 6)

# Two rounds of retrieval can produce more chunks than any answer needs. The cap
# is on what reaches the prompt, after grading and ordering, so what is dropped
# is always the least useful of what we have.
MAX_CHUNKS = env_int("CRAG_MAX_CHUNKS", 8)

# Relevant chunks come before partial ones, whatever search turned them up.
GRADE_ORDER = {"relevant": 0, "partial": 1, "irrelevant": 2}

Retriever = Callable[[Sequence[dict[str, Any]], str], Awaitable[dict[str, Any]]]


async def corrective_retrieve(
    history: Sequence[dict[str, Any]], question: str, base_retrieve: Retriever
) -> dict[str, Any]:
    """Retrieve, grade, and correct.

    ``base_retrieve`` is adaptive_retrieve or fixed_retrieve; whichever one runs,
    this wraps it. It supplies the first round of matches.

    Returns the same shape either of them returns — context, sources,
    standaloneQuestion, routing — plus:

      quality.retrieval   how good the context is, before the answer exists
      correction          what this layer did about it, and what that cost
    """
    started_at = time.monotonic()

    base = await base_retrieve(history, question)

    # Nothing was retrieved, by design — chit-chat, or a question the
    # conversation already answers. There is no retrieval to evaluate, and
    # grading an empty list would only bill a call to say so.
    if base.get("routing") and base["routing"].get("retrieved") is False:
        return {**base, "correction": {"applied": False, "reason": "no-retrieval"}}

    # Retrieval ran against the condensed question, so that is the question the
    # chunks have to be relevant *to*.
    query = base.get("standaloneQuestion") or question

    initial = await grade_matches(query, base.get("matches") or [])

    attempts = [
        {
            "query": query,
            "strategy": (base.get("routing") or {}).get("strategy", "vector"),
            "verdict": initial["verdict"],
            "found": len(base.get("matches") or []),
            "kept": len(initial["kept"]),
            "gradedBy": initial["by"],
        }
    ]

    kept = initial["kept"]
    verdict = initial["verdict"]

    # Rated the same way in both paths — over the chunks actually being sent, so
    # the number means the same thing whether or not a correction round ran.
    quality = _rate(kept, [initial])

    tried = [query]

    # The correction rounds. A `correct` verdict never enters this loop.
    round_index = 0

    while verdict != "correct" and round_index < MAX_CORRECTIONS:
        strategy = (
            FALLBACK_STRATEGIES[round_index]
            if round_index < len(FALLBACK_STRATEGIES)
            else (FALLBACK_STRATEGIES[-1] if FALLBACK_STRATEGIES else None)
        )

        search = SEARCHES.get(strategy) if strategy else None

        if not search:
            break

        rewritten = await rewrite_query(query, tried)

        # No new query means no new chunks — the same words return the same
        # rows, and a second search would be a round trip spent confirming that.
        if not rewritten:
            break

        tried.append(rewritten)

        found = await search(rewritten, FALLBACK_TOP_K)

        # Graded against the rewritten query, because that is what was searched —
        # and the rewrite is only a good one if what it found answers the
        # question the user actually asked, so the original is what the grader
        # is shown.
        retry = await grade_matches(query, found)

        attempts.append(
            {
                "query": rewritten,
                "strategy": strategy,
                "verdict": retry["verdict"],
                "found": len(found),
                "kept": len(retry["kept"]),
                "gradedBy": retry["by"],
            }
        )

        # Both rounds' keepers, deduplicated. The first round's chunks were
        # graded too, and a partial from it is worth more than nothing from
        # this one.
        kept = _merge(kept, retry["kept"])

        verdict = _better(verdict, retry["verdict"])

        # Quality is reported for the context that is actually being sent, so it
        # is recomputed over the merged set rather than carried over from a round.
        quality = _rate(kept, [initial, retry])

        round_index += 1

    context = kept[:MAX_CHUNKS]

    correction = {
        "applied": len(attempts) > 1,
        "verdict": verdict,
        "initialVerdict": initial["verdict"],
        "attempts": attempts,
        "ms": round((time.monotonic() - started_at) * 1000),
    }

    print(
        f'Corrective: {initial["verdict"]}'
        + (
            f" -> {verdict} after {len(attempts) - 1} correction(s)"
            if correction["applied"]
            else ""
        )
        + f', {len(context)} chunk(s) kept, score {quality["score"]}, {correction["ms"]}ms'
    )

    return {
        **base,
        # Rebuilt from what survived grading, not from what the search returned.
        # This is the whole point of the layer: an irrelevant chunk never reaches
        # the prompt, and never appears in `sources` as though it had.
        "context": to_context(context),
        "sources": to_sources(context),
        "matches": context,
        "quality": {
            "retrieval": {
                # 0 when nothing survived grading — a result, not a failure.
                "score": quality["score"],
                "verdict": verdict,
                # How much of the searching was worth paying for. A low
                # precision with a high score is a precise context bought with
                # a lot of rows.
                "precision": quality["precision"],
                "chunks": len(context),
                "retrieved": quality["retrieved"],
                **quality["counts"],
            }
        },
        "correction": correction,
    }


def _merge(first: Sequence[Match], second: Sequence[Match]) -> list[Match]:
    """Two rounds of keepers, as one list.

    Ordered by grade and then by the order they arrived, deliberately *not* by
    score: the first round's scores are cosine similarities and a hybrid round's
    are fused text ranks, so sorting the two together would be comparing numbers
    that mean different things. Position within a round is the comparable signal.
    """
    by_key: dict[str, dict[str, Any]] = {}

    for arrival, match in enumerate([*first, *second]):
        key = key_of(match)

        existing = by_key.get(key)

        # The same chunk found twice keeps its better grade — two searches
        # agreeing it is relevant is not a reason to demote it.
        if existing is None or GRADE_ORDER[match["grade"]] < GRADE_ORDER[existing["match"]["grade"]]:
            by_key[key] = {
                "match": match,
                "arrival": existing["arrival"] if existing else arrival,
            }

    ordered = sorted(
        by_key.values(), key=lambda entry: (GRADE_ORDER[entry["match"]["grade"]], entry["arrival"])
    )

    return [entry["match"] for entry in ordered]


def _better(a: str, b: str) -> str:
    """`correct` beats `ambiguous` beats `incorrect`."""
    rank = {"correct": 0, "ambiguous": 1, "incorrect": 2}

    return a if rank[a] <= rank[b] else b


def _rate(kept: Sequence[Match], rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Retrieval quality: two numbers, because they answer two different questions.

      score      how good the context being sent is — the weighted mean of its
                 grades. 1.0 is a prompt whose every passage answers the
                 question, 0 is a prompt with no context at all.
      precision  how much of the searching was worth paying for: the share of
                 retrieved rows that survived grading. A high score with a low
                 precision is a clean context bought with a lot of rows, which
                 is a working pipeline with top_k set too wide.
    """
    counts = {"relevant": 0, "partial": 0, "irrelevant": 0}

    for match in kept:
        counts[match["grade"]] += 1

    retrieved = sum(len(round_result["graded"]) for round_result in rounds)

    weighted = counts["relevant"] + counts["partial"] * 0.5

    return {
        "score": 0 if not kept else round(weighted / len(kept), 3),
        "precision": 0 if retrieved == 0 else round(len(kept) / retrieved, 3),
        "retrieved": retrieved,
        "counts": counts,
    }
