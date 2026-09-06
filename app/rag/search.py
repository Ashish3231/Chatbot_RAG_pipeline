"""The three ways this app can search the documents, and the shaping of what
comes back. Which one runs is decided in adaptive.py; this file only knows how
to run them.
"""

import asyncio
import re
from typing import Any, Sequence

from app.config import env_float, env_int
from app.embeddings import EMBEDDER, generate_embedding
from app.llm import ROUTER_MODEL, content_of, openai
from app.store.pgvector import keyword_chunks, query_chunks

Match = dict[str, Any]

# A chunk this far below the best match is noise, not context. Relative, so it
# works whatever similarity scores the embedding model happens to produce.
SCORE_RATIO = env_float("ADAPTIVE_SCORE_RATIO", 0.6)

# ...and an absolute floor, for when nothing matched well and the best match is
# itself irrelevant. Set it to 0 to keep whatever the search returns.
MIN_SCORE = env_float("ADAPTIVE_MIN_SCORE", 0.2)

# How many rephrasings a complex question is searched with, the original
# included.
QUERY_VARIANTS = env_int("ADAPTIVE_QUERY_VARIANTS", 3)


# ---------------------------------------------------------------------------
# The searches
# ---------------------------------------------------------------------------


async def search_by_vector(question: str, top_k: int) -> list[Match]:
    """The plain search: embed the question, ask the store for the nearest chunks."""
    embedding = await generate_embedding(question)

    return await query_chunks(EMBEDDER, embedding, top_k)


async def vector_search(question: str, top_k: int) -> list[Match]:
    """Strategy 1 — meaning only, with the weak tail trimmed off. What most
    questions want."""
    return trim_by_score(await search_by_vector(question, top_k))


async def hybrid_search(question: str, top_k: int) -> list[Match]:
    """Strategy 2 — meaning and exact words together.

    Embeddings are good at meaning and bad at exact strings: "WPS 365" and
    "WPS 366" sit almost on top of each other in vector space. When the user
    wants those characters found, the keyword half is what finds them.

    The two halves run at the same time, so this costs about what the slower of
    them costs rather than the sum.
    """
    embedding = await generate_embedding(question)

    by_vector, by_keyword = await asyncio.gather(
        query_chunks(EMBEDDER, embedding, top_k),
        keyword_chunks(EMBEDDER, question, top_k),
    )

    # No score trimming here: the two lists are scored on scales that do not
    # compare, and fusion has already put the agreed-on chunks at the top.
    return fuse([by_vector, by_keyword], top_k, ["vector", "keyword"])


async def multi_query_search(question: str, top_k: int) -> list[Match]:
    """Strategy 3 — ask the same thing several ways.

    A broad question rarely shares wording with every passage that answers it.
    Rephrasing costs one small LLM call; the rephrasings are then embedded in a
    single batched call, so the extra searches are nearly free.
    """
    queries = await expand_query(question)

    embeddings = await generate_embedding(queries)

    lists = await asyncio.gather(
        *(query_chunks(EMBEDDER, embedding, top_k) for embedding in embeddings)
    )

    # Every list came from the same embedding model, so these scores do compare
    # and the weak tail can go.
    return trim_by_score(fuse(list(lists), top_k))


EXPAND_PROMPT = """
  You write alternative search queries for a document search.

  Given a question, write different ways of asking the same thing, using the
  words a document might use instead of the words the user chose. Keep each on
  its own line, with no numbering and no other text.
"""


async def expand_query(question: str) -> list[str]:
    """The original question plus a couple of rephrasings of it."""
    try:
        response = await openai.chat.completions.create(
            model=ROUTER_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": EXPAND_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\nWrite {QUERY_VARIANTS - 1} of them.",
                },
            ],
        )

        variants = [
            stripped
            for line in content_of(response).split("\n")
            if (stripped := re.sub(r"^[-*\d.\s]+", "", line).strip())
        ][: QUERY_VARIANTS - 1]

        return [question, *variants]
    except Exception as error:
        # A failed expansion is not a failed search: fall back to searching with
        # the question the user actually asked.
        print(f"Query expansion failed, searching the original only: {error}")

        return [question]


# ---------------------------------------------------------------------------
# Ranking and shaping
# ---------------------------------------------------------------------------


def trim_by_score(matches: list[Match]) -> list[Match]:
    """Drop matches that are much worse than the best one.

    A search always returns top_k rows, whether or not the document has top_k
    relevant passages — the tail is padding that costs tokens and gives the
    model something irrelevant to be distracted by.
    """
    if not matches:
        return matches

    best = matches[0]["score"]

    # Nothing in the store is close to the question. Returning no context is the
    # honest outcome: the system prompt then has the model say so, rather than
    # dressing up the nearest unrelated passage.
    if best < MIN_SCORE:
        return []

    return [match for match in matches if match["score"] >= best * SCORE_RATIO]


def key_of(match: Match) -> str:
    """The id a chunk is deduplicated by when two searches both return it."""
    return f'{match["metadata"]["source"]}#{match["metadata"]["chunkIndex"]}'


def fuse(lists: Sequence[list[Match]], top_k: int, labels: Sequence[str] = ()) -> list[Match]:
    """Merge several ranked lists into one, by reciprocal rank fusion.

    Vector similarity and text rank are different numbers on different scales,
    so they cannot be compared or added. Their *positions* can be: a chunk near
    the top of both lists is a better answer than one that only tops either.
    Each list contributes 1/(K + rank), so a high rank helps and a long tail
    barely moves anything.
    """
    K = 60  # the usual constant: enough to flatten the tail of a list

    scores: dict[str, float] = {}
    best: dict[str, Match] = {}

    for list_index, ranked in enumerate(lists):
        for rank, match in enumerate(ranked):
            key = key_of(match)

            scores[key] = scores.get(key, 0) + 1 / (K + rank + 1)

            # Which searches turned this chunk up. Only worth recording when the
            # lists are different kinds of search: a chunk both of them agree on
            # means something different from one only the keyword half found.
            # The rephrasings of a multi-query search are all the same kind, so
            # it passes no labels and its sources carry no `via`.
            via = None

            if list_index < len(labels):
                via = set(best.get(key, {}).get("via") or ())
                via.add(labels[list_index])

            # Keep the copy with the best score, so what is reported back as
            # `score` stays the most flattering true similarity we saw.
            if key not in best or match["score"] > best[key]["score"]:
                best[key] = {**match, **({"via": via} if via else {})}
            elif via:
                best[key]["via"] = via

    ordered = sorted(scores.items(), key=lambda entry: entry[1], reverse=True)[:top_k]

    return [best[key] for key, _ in ordered]


def to_context(matches: Sequence[Match]) -> str:
    """The retrieved passages, as one block of text for the prompt."""
    return "\n\n---\n\n".join(match["text"] for match in matches)


def to_sources(matches: Sequence[Match]) -> list[dict[str, Any]]:
    """Which chunks the answer was built from. Cheap to carry, and the fastest
    way to tell a grounded answer from an invented one."""
    sources = []

    for match in matches:
        source = {
            "source": match["metadata"]["source"],
            "chunk": match["metadata"]["chunkIndex"],
            "score": round(float(match["score"]), 3),
        }

        # Only a fused search has two kinds of score to tell apart.
        if match.get("via"):
            source["via"] = "+".join(sorted(match["via"]))

        sources.append(source)

    return sources
