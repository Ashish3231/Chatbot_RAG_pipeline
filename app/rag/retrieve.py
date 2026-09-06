"""The fixed pipeline, kept for comparison: always rewrite, always search,
always the same number of chunks. ADAPTIVE_RAG=false serves /chat from here
instead, which is the way to see what the routing is buying you.
"""

from typing import Any, Sequence

from app.config import env_int
from app.rag.condense import condense_question
from app.rag.search import search_by_vector, to_context, to_sources

# Three chunks is enough context for a focused question without burying the
# answer in near-misses.
TOP_K = env_int("TOP_K", 3)


async def fixed_retrieve(history: Sequence[dict[str, Any]], question: str) -> dict[str, Any]:
    """Same return shape as adaptive_retrieve(), minus the routing decisions."""
    standalone_question = await condense_question(history, question)

    matches = await search_by_vector(standalone_question, TOP_K)

    print(f'Retrieved {len(matches)} chunks for: "{standalone_question}"')

    return {
        "context": to_context(matches),
        "sources": to_sources(matches),
        "matches": matches,
        "standaloneQuestion": standalone_question,
    }
