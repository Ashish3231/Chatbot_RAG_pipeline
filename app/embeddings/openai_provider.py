"""Embeddings via the OpenAI API. Needs OPENAI_API_KEY.

Named ``openai_provider`` rather than ``openai`` so it cannot be confused with
the SDK package it imports.
"""

import os
from typing import Sequence

from openai import AsyncOpenAI

from app.config import env_str

# text-embedding-3-small: 1536 dimensions, cheapest of the current models.
EMBEDDING_MODEL = env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client

    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY. Add it to .env (see .env.example).")

        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    return _client


async def generate_embedding(text: str | Sequence[str]):
    """Embed one string, or a batch of strings.

    Returns list[float] for a string, list[list[float]] for a list.
    """
    is_batch = not isinstance(text, str)
    texts = list(text) if is_batch else [text]

    # The API accepts a list, so a batch costs one round trip.
    response = await _get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)

    # Results come back unordered in theory; index tells us where each belongs.
    embeddings: list[list[float]] = [[] for _ in texts]

    for item in response.data:
        embeddings[item.index] = list(item.embedding)

    return embeddings if is_batch else embeddings[0]
