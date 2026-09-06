"""Embeddings from a locally running Ollama server.

Pull the model once:  ollama pull nomic-embed-text
"""

from typing import Sequence

import httpx

from app.config import env_str

EMBEDDING_MODEL = env_str("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

BASE_URL = env_str("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


async def generate_embedding(text: str | Sequence[str]):
    """Embed one string, or a batch of strings.

    Returns list[float] for a string, list[list[float]] for a list.
    """
    is_batch = not isinstance(text, str)
    texts = list(text) if is_batch else [text]

    # /api/embed (not the older /api/embeddings) takes a batch and returns
    # { embeddings: [[...]] } in the same order as the input.
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": texts},
            )
        except httpx.RequestError as error:
            raise RuntimeError(
                f"Ollama embedding failed: {error}. Is Ollama running at {BASE_URL}?"
            ) from error

        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama embedding failed ({response.status_code}): {response.text}. "
                f"Is Ollama running at {BASE_URL}?"
            )

        data = response.json()

    return data["embeddings"] if is_batch else data["embeddings"][0]
