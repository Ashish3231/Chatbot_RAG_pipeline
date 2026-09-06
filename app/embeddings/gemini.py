"""Embeddings via the Gemini API, called over REST so there is no extra SDK to
install. Needs GEMINI_API_KEY."""

import os
from typing import Sequence

import httpx

from app.config import env_str

EMBEDDING_MODEL = env_str("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")

    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to .env (see .env.example).")

    return key


async def generate_embedding(text: str | Sequence[str]):
    """Embed one string, or a batch of strings.

    Returns list[float] for a string, list[list[float]] for a list.
    """
    is_batch = not isinstance(text, str)
    texts = list(text) if is_batch else [text]

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{API_BASE}/models/{EMBEDDING_MODEL}:batchEmbedContents",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": _get_api_key(),
            },
            json={
                "requests": [
                    {
                        "model": f"models/{EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": content}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                    }
                    for content in texts
                ]
            },
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini embedding failed ({response.status_code}): {response.text}"
            )

        data = response.json()

    embeddings = [entry["values"] for entry in data["embeddings"]]

    return embeddings if is_batch else embeddings[0]
