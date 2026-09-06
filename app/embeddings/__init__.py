"""Picks the embedding provider, once, for everything that needs one.

Every module in this folder exports the same ``generate_embedding(text)`` that
takes a string or a list of strings, so swapping providers is one env var. The
import is done here and lazily, so the unused providers are never loaded (the
Hugging Face one in particular pulls in a whole ONNX or Torch runtime).
"""

import importlib
import sys

from app.config import env_str

PROVIDERS = {
    "huggingface": "app.embeddings.huggingface",
    "openai": "app.embeddings.openai_provider",
    "gemini": "app.embeddings.gemini",
    "ollama": "app.embeddings.ollama",
}

PROVIDER = env_str("EMBEDDING_PROVIDER", "huggingface")

if PROVIDER not in PROVIDERS:
    print(
        f'Unknown EMBEDDING_PROVIDER "{PROVIDER}". '
        f"Use one of: {', '.join(PROVIDERS)}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

_module = importlib.import_module(PROVIDERS[PROVIDER])

generate_embedding = _module.generate_embedding
EMBEDDING_MODEL: str = _module.EMBEDDING_MODEL


class Embedder:
    """Stored alongside the vectors so a later run can check it is reading
    vectors the current model can actually be compared against."""

    __slots__ = ("provider", "model")

    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model


EMBEDDER = Embedder(PROVIDER, EMBEDDING_MODEL)

print(f"Embeddings: {PROVIDER} ({EMBEDDING_MODEL})")

__all__ = ["generate_embedding", "EMBEDDING_MODEL", "EMBEDDER", "Embedder"]
