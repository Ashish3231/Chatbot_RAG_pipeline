"""Local embeddings. No API key, no network after the first run: the model is
downloaded once into the local cache.

The Node original used ``@huggingface/transformers``, which runs the model on
ONNX. The closest thing on this side is ``fastembed``, which is also ONNX and
also downloads a ready-made model. ``sentence-transformers`` (Torch) is the
fallback, so whichever one installs on your Python is the one that runs.
"""

import asyncio
import math
from typing import Sequence

from app.config import env_str

# The Node original named its models the way the JS ONNX ports are published
# ("Xenova/..."); the Python runtimes know the same weights under their original
# names. Mapping them means an .env copied straight over from that project
# works, and the name recorded next to the vectors stays the canonical one.
ALIASES = {
    "Xenova/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "Xenova/all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
    "Xenova/bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "Xenova/bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
}

# Small, fast sentence-transformer. 384 dimensions.
_configured = env_str("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMBEDDING_MODEL = ALIASES.get(_configured, _configured)

if EMBEDDING_MODEL != _configured:
    print(f"Embedding model {_configured} -> {EMBEDDING_MODEL} (the same weights)")

# Cache the task, not the result, so concurrent callers share one load.
_loading: asyncio.Task | None = None
_lock = asyncio.Lock()


def _load_encoder():
    """Build the function that turns a list of strings into a list of vectors.

    Blocking, and slow the first time — it downloads the model — so it is only
    ever called inside a worker thread.
    """
    try:
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name=EMBEDDING_MODEL)

        def encode(texts: list[str]) -> list[list[float]]:
            return [list(map(float, vector)) for vector in model.embed(texts)]

        return encode
    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)

        def encode(texts: list[str]) -> list[list[float]]:
            vectors = model.encode(texts, normalize_embeddings=True)

            return [list(map(float, vector)) for vector in vectors]

        return encode
    except ImportError as error:
        raise RuntimeError(
            "The huggingface embedding provider needs a local model runtime. "
            "Install one of:\n"
            "    pip install fastembed            (ONNX, light)\n"
            "    pip install sentence-transformers (Torch, heavier)\n"
            "Or set EMBEDDING_PROVIDER to openai, gemini or ollama in .env."
        ) from error


async def _get_encoder():
    global _loading

    async with _lock:
        if _loading is None:
            print(f"Loading local embedding model ({EMBEDDING_MODEL})...")

            _loading = asyncio.create_task(asyncio.to_thread(_load_encoder))

    return await _loading


def _normalise(vector: list[float]) -> list[float]:
    """Cosine similarity in the store assumes unit vectors. Both runtimes above
    normalise already; doing it again is cheap and makes that not matter."""
    magnitude = math.sqrt(sum(value * value for value in vector))

    return vector if magnitude == 0 else [value / magnitude for value in vector]


async def generate_embedding(text: str | Sequence[str]):
    """Embed one string, or a batch of strings.

    Returns list[float] for a string, list[list[float]] for a list.
    """
    encode = await _get_encoder()

    is_batch = not isinstance(text, str)
    texts = list(text) if is_batch else [text]

    # The encoder is CPU-bound and blocking; off the event loop it goes, so the
    # server can still answer other requests while a batch is embedding.
    embeddings = await asyncio.to_thread(encode, texts)

    embeddings = [_normalise(vector) for vector in embeddings]

    return embeddings if is_batch else embeddings[0]
