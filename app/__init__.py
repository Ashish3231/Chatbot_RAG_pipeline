"""The RAG chatbot package.

Importing this package loads .env first, so every module below can read its
settings at import time the way the Node original did.
"""

from app import config  # noqa: F401  (imported for its side effect: load_dotenv)

__all__ = ["config"]
