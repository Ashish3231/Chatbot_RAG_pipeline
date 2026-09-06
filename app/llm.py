"""The chat model: one client, and the two models that use it."""

import os
import sys

from openai import AsyncOpenAI

from app.config import env_str

if not os.environ.get("OPENAI_API_KEY"):
    print("Missing OPENAI_API_KEY. Add it to .env (see .env.example).", file=sys.stderr)
    raise SystemExit(1)

# Async, because every call site awaits it: the server handles requests
# concurrently and a blocking client would serialise them.
openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# The model that answers the question.
CHAT_MODEL = env_str("CHAT_MODEL", "gpt-3.5-turbo")

# The model that classifies and rephrases questions. Those are short calls on
# short prompts, so the smallest model you have is the right one for them,
# whatever answers the question.
ROUTER_MODEL = env_str("ROUTER_MODEL", CHAT_MODEL)


def content_of(response) -> str:
    """The assistant's text, or '' — the equivalent of the JS optional chain
    ``response.choices[0]?.message?.content?.trim() ?? ''``."""
    if not response.choices:
        return ""

    return (response.choices[0].message.content or "").strip()
