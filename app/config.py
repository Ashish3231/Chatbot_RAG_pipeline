"""Loading .env, and nothing else.

It has to happen before anything else reads os.environ. Every module in this
package reads its settings at import time, the way the Node original read
``process.env``, so this module is imported first by ``app/__init__.py`` and
the rest of the package can assume the environment is already populated.

In production the platform usually injects env vars, so a missing file is fine.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# The project root: the directory holding .env, compose.yaml and the PDF.
ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Typed readers
# ---------------------------------------------------------------------------
# JavaScript coerces on the way in (`Number(process.env.TOP_K || 3)`), so these
# are the equivalents. Each one falls back to the default when the variable is
# unset or empty, and an unparseable value is a configuration error worth
# hearing about rather than silently defaulting.


def env_str(name: str, default: str) -> str:
    value = os.environ.get(name)

    return value if value else default


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)

    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)

    return float(value) if value else default


def env_bool(name: str, default: bool) -> bool:
    """`ADAPTIVE_RAG !== 'false'` — anything but the literal string is the default."""
    value = os.environ.get(name)

    if value is None or value == "":
        return default

    return value.strip().lower() not in {"false", "0", "no", "off"}
