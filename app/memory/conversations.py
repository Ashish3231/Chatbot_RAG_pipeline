"""Conversation memory: the part that lets /chat remember what was already said.

Kept in this process, keyed by session id. History is cheap to rebuild — the
client still has it — which is why it lives in a dict and not in Postgres the
way the embeddings do. Swap this module for a table or Redis when history has
to survive a restart or be shared by a second instance; the functions below are
the whole surface.
"""

import re
import time
import uuid
from typing import Any

from app.config import env_int

# How many question/answer pairs are carried into the next prompt. Older turns
# are dropped: they cost tokens on every request and rarely change the answer.
HISTORY_TURNS = env_int("HISTORY_TURNS", 5)

# A session is forgotten after this long without a message...
TTL_SECONDS = env_int("SESSION_TTL_MINUTES", 60) * 60

# ...and the least recently used are evicted past this many, so a busy day
# cannot grow the process without bound.
MAX_SESSIONS = env_int("MAX_SESSIONS", 1000)

# id -> { "messages": [{ "role", "content" }], "updatedAt": float seconds }
#
# A dict preserves insertion order, which is what the eviction below relies on
# the same way the JS original relied on Map ordering.
_sessions: dict[str, dict[str, Any]] = {}

_SESSION_ID = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


def create_session_id() -> str:
    return str(uuid.uuid4())


def is_session_id(value: Any) -> bool:
    """Whether a client-supplied id is one this server could have issued. An id
    is echoed back in a response header, and an unchecked string from a request
    body is a bad thing to put there."""
    return isinstance(value, str) and _SESSION_ID.match(value) is not None


def get_history(session_id: str) -> list[dict[str, str]]:
    """The recent turns of a conversation, oldest first, ready to splice into a
    prompt or to hand back to a client. Unknown or expired ids simply start a
    new conversation.

    A copy, not the stored list: append_turn mutates that one, so a caller
    holding the original would see both.
    """
    _expire()

    session = _sessions.get(session_id)

    print(f"get_history({session_id}) => {len(session['messages']) if session else 0} turns")

    return [dict(turn) for turn in (session["messages"] if session else [])]


def append_turn(session_id: str, question: str, answer: str) -> None:
    """Record one exchange, and start the session if this was its first message."""
    session = _sessions.get(session_id) or {"messages": [], "updatedAt": 0.0}

    session["messages"].append({"role": "user", "content": question})
    session["messages"].append({"role": "assistant", "content": answer})

    # Trim on write, so a long-running conversation holds a bounded list.
    if len(session["messages"]) > HISTORY_TURNS * 2:
        session["messages"] = session["messages"][-HISTORY_TURNS * 2 :]

    session["updatedAt"] = time.time()

    # Delete before setting: a dict iterates in insertion order, so re-inserting
    # on every write keeps the oldest entry first — which is what lets _expire()
    # and the eviction below stop at the first entry they want to keep.
    _sessions.pop(session_id, None)
    _sessions[session_id] = session

    while len(_sessions) > MAX_SESSIONS:
        del _sessions[next(iter(_sessions))]


def list_sessions() -> list[dict[str, Any]]:
    """Every conversation currently in memory, most recently used first.
    Metadata only — what was said stays in the session.

    Note what an id is: the whole of a conversation's security. Anyone holding
    one can continue that conversation and read its history back. Listing them
    is a debugging convenience, fine on localhost; put it behind auth, or take
    it out, before this serves anyone but you.
    """
    _expire()

    now = time.time()

    # Insertion order runs least-recently-used first, so reverse it.
    return [
        {
            "sessionId": session_id,
            # append_turn writes a question and an answer together, so this is exact.
            "turns": len(session["messages"]) / 2,
            "lastMessageAt": _iso(session["updatedAt"]),
            "idleSeconds": round(now - session["updatedAt"]),
        }
        for session_id, session in reversed(list(_sessions.items()))
    ]


def reset_session(session_id: str) -> bool:
    """Forget one conversation. Returns whether there was anything to forget."""
    return _sessions.pop(session_id, None) is not None


def _expire() -> None:
    """Drop idle sessions. Cheap: it stops at the first one still alive."""
    cutoff = time.time() - TTL_SECONDS

    for session_id, session in list(_sessions.items()):
        if session["updatedAt"] >= cutoff:
            break

        del _sessions[session_id]


def _iso(timestamp: float) -> str:
    """The same shape JavaScript's toISOString() produces."""
    from datetime import datetime, timezone

    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
