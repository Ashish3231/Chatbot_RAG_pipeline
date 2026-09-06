"""Step 3 of corrective RAG: the search failed, so ask a different question.

This is not condense.py. That one resolves a follow-up against the conversation
("and English?" -> "who teaches English in class 10-A?") and runs before
anything is searched. This one runs *after* a search came back with nothing
worth using, and its problem is different: the question is already standalone,
it simply does not share vocabulary with the document that answers it.

Users ask in the words they think in; documents are written in the words an
institution files things under. "Can I take time off for a funeral?" and
"bereavement leave entitlement" are the same question, and an embedding of the
first lands nowhere near a chunk written as the second. Rewriting is the step
that crosses that gap.
"""

from typing import Sequence

from app.config import env_str
from app.llm import ROUTER_MODEL, content_of, openai

REWRITE_MODEL = env_str("REWRITE_MODEL", ROUTER_MODEL)

REWRITE_PROMPT = """
  You rewrite a search query that failed to find anything useful.

  The question is fine as a question; it just does not match how the document
  is written. Rewrite it the way the document would put it: the formal or
  institutional term instead of the everyday one, the noun phrase instead of
  the sentence, and the specific words a heading or a clause would use.

  Rules:
  - Keep the same subject. Do not answer, narrow, broaden or invent details.
  - Drop the conversational scaffolding ("can you tell me", "I was wondering").
  - Keep any code, id or quoted phrase exactly as it was given.

  Reply with the rewritten query on one line and nothing else.

  Examples:
    "can I take time off for a funeral?"  ->  "bereavement leave entitlement"
    "who do I tell if I am sick?"         ->  "sick leave notification procedure"
    "what does WPS 365 say about pay?"    ->  "WPS 365 salary payment requirements"
"""


async def rewrite_query(question: str, tried: Sequence[str] = ()) -> str | None:
    """Rewrite a query that retrieval could not serve.

    ``tried`` is the queries already searched, so a second correction round does
    not spend a call arriving back at one of them.

    Returns the rewritten query, or None when there is no usable new one — the
    caller then stops correcting rather than searching the same thing twice.
    """
    try:
        content = f"Query: {question}"

        if len(tried) > 1:
            quoted = ", ".join(f'"{query}"' for query in tried)
            content += f"\nAlready tried, do not repeat: {quoted}"

        response = await openai.chat.completions.create(
            model=REWRITE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": content},
            ],
        )

        rewritten = content_of(response).strip("\"'")

        # A model that ignored the instruction and answered the question instead
        # would send a paragraph to the search. Length is the cheap tell.
        if not rewritten or len(rewritten) > len(question) + 120:
            return None

        # No new query, no second search: the same words return the same chunks.
        seen = {query.strip().lower() for query in tried}

        if rewritten.lower() in seen:
            return None

        print(f'Rewrote: "{question}" -> "{rewritten}"')

        return rewritten
    except Exception as error:
        # The correction round is best-effort. Without a rewrite there is simply
        # nothing better to search for, and the caller answers on what it has.
        print(f"Query rewrite failed, keeping the original: {error}")

        return None
