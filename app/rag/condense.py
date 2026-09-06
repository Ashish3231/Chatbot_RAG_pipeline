"""Rewriting a follow-up into a question that stands on its own.

"And who teaches her English?" shares almost no words with the chunk that
answers it, and "how many are there?" shares none at all. So retrieval runs
against a rewritten version of the question. This one step is most of what
separates a conversation from a series of unrelated questions.
"""

from typing import Any, Sequence

from app.config import env_int
from app.llm import CHAT_MODEL, content_of, openai

CONDENSE_PROMPT = """
  You rewrite a follow-up question so that it can be understood on its own.

  Most questions need no rewriting. Change the question ONLY where it cannot be
  understood without the conversation: a pronoun with no antecedent ("who is
  she?"), an ellipsis ("and English?"), or a back-reference ("that class", "the
  same teacher"). Resolve exactly those and change nothing else.

  If the question already makes sense on its own, repeat it back word for word.
  Never work in a name, fact or detail from the conversation that the question
  did not itself refer to.

  Do not answer the question. Reply with the question and nothing else.

  Examples, for a conversation about class 9-B and its Physics teacher Meera Iyer:
    "And Chemistry?"            -> "Who teaches Chemistry in class 9-B?"
    "Where did she study?"      -> "Where did Meera Iyer study?"
    "How many students?"        -> "How many students are in class 9-B?"
    "What does a bus cost?"     -> "What does a bus cost?"
"""

# Only the most recent exchanges are worth showing the rewriter. A reference
# points at something just said, and the further back the history runs the more
# unrelated names there are for it to drag into the question.
CONDENSE_TURNS = env_int("CONDENSE_TURNS", 2)


async def condense_question(history: Sequence[dict[str, Any]], question: str) -> str:
    # The first question of a conversation has nothing to resolve against.
    if len(history) == 0:
        return question

    response = await openai.chat.completions.create(
        model=CHAT_MODEL,
        # The rewrite should be predictable, not creative.
        temperature=0,
        messages=[
            {"role": "system", "content": CONDENSE_PROMPT},
            *history[-CONDENSE_TURNS * 2 :],
            {"role": "user", "content": f"Follow-up question: {question}"},
        ],
    )

    standalone = content_of(response)

    # A model that ignored the instruction and answered instead would search for
    # its own guess, so anything implausible falls back to the original.
    if not standalone or len(standalone) > len(question) + 300:
        return question

    if standalone != question:
        print(f'Condensed: "{question}" -> "{standalone}"')

    return standalone
