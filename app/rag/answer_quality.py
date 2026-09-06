"""Retrieval quality is not answer quality, and confusing the two is how a RAG
system gets debugged in the wrong place for a week.

They are separate measurements of separate steps, and every combination of them
happens:

                   | answer is grounded   | answer is not
  -----------------+----------------------+---------------------------------
  context is good  | grounded             | missed — the passage was right
                   |                      | there and the model did not use
                   |                      | it. A generation problem: prompt,
                   |                      | model, or context ordering.
  -----------------+----------------------+---------------------------------
  context is bad   | unsupported — it     | honest-miss — nothing was found
                   | answered anyway,     | and it said so. The pipeline
                   | from what the model  | working exactly as intended, and
                   | already knew. The    | not a bug to chase.
                   | dangerous quadrant.  |

Only the top-left is a good answer, and only the bottom-right is a good failure.
The other two look identical from the outside — a user sees "an answer" or "I
don't know" either way — which is why the diagnosis is worth computing and
reporting rather than inferring later from a transcript.

The cheap signals below cost nothing and always run. The LLM judge is one extra
call and is opt-in, because grading your own homework on every request doubles
the bill of the endpoint it is measuring.
"""

import os
import re
from typing import Any

from app.config import env_float, env_str
from app.llm import ROUTER_MODEL, content_of, openai

JUDGE_MODEL = env_str("JUDGE_MODEL", ROUTER_MODEL)

# Below this share of its content words appearing in the context, an answer is
# saying things the context did not. Not proof — a correct answer can be phrased
# entirely in its own words — which is why it is a signal reported alongside the
# judge and not a verdict on its own.
SUPPORT_MIN = env_float("ANSWER_SUPPORT_MIN", 0.35)

# Whether the LLM judge runs by default. Per-request `"evaluate": true` turns it
# on for one call whatever this says.
JUDGE_BY_DEFAULT = os.environ.get("ANSWER_QUALITY") == "true"

# The model was told to say exactly this when the context does not cover the
# question. Recognising the refusal is what separates "could not" from "would
# not" — an abstention over good context is a bug, over bad context it is the
# system working.
ABSTENTIONS = [
    re.compile(r"i don'?t have enough information", re.IGNORECASE),
    re.compile(r"i do not have enough information", re.IGNORECASE),
    re.compile(
        r"\b(the )?context (does not|doesn'?t) (contain|cover|mention|include|provide)",
        re.IGNORECASE,
    ),
    re.compile(r"\bno (relevant )?information (is )?(available|provided|found)\b", re.IGNORECASE),
    re.compile(r"\bi (don'?t|do not) know\b", re.IGNORECASE),
]

# Words too common to say anything about whether a sentence came from the
# context. Short tokens are dropped by length, so this only needs the frequent
# long ones.
STOPWORDS = {
    "that", "this", "with", "from", "have", "has", "had", "been", "were", "was",
    "will", "would", "could", "should", "there", "their", "they", "them", "then",
    "than", "when", "what", "which", "while", "about", "into", "your", "you",
    "and", "the", "for", "are", "not", "but", "its", "it", "is", "of", "in", "to",
    "according", "provided", "information", "context", "document", "documents",
}

_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")


def content_words(text: str) -> list[str]:
    return [
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 3 and word not in STOPWORDS
    ]


def grounding_signals(answer: str, context: str) -> dict[str, Any]:
    """The free signals: did it refuse, and how much of what it said appears in
    the context it was given.

    `support` is 0..1, the share of the answer's content words found in the
    context — or None where overlap cannot measure it: an abstention has no
    claim in it to support, and "35" has nothing quotable in it either. Both
    report `grounded: None`, which means unknown and not ungrounded. Those are
    the answers the LLM judge is actually worth its call on.
    """
    abstained = any(pattern.search(answer) for pattern in ABSTENTIONS)

    words = content_words(answer)

    # Nothing quotable in the answer ("Yes.", "35"): overlap cannot measure it,
    # and reporting a confident 0 would be worse than reporting nothing.
    if abstained or not words or not context:
        return {"abstained": abstained, "support": None, "grounded": None}

    in_context = set(content_words(context))

    found = sum(1 for word in words if word in in_context)

    support = round(found / len(words), 3)

    return {"abstained": False, "support": support, "grounded": support >= SUPPORT_MIN}


JUDGE_PROMPT = """
  You check an answer against the passages it was supposed to be built from.

  Reply with exactly two lines and nothing else:

    supported: yes | partly | no
    reason: <one short sentence>

  "supported" is about the passages alone, never about whether the answer is
  true in general. An answer whose every claim appears in the passages is yes.
  One that adds a claim the passages do not make is partly. One built on
  something other than the passages is no.

  An answer that says it does not have enough information is "yes" when the
  passages indeed do not cover the question, and "no" when they do.
"""


async def judge_answer(question: str, context: str, answer: str) -> dict[str, str] | None:
    """The paid signal: ask a model whether the answer is actually supported by
    the context. Slower and more accurate than word overlap, and one extra call.

    Returns None when the call fails — a judge is an observation of the request,
    never a part of it, and must not be able to fail one.
    """
    try:
        response = await openai.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Passages:\n{context or '(none were retrieved)'}\n\n"
                        f"Answer: {answer}"
                    ),
                },
            ],
        )

        text = content_of(response)

        supported_match = re.search(r"supported:\s*(yes|partly|no)", text, re.IGNORECASE)

        reason_match = re.search(r"reason:\s*(.+)", text, re.IGNORECASE)

        if not supported_match:
            return None

        verdict = {"supported": supported_match.group(1).lower()}

        if reason_match:
            verdict["reason"] = reason_match.group(1).strip()

        return verdict
    except Exception as error:
        print(f"Answer judging failed, reporting the free signals only: {error}")

        return None


def diagnose(retrieval: dict[str, Any] | None, answer: dict[str, Any]) -> str:
    """The two measurements, read together: which quadrant of the table at the
    top of this file the request landed in.

    Returns one of `grounded`, `missed`, `unsupported`, `honest-miss`,
    `unmeasured`, `no-retrieval`.
    """
    # Retrieval was deliberately skipped, so there is no context to be grounded
    # in and nothing to diagnose. Answering from the conversation is the whole
    # intent of that route.
    if not retrieval:
        return "no-retrieval"

    # The judge, when it ran, outranks the overlap heuristic: it read the
    # answer, the heuristic counted words.
    grounded = (
        answer["judge"]["supported"] != "no" if answer.get("judge") else answer.get("grounded")
    )

    # Only a `correct` verdict means a passage that actually answers the
    # question reached the prompt. `ambiguous` is the on-topic near miss —
    # passages about the right subject with the answer not in them — and an
    # abstention over those is the system working, not the generator failing.
    context_has_answer = retrieval.get("verdict") == "correct" and retrieval.get("chunks", 0) > 0

    if answer.get("abstained"):
        return "missed" if context_has_answer else "honest-miss"

    if not context_has_answer:
        # It answered without a passage that answers the question. Where the
        # judge read the answer against the passages and disagreed, believe the
        # judge: it saw the answer, the chunk grader only ever saw the passages.
        judge = answer.get("judge")

        return "grounded" if judge and judge["supported"] == "yes" else "unsupported"

    # A one-word answer over good context: nothing to count, so nothing is
    # claimed either way. Reporting this as ungrounded would make "35" look like
    # a hallucination, which is the opposite of useful.
    if grounded is None:
        return "unmeasured"

    return "grounded" if grounded else "unsupported"


async def evaluate_answer(
    *,
    question: str,
    context: str,
    answer: str,
    retrieval: dict[str, Any] | None,
    judge: bool,
) -> dict[str, Any]:
    """Both halves of the answer-side measurement, and the diagnosis they add up
    to. The one thing server.py has to call.

    ``judge`` is whether to spend the extra call on the LLM judge.
    """
    signals = grounding_signals(answer, context)

    # The judge is skipped where there is no claim for it to check: an empty
    # answer (a stream that died half way is a transport failure, not a quality
    # one), and an abstention, whose diagnosis `abstained` already settles.
    verdict = (
        await judge_answer(question, context, answer)
        if judge and answer and not signals["abstained"]
        else None
    )

    measured = {**signals, **({"judge": verdict} if verdict else {})}

    return {"answer": measured, "diagnosis": diagnose(retrieval, measured)}
