"""Step 2 of corrective RAG: is what we just retrieved actually any good?

A vector search always returns its top_k rows. It has no way of saying "none of
these answer the question" — the nearest chunk to "what does a bus cost?" in a
document about leave policy is still *a* chunk, and it is handed to the model as
though it were context. That is where a confident wrong answer comes from: not a
bad model, but a prompt whose context was never checked.

So the chunks are graded before they are used, and each one comes back as:

  relevant     answers the question, or a part of it
  partial      about the right subject, does not contain the answer
  irrelevant   nothing to do with what was asked

The grades then decide the verdict for the whole retrieval, which is what
corrective.py routes on. Grading is one small LLM call for the whole batch —
per-chunk calls would cost more than the answer does.
"""

import re
from typing import Any, Sequence

from app.config import env_bool, env_int, env_str
from app.llm import ROUTER_MODEL, content_of, openai
from app.rag.search import Match

# Grading is a short call on short text: the smallest model available is the
# right one, whatever answers the question.
GRADE_MODEL = env_str("GRADE_MODEL", ROUTER_MODEL)

# How much of each chunk the grader is shown. Relevance is nearly always visible
# in the opening lines, and the whole point of this step is that it stays
# cheaper than the mistake it prevents.
SNIPPET = env_int("GRADE_SNIPPET", 400)

# Whether `partial` chunks are worth putting in the prompt. They are context
# without the answer in them — useful for a broad question, noise for a pointed
# one.
KEEP_PARTIAL = env_bool("GRADE_KEEP_PARTIAL", True)

# Grades are keyed by the question and the exact chunks graded, so a repeated
# question costs nothing the second time. Bounded, oldest out first.
MAX_CACHE = env_int("GRADE_CACHE", 300)

# What each grade is worth when the retrieval is scored as a whole.
WEIGHTS = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0}

GRADE_PROMPT = """
  You grade passages retrieved for a question, one line each.

  For every numbered passage reply with its number and one word:

    yes      the passage contains the answer, or part of it
    partly   the passage is about the right subject but does not answer it
    no       the passage has nothing to do with the question

  Judge each passage on its own, against the question as asked. A passage that
  is merely on a related topic is "partly", not "yes". Grade what the passage
  says, never what you happen to know.

  Reply with one line per passage, formatted "1: yes", and nothing else.
"""

_cache: dict[str, list[str]] = {}


def _remember(key: str, value: list[str]) -> None:
    _cache[key] = value

    if len(_cache) > MAX_CACHE:
        del _cache[next(iter(_cache))]


def _cache_key(question: str, matches: Sequence[Match]) -> str:
    """The chunks graded, in order, so the same set is only ever paid for once."""
    ids = [
        f'{match["metadata"]["source"]}#{match["metadata"]["chunkIndex"]}' for match in matches
    ]

    return f'{question.strip().lower()}::{",".join(ids)}'


async def grade_matches(question: str, matches: Sequence[Match]) -> dict[str, Any]:
    """Grade every retrieved chunk, then the retrieval as a whole.

    Returns { verdict, graded, kept, score, counts, by }, where `verdict` is one
    of 'correct' | 'ambiguous' | 'incorrect', `kept` is the subset worth putting
    in the prompt, and `score` is the share of the retrieved chunks that earned
    their place — retrieval quality, on its own, before a word of the answer has
    been generated.
    """
    # Nothing came back. No call to make, and the verdict is not in doubt.
    if not matches:
        return {
            "verdict": "incorrect",
            "graded": [],
            "kept": [],
            "score": 0,
            "counts": {"relevant": 0, "partial": 0, "irrelevant": 0},
            "by": "empty",
        }

    key = _cache_key(question, matches)

    cached = _cache.get(key)

    if cached:
        return {**_summarise(matches, cached), "by": "cache"}

    try:
        labels = await _grade_by_llm(question, matches)
    except Exception as error:
        # Grading is a guard on the context, never a reason to fail the request.
        # The safe mistake here is to behave like an ungraded pipeline — keep
        # what was retrieved — rather than to spend a correction round on a
        # verdict we do not actually have.
        print(f"Grading failed, using the retrieval as it stands: {error}")

        return {
            **_summarise(matches, ["relevant"] * len(matches)),
            "by": "skipped",
        }

    _remember(key, labels)

    return {**_summarise(matches, labels), "by": "llm"}


async def _grade_by_llm(question: str, matches: Sequence[Match]) -> list[str]:
    """One call, every chunk, one line of answer each."""
    passages = "\n\n".join(
        f'[{index + 1}] {match["text"][:SNIPPET]}' for index, match in enumerate(matches)
    )

    response = await openai.chat.completions.create(
        model=GRADE_MODEL,
        # A verdict, not an opinion.
        temperature=0,
        messages=[
            {"role": "system", "content": GRADE_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nPassages:\n\n{passages}"},
        ],
    )

    return _parse_grades(content_of(response), len(matches))


WORDS = {"yes": "relevant", "partly": "partial", "partial": "partial", "no": "irrelevant"}

_GRADE_LINE = re.compile(r"^\s*\[?(\d+)\]?\s*[:.)\-]?\s*(yes|partly|partial|no)\b", re.IGNORECASE)


def _parse_grades(text: str, count: int) -> list[str]:
    """"1: yes" -> position 0 is relevant. Anything the model did not grade, or
    graded in a way we cannot read, counts as `partial`: unread is not the same
    as unusable, and a wrong `irrelevant` throws away the answer."""
    labels = ["partial"] * count

    for line in text.split("\n"):
        matched = _GRADE_LINE.match(line.strip())

        if not matched:
            continue

        index = int(matched.group(1)) - 1

        if 0 <= index < count:
            labels[index] = WORDS[matched.group(2).lower()]

    return labels


def _summarise(matches: Sequence[Match], labels: Sequence[str]) -> dict[str, Any]:
    """The grades, turned into the decision corrective.py routes on.

    The verdict is about whether the answer is *in there*, so it is driven by
    the best grade and not by the average: one chunk that answers the question
    is a good retrieval even alongside nine that do not. The average is reported
    separately as `score`, which is the other question — how much of what we are
    about to pay tokens for is worth sending.
    """
    graded = [{**match, "grade": labels[index]} for index, match in enumerate(matches)]

    counts = {"relevant": 0, "partial": 0, "irrelevant": 0}

    for label in labels:
        counts[label] += 1

    kept = [
        match
        for match in graded
        if match["grade"] == "relevant" or (KEEP_PARTIAL and match["grade"] == "partial")
    ]

    if counts["relevant"] > 0:
        verdict = "correct"
    elif counts["partial"] > 0:
        verdict = "ambiguous"
    else:
        verdict = "incorrect"

    score = sum(WEIGHTS[label] for label in labels) / len(labels)

    return {
        "verdict": verdict,
        "graded": graded,
        "kept": kept,
        "score": round(score, 3),
        "counts": counts,
    }
