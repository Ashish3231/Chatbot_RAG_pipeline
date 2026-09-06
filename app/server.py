"""The HTTP layer: FastAPI where the original was Express.

Run it with:

    uvicorn app.server:app --port 4000 --reload
"""

import os
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.config import ROOT, env_bool, env_str
from app.documents import CHUNK_SIZE, load_pdf_document, recursive_split
from app.embeddings import EMBEDDER, generate_embedding
from app.llm import CHAT_MODEL, content_of, openai
from app.memory.conversations import (
    append_turn,
    create_session_id,
    get_history,
    is_session_id,
    list_sessions,
    reset_session,
)
from app.rag.adaptive import adaptive_retrieve
from app.rag.answer_quality import JUDGE_BY_DEFAULT, evaluate_answer
from app.rag.corrective import corrective_retrieve
from app.rag.retrieve import fixed_retrieve
from app.store.pgvector import TABLE_NAME, add_chunks, close_store, count_chunks, has_source

# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

PDF_PATH = env_str("PDF_PATH", str(ROOT / "WPS 365.pdf"))


# ---------------------------------------------------------------------------
# Indexing: chunk -> embed -> store in Postgres
# ---------------------------------------------------------------------------


async def index_text(text: str, metadata: dict[str, Any]) -> None:
    """The shared pipeline: chunk any text, embed the chunks, write them to
    Postgres."""
    source = metadata["source"]

    # Embedding is the slow, and for hosted providers the paid, part of the run.
    # Postgres keeps the vectors, so once a source is in there a restart can
    # skip straight to answering questions.
    if await has_source(EMBEDDER, source):
        print(f"Already indexed: {source} (drop the table to rebuild)")
        return

    chunks = [chunk for chunk in recursive_split(text, CHUNK_SIZE) if chunk.strip()]

    if not chunks:
        return

    # One batched call is much faster than embedding chunks one at a time.
    embeddings = await generate_embedding(chunks)

    await add_chunks(
        EMBEDDER, source=source, chunks=chunks, embeddings=embeddings, metadata=metadata
    )

    print(
        f"Indexed {len(chunks)} chunks from {source} "
        f"({len(embeddings[0])} dimensions each)"
    )


async def build_vector_store(file_path: str) -> None:
    """Source 1: a PDF file on disk."""
    document = await load_pdf_document(file_path)

    await index_text(document["content"], document["metadata"])


async def build_vector_store_from_text(text: str, source: str = "inline-text") -> None:
    """Source 2: a plain paragraph of text, no file involved."""
    await index_text(text, {"source": source, "type": "text"})


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Two pipelines, same shape in and out, so /chat can await either one:
#
#   adaptive_retrieve  classifies the question, then decides whether to search
#                      at all, how many chunks to ask for and which search to
#                      run (rag/adaptive.py)
#   fixed_retrieve     always rewrites, always searches, always TOP_K chunks
#                      (rag/retrieve.py)
#
# Set ADAPTIVE_RAG=false to fall back to the fixed one — useful for seeing what
# the routing is actually buying you.
ADAPTIVE_RAG = env_bool("ADAPTIVE_RAG", True)

base_retrieve = adaptive_retrieve if ADAPTIVE_RAG else fixed_retrieve

# ...and a layer on top of whichever one runs. Both of them decide how to search
# *before* searching, and neither looks at what came back: an empty corpus still
# returns topK rows, and those rows still go into the prompt.
#
# The corrective layer grades what was retrieved, and when it does not answer the
# question, rewrites the query and searches again with a different strategy
# (rag/corrective.py). CORRECTIVE_RAG=false removes it, which is how to see what
# the checking is buying you.
CORRECTIVE_RAG = env_bool("CORRECTIVE_RAG", True)


async def retrieve(history: Sequence[dict[str, Any]], question: str) -> dict[str, Any]:
    if CORRECTIVE_RAG:
        return await corrective_retrieve(history, question, base_retrieve)

    return await base_retrieve(history, question)


print(
    f"Retrieval: {'adaptive' if ADAPTIVE_RAG else 'fixed'}"
    f"{' + corrective' if CORRECTIVE_RAG else ''}"
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

PARAGRAPH = """
  School: ABC International School

  Class: 10-A
  Mathematics Teacher: Rahul Sharma
  English Teacher: Priya Singh
  Total Students: 35
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: fill the vector store before taking requests."""
    try:
        await build_vector_store(PDF_PATH)
        await build_vector_store_from_text(PARAGRAPH, "school-info")

        print(f'Table "{TABLE_NAME}" holds {await count_chunks(EMBEDDER)} chunks')
    except Exception as error:
        # Without the vector store /chat has nothing to retrieve, so stop here
        # rather than serving answers with an empty context.
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    yield

    await close_store()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> PlainTextResponse:
    return PlainTextResponse("Hello, World!")


SYSTEM_PROMPT = """
  You are a helpful school assistant.

  You are given two different things, and they answer different questions:

  - The conversation: what the user has told you, and what you have already
    said. This is the authority on the user and on the chat itself.
  - The context below: passages retrieved from the school's documents for this
    question alone. It is the authority on the school. It is fetched fresh
    every turn and may be irrelevant to what was asked.

  Answer a question about the school from the context. If the context does not
  cover it, say "I don't have enough information." — do not guess.

  Answer a question about the user or about this conversation from the
  conversation. Never take a detail from the context to fill in, extend or
  correct something the user told you: if they gave their name as "Abhinay",
  that is their name, whatever a document says about somebody similar.

  When the context has nothing to do with the question, ignore it rather than
  working it into the answer.
"""


class ChatRequest:
    """Everything an endpoint needs out of the body."""

    __slots__ = ("session_id", "question", "history")

    def __init__(
        self,
        session_id: str | None,
        question: str,
        history: list[dict[str, str]],
    ) -> None:
        self.session_id = session_id
        self.question = question
        self.history = history


def parse_chat_request(body: Any) -> ChatRequest | None:
    """Two request shapes are accepted:

      { message, sessionId? }  the server remembers the conversation, and names
                               it in the X-Session-Id response header
      { messages: [...] }      the client keeps its own history; nothing is stored

    Returns None if the body carried no question. History comes from the
    server's session or from the request — never both.
    """
    if not isinstance(body, dict):
        return None

    session_id = body.get("sessionId")
    message = body.get("message")
    messages = body.get("messages")

    if isinstance(message, str) and message.strip():
        # An id we did not issue starts a new conversation rather than being
        # trusted.
        chat_id = session_id if is_session_id(session_id) else create_session_id()

        return ChatRequest(chat_id, message.strip(), get_history(chat_id))

    if isinstance(messages, list) and messages:
        last = messages[-1]
        question = last.get("content") if isinstance(last, dict) else None

        if isinstance(question, str) and question.strip():
            return ChatRequest(None, question.strip(), to_history(messages[:-1]))

    return None


def to_history(messages: Sequence[Any]) -> list[dict[str, str]]:
    """Only the two roles a conversation is made of, and only role and content."""
    return [
        {"role": entry["role"], "content": entry["content"]}
        for entry in messages
        if isinstance(entry, dict)
        and entry.get("role") in ("user", "assistant")
        and isinstance(entry.get("content"), str)
    ]


def build_messages(
    history: Sequence[dict[str, str]], question: str, context: str
) -> list[dict[str, str]]:
    """The prompt: the instructions, the retrieved context, then the conversation."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Adaptive routing answers some questions without retrieving anything. An
    # empty "Context:" block would only be tokens spent telling the model to
    # ignore it, so it is left out instead.
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    messages.extend(history)
    messages.append({"role": "user", "content": question})

    return messages


async def complete_answer(messages: Sequence[dict[str, str]]) -> str:
    """Ask for the whole answer at once."""
    response = await openai.chat.completions.create(model=CHAT_MODEL, messages=list(messages))

    return content_of(response)


async def send_answer(
    *,
    session_id: str | None,
    history: list[dict[str, str]],
    question: str,
    stream: bool,
    retrieval: dict[str, Any],
    evaluate: bool = False,
):
    """The half of a request that is the same everywhere: build the prompt, get
    the answer, remember the turn, reply.

    It takes the retrieved context as *data*, not as a callback — so each
    endpoint below reads straight down the page instead of nesting.
    """
    context = retrieval.get("context", "")
    sources = retrieval.get("sources", [])
    standalone_question = retrieval.get("standaloneQuestion")
    routing = retrieval.get("routing")
    quality = retrieval.get("quality")
    correction = retrieval.get("correction")

    messages = build_messages(history, question, context)

    headers: dict[str, str] = {}

    if session_id:
        headers["X-Session-Id"] = session_id

    # Retrieval quality is known before a token is generated, so it is the one
    # measurement a stream can still carry in a header. Answer quality is not:
    # by the time it exists the body is on the wire, so it is logged instead.
    if quality:
        headers["X-Retrieval-Quality"] = (
            f'{quality["retrieval"]["verdict"]};{quality["retrieval"]["score"]}'
        )

    async def measure(answer: str) -> dict[str, Any] | None:
        """The other half of the measurement. Retrieval quality says whether the
        right passages were found; this says whether the answer used them — and
        the two disagreeing is the most informative thing either can tell you.
        Only worth computing where there was a retrieval to be grounded in."""
        if not quality:
            return None

        evaluation = await evaluate_answer(
            question=question,
            context=context,
            answer=answer,
            retrieval=quality["retrieval"],
            judge=evaluate,
        )

        support = evaluation["answer"]["support"]
        judge = evaluation["answer"].get("judge")

        print(
            f'Answer: {evaluation["diagnosis"]}'
            + ("" if support is None else f", support {support}")
            + (f', judged {judge["supported"]}' if judge else "")
        )

        return evaluation

    if stream:
        # Opened here rather than inside the generator: a request that fails to
        # start is still a failed request, and the caller's `except` can only
        # turn it into a 500 while nothing has been sent. Once the generator is
        # running the status line is already on the wire.
        response = await openai.chat.completions.create(
            model=CHAT_MODEL, messages=messages, stream=True
        )

        async def body() -> AsyncIterator[str]:
            """Stream the answer back as plain text, keeping the whole of it —
            it is needed to record the turn."""
            answer = ""

            async for chunk in response:
                content = chunk.choices[0].delta.content if chunk.choices else None

                if content:
                    answer += content
                    yield content

            # Only a completed answer is remembered. A stream that died half way
            # would otherwise leave a truncated reply behind to confuse every
            # later turn.
            if session_id and answer:
                append_turn(session_id, question, answer)

            await measure(answer)

        return StreamingResponse(
            body(), media_type="text/plain; charset=utf-8", headers=headers
        )

    answer = await complete_answer(messages)

    if session_id and answer:
        append_turn(session_id, question, answer)

    evaluation = await measure(answer)

    # The conversation as it now stands, this exchange included, so a client can
    # render the transcript without keeping a second copy of it. The stateless
    # shape stores nothing, so its history is what it sent plus this turn.
    transcript = (
        get_history(session_id)
        if session_id
        else [
            *history,
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )

    payload: dict[str, Any] = {}

    # First, because it is the field the next request has to send back.
    if session_id:
        payload["sessionId"] = session_id

    payload["answer"] = answer

    # Only when the rewriter actually changed something: this is the question
    # retrieval ran on, and the reason the chunks are what they are.
    if standalone_question and standalone_question != question:
        payload["standaloneQuestion"] = standalone_question

    payload["sources"] = sources

    # How the question was routed, and what that cost. Absent on the endpoints
    # that do not route.
    if routing:
        payload["routing"] = routing

    # What the retrieval was graded at, what the answer was graded at, and the
    # one word that reads them together.
    if evaluation:
        payload["quality"] = {
            **quality,
            "answer": evaluation["answer"],
            "diagnosis": evaluation["diagnosis"],
        }

    # Whether the retrieval had to be corrected, and what each round did.
    if correction:
        payload["correction"] = correction

    payload["history"] = transcript

    return JSONResponse(payload, headers=headers)


def handle_error(error: Exception) -> JSONResponse:
    """The one thing every endpoint does with a failure."""
    print("Error occurred:", error, file=sys.stderr)
    traceback.print_exception(error)

    return JSONResponse({"error": "Internal Server Error"}, status_code=500)


async def read_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        return {}

    return body if isinstance(body, dict) else {}


@app.post("/chat")
async def chat(request: Request):
    """Retrieval-augmented chat: find the context, then answer with it."""
    body = await read_body(request)

    parsed = parse_chat_request(body)

    if not parsed:
        return JSONResponse({"error": "message (or messages) is required"}, status_code=400)

    try:
        # Retrieve, grade, and correct if the grade was poor. Every pipeline
        # returns the same thing — context, sources, and what it did to get them —
        # so which one is configured is settled above and not here.
        retrieval = await retrieve(parsed.history, parsed.question)

        return await send_answer(
            session_id=parsed.session_id,
            history=parsed.history,
            question=parsed.question,
            # The default reply is one JSON object, so the session id comes back
            # where a REST client can read it. Streaming is what a chat UI wants,
            # and stays one flag away.
            stream=body.get("stream") is True,
            # The LLM judge on the answer is an extra call per request, so it is
            # off unless this request asks for it or ANSWER_QUALITY=true says
            # always.
            evaluate=body.get("evaluate") is True or JUDGE_BY_DEFAULT,
            retrieval=retrieval,
        )
    except Exception as error:
        return handle_error(error)


@app.get("/sessions")
async def sessions():
    """Every conversation the server is still holding. Sessions drop off this
    list on their own once SESSION_TTL_MINUTES passes without a message.

    An id is enough to continue someone's conversation, so handing out the whole
    list is a thing to do on localhost and not in front of anyone else.
    """
    listed = list_sessions()

    return {"count": len(listed), "sessions": listed}


@app.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Forget a conversation: the next message with this id starts a fresh one."""
    return {"cleared": reset_session(session_id)}


@app.post("/pdf-loader")
async def pdf_loader(request: Request):
    """No retrieval: stuff the whole PDF into the prompt. Fine for small
    documents. Nothing to route or rewrite either — the context does not depend
    on the question, so there is no `routing` on the reply."""
    body = await read_body(request)

    parsed = parse_chat_request(body)

    if not parsed:
        return JSONResponse({"error": "message (or messages) is required"}, status_code=400)

    try:
        document = await load_pdf_document(PDF_PATH)

        return await send_answer(
            session_id=parsed.session_id,
            history=parsed.history,
            question=parsed.question,
            stream=body.get("stream") is True,
            retrieval={
                "context": document["content"],
                "sources": [{"source": PDF_PATH}],
            },
        )
    except Exception as error:
        return handle_error(error)


def main() -> None:
    """`python -m app.server` — the equivalent of `npm start`."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT") or 4000))


if __name__ == "__main__":
    main()
