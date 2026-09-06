# RAG chatbot (Python)

A retrieval-augmented school assistant: a document goes into Postgres as
embedded chunks, and every question is routed, retrieved for, graded, corrected
if the grade was poor, answered, and then measured.

This is a port of the Node/Express original in `../chatbot`. Same endpoints,
same request and response shapes, same environment variables, same behaviour.

```
                      ┌─────────────────────────────────────────┐
  question ─────────► │ classify  (rules → LLM → cache)         │  rag/classify.py
                      └──────────────────┬──────────────────────┘
                                         ▼
                      ┌─────────────────────────────────────────┐
                      │ route: retrieve at all? topK? strategy? │  rag/adaptive.py
                      └──────────────────┬──────────────────────┘
                            no │         │ yes
                               │         ▼
                               │  ┌──────────────────────────────┐
                               │  │ condense the follow-up       │  rag/condense.py
                               │  │ search: vector | hybrid |    │  rag/search.py
                               │  │         multi-query          │
                               │  └──────────────┬───────────────┘
                               │                 ▼
                               │  ┌──────────────────────────────┐
                               │  │ grade each chunk             │  rag/grade.py
                               │  │ correct: rewrite + research  │  rag/corrective.py
                               │  └──────────────┬───────────────┘
                               ▼                 ▼
                      ┌─────────────────────────────────────────┐
                      │ answer, then measure it                 │  rag/answer_quality.py
                      └─────────────────────────────────────────┘
```

## Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt fastembed

cp .env.example .env        # then put your OPENAI_API_KEY in it
docker compose up -d --wait # Postgres + pgvector on :5432

uvicorn app.server:app --port 4000 --reload
```

`.venv` here is already built that way, so `source .venv/bin/activate` is
enough. Python 3.12 rather than the 3.14 on your PATH, and the extra
`fastembed`, are both only for the local embedding option — see the table
below. On 3.14, `pip install -r requirements.txt` alone works with
`EMBEDDING_PROVIDER=openai`.

The first start reads `WPS 365.pdf`, chunks it, embeds it and writes it to
Postgres. Later starts find the source already indexed and skip straight to
answering — drop the table to rebuild.

### Choosing an embedding backend

`EMBEDDING_PROVIDER` picks one of four, and only the chosen one is imported:

| value                   | what it needs                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `huggingface` (default) | `pip install fastembed` (ONNX, light) or `sentence-transformers` (Torch). Runs locally, no key. **Python ≤ 3.13.** |
| `openai`                | `OPENAI_API_KEY`. Nothing extra to install.                                                                        |
| `gemini`                | `GEMINI_API_KEY`.                                                                                                  |
| `ollama`                | A local Ollama: `ollama pull nomic-embed-text`.                                                                    |

Neither local runtime is in `requirements.txt`, because three of the four
providers do not need one and Torch is a large download. Install one, or point
`EMBEDDING_PROVIDER` at a hosted provider.

> **On Python 3.14 the local option does not install yet** — neither
> `onnxruntime` (what fastembed needs) nor `torch` (what sentence-transformers
> needs) publishes a 3.14 wheel, and `python3` on this machine is 3.14. That is
> why the bundled `.venv` is 3.12. If you would rather stay on 3.14, set
> `EMBEDDING_PROVIDER=openai`: it needs nothing beyond `requirements.txt` and
> reuses the key the chat model already requires.

The local model is `sentence-transformers/all-MiniLM-L6-v2` — the same 384
dimensions as the `Xenova/all-MiniLM-L6-v2` the Node version ran on ONNX, and
the same vectors.

Vectors from different models are not comparable, so the table records which
model wrote it and refuses to serve a run using a different one. Use a separate
`PG_TABLE` per model, or drop the table.

## Endpoints

| method   | path                | what it does                                               |
| -------- | ------------------- | ---------------------------------------------------------- |
| `POST`   | `/chat`             | Retrieve, then answer. The endpoint.                       |
| `POST`   | `/pdf-loader`       | No retrieval: the whole PDF into the prompt.               |
| `GET`    | `/sessions`         | Conversations still in memory. Localhost only — see below. |
| `DELETE` | `/chat/{sessionId}` | Forget one conversation.                                   |

`/chat` takes either shape:

```jsonc
{ "message": "Who teaches maths?", "sessionId": "..." }  // server remembers
{ "messages": [{ "role": "user", "content": "..." }] }   // client remembers
```

with `"stream": true` for a plain-text stream, and `"evaluate": true` to spend
one extra call judging the answer.

```bash
curl -s localhost:4000/chat -H 'content-type: application/json' \
  -d '{"message":"Who teaches mathematics in class 10-A?"}'
```

The reply carries the answer and the reasoning behind it: `sources`, `routing`
(what the question was classified as and what that cost), `correction` (whether
retrieval had to be retried), `quality` (how good the context was, and whether
the answer used it), and the `history` so far. A session id comes back in the
`X-Session-Id` header — including on a stream, where the body has no room for
it, alongside `X-Retrieval-Quality`.

## The two things worth understanding

**Adaptive** decides how hard to work _before_ searching. "hi" gets no
retrieval at all — no embedding call, no database round trip, no context
tokens. An exact code gets a hybrid search that can match the literal string. A
comparison gets more chunks and several rephrasings. The routing table at the
top of `rag/adaptive.py` is the whole policy.

**Corrective** checks the result _after_ searching, which adaptive never does.
A vector search always returns its topK rows — against a corpus holding nothing
on the subject, it returns topK irrelevant rows and they go into the prompt as
though they answered the question. So the chunks are graded, and a bad grade
buys one rewrite and one different search. If both rounds come back irrelevant,
the prompt gets no context and the model says it does not know — having checked
that it does not, rather than having failed to notice.

Set `ADAPTIVE_RAG=false` or `CORRECTIVE_RAG=false` to take either one out and
see what it was buying you.

## Layout

| file                          | what lives there                                                                 |
| ----------------------------- | -------------------------------------------------------------------------------- |
| `app/server.py`               | HTTP, prompt assembly, streaming, startup indexing                               |
| `app/documents.py`            | PDF loading and recursive chunking                                               |
| `app/config.py`               | `.env`, and the typed readers for it                                             |
| `app/llm.py`                  | The OpenAI client and the two models                                             |
| `app/embeddings/`             | Four providers behind one `generate_embedding`                                   |
| `app/store/pgvector.py`       | The table, the vector search, the keyword search                                 |
| `app/memory/conversations.py` | Session history, in this process                                                 |
| `app/rag/`                    | classify, condense, search, adaptive, grade, rewrite, corrective, answer_quality |

## Notes on the port

- **Express → FastAPI/uvicorn**, both async end to end. Streaming is a
  `StreamingResponse` where the original wrote to the response object.
- **`node-postgres` → `asyncpg`**, which also uses `$1` placeholders, so the SQL
  is unchanged. `jsonb` gets a codec so metadata arrives as a dict.
- **`pdf-parse` → `pypdf`**. Parsing is blocking, so it runs in a worker thread.
- **`@huggingface/transformers` → `fastembed`** (also ONNX), falling back to
  `sentence-transformers`.
- Module-level constants are read from the environment at import time, the way
  the original read `process.env` — so `.env` is loaded by `app/__init__.py`
  before anything else.
- `index.js` was an earlier in-memory prototype superseded by `server.js`; only
  the latter is ported. Its cosine-similarity helper is gone with it — pgvector
  computes distances now.

## Before this serves anyone but you

`GET /sessions` hands out every live session id, and an id is the whole of a
conversation's security: anyone holding one can continue that conversation and
read its history back. It is a debugging convenience for localhost. Put it
behind auth, or take it out.

Session history lives in this process, so it does not survive a restart and is
not shared by a second instance. `app/memory/conversations.py` is the whole
surface to swap for Redis or a table when it needs to.
