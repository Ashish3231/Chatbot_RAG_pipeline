"""The vector database step of the pipeline, on Postgres + pgvector.

Needs a Postgres with the vector extension available:

    docker compose up -d --wait          (see compose.yaml)

Everything lives in one table: the chunk text, its metadata, and the embedding
in a `vector` column that pgvector can search by cosine distance.
"""

import asyncio
import json
import re
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

import asyncpg

from app.config import env_str
from app.embeddings import Embedder

DATABASE_URL = env_str("DATABASE_URL", "postgres://postgres:postgres@localhost:5432/chatbot")

# One table per embedding model (see the model check below), so the table name
# is configurable. It goes into DDL as an identifier and cannot be a bind
# parameter, so it is whitelisted rather than escaped.
TABLE_NAME = env_str("PG_TABLE", "chunks")

if not re.fullmatch(r"[a-z_][a-z0-9_]*", TABLE_NAME):
    raise RuntimeError(
        f'Invalid PG_TABLE "{TABLE_NAME}": use lower-case letters, digits and underscores.'
    )

# Where we remember which model wrote a table. Shared by every table.
REGISTRY = "rag_embedding_models"

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()

# The table is only created once we know the embedding width, which we learn
# from the first batch of vectors. Until then queries have nothing to read.
_ready = False


async def _init_connection(connection: asyncpg.Connection) -> None:
    """asyncpg hands back jsonb as text unless told otherwise. Decoding it here
    means `metadata` arrives as a dict everywhere it is read."""
    await connection.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def get_pool() -> asyncpg.Pool:
    """The connection pool, created on first use.

    ``asyncpg.create_pool`` is a coroutine, so unlike ``new pg.Pool()`` it
    cannot happen at import time; the lock is what keeps two concurrent first
    requests from building two pools.
    """
    global _pool

    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = await asyncpg.create_pool(
                    dsn=DATABASE_URL, init=_init_connection, min_size=1, max_size=10
                )

    return _pool


def to_vector(embedding: Sequence[float]) -> str:
    """pgvector's text format is '[1,2,3]'."""
    return json.dumps(list(embedding))


async def _ensure_schema(embedder: Embedder, dimensions: int | None = None) -> bool:
    """Make sure the extension, registry and chunk table exist.

    ``dimensions`` is required the first time, when the table is created.
    Returns whether the chunk table now exists.
    """
    global _ready

    if _ready:
        return True

    try:
        pool = await get_pool()

        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await pool.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {REGISTRY} (
              table_name text PRIMARY KEY,
              provider   text NOT NULL,
              model      text NOT NULL,
              dimensions integer NOT NULL
            )
            """
        )

        registered = await pool.fetchrow(
            f"SELECT * FROM {REGISTRY} WHERE table_name = $1", TABLE_NAME
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not set up Postgres at {redact_url(DATABASE_URL)}: {describe(error)}\n"
            f"Start it in another terminal with:  docker compose up -d --wait"
        ) from error

    if registered:
        # Vectors are only comparable to others from the same model — different
        # models give different dimensions and a different geometry. Mixing them
        # silently returns nonsense, so fail loudly instead.
        if registered["model"] != embedder.model:
            raise RuntimeError(
                f'Table "{TABLE_NAME}" holds {registered["model"]} vectors but this run '
                f"embeds with {embedder.model}. Set PG_TABLE to a different name, or drop "
                f"the table to start over."
            )

        # A table created before keyword_chunks existed has no full-text index yet.
        await _ensure_fts_index()

        _ready = True
        return True

    if not dimensions:
        # Nothing indexed yet, and no vectors in hand to size the column with.
        return False

    pool = await get_pool()

    await pool.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
          id          text PRIMARY KEY,
          source      text NOT NULL,
          chunk_index integer NOT NULL,
          content     text NOT NULL,
          metadata    jsonb NOT NULL DEFAULT '{{}}',
          embedding   vector({dimensions}) NOT NULL
        )
        """
    )

    # Approximate nearest-neighbour index. Without it pgvector still answers,
    # by scanning every row — fine for a handful of chunks, not for a corpus.
    await pool.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
          ON {TABLE_NAME} USING hnsw (embedding vector_cosine_ops)
        """
    )

    # Re-indexing a document deletes its rows by source first.
    await pool.execute(
        f"CREATE INDEX IF NOT EXISTS {TABLE_NAME}_source_idx ON {TABLE_NAME} (source)"
    )

    await _ensure_fts_index()

    await pool.execute(
        f"""
        INSERT INTO {REGISTRY} (table_name, provider, model, dimensions)
        VALUES ($1, $2, $3, $4) ON CONFLICT (table_name) DO NOTHING
        """,
        TABLE_NAME,
        embedder.provider,
        embedder.model,
        dimensions,
    )

    _ready = True
    return True


async def _ensure_fts_index() -> None:
    """The index behind keyword_chunks. Separate from the table creation because
    tables created before that search existed need it too, and CREATE INDEX IF
    NOT EXISTS costs nothing on the runs where it is already there."""
    pool = await get_pool()

    await pool.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_content_fts_idx
          ON {TABLE_NAME} USING gin (to_tsvector('english', content))
        """
    )


def describe(error: BaseException) -> str:
    """A refused connection can arrive with an empty message of its own. Dig out
    something readable."""
    if str(error):
        return str(error)

    if isinstance(error, BaseExceptionGroup):
        messages = {str(inner) or type(inner).__name__ for inner in error.exceptions}

        return "; ".join(sorted(messages))

    return type(error).__name__


def redact_url(url: str) -> str:
    """Keep the password out of error messages and logs."""
    try:
        parts = urlsplit(url)

        if parts.password:
            host = parts.hostname or ""

            if parts.port:
                host = f"{host}:{parts.port}"

            netloc = f"{parts.username}:***@{host}"

            return urlunsplit(parts._replace(netloc=netloc))

        return url
    except ValueError:
        return url


async def count_chunks(embedder: Embedder) -> int:
    """How many chunks the table holds in total."""
    if not await _ensure_schema(embedder):
        return 0

    pool = await get_pool()

    return await pool.fetchval(f"SELECT count(*)::int AS count FROM {TABLE_NAME}")


async def has_source(embedder: Embedder, source: str) -> bool:
    """Whether anything from this source has been indexed already."""
    if not await _ensure_schema(embedder):
        return False

    pool = await get_pool()

    row = await pool.fetchrow(f"SELECT 1 FROM {TABLE_NAME} WHERE source = $1 LIMIT 1", source)

    return row is not None


# Six bind parameters per row, against a wire-protocol limit of 65535.
INSERT_BATCH = 500


async def add_chunks(
    embedder: Embedder,
    *,
    source: str,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadata: dict[str, Any],
) -> None:
    """Write one document's chunks into the table, replacing whatever was stored
    for that source before. Ids are derived from the source and the chunk
    position, so re-indexing updates rows in place rather than duplicating them.
    """
    await _ensure_schema(embedder, len(embeddings[0]))

    pool = await get_pool()

    async with pool.acquire() as connection:
        # One transaction, so a failure part-way cannot leave the source half
        # deleted and half rewritten.
        async with connection.transaction():
            # Drop stale rows first: a re-indexed document may split into fewer
            # chunks than last time, and those extra ids would otherwise linger.
            await connection.execute(f"DELETE FROM {TABLE_NAME} WHERE source = $1", source)

            for start in range(0, len(chunks), INSERT_BATCH):
                batch = chunks[start : start + INSERT_BATCH]

                values: list[Any] = []
                rows: list[str] = []

                for i, chunk in enumerate(batch):
                    index = start + i

                    # metadata goes in as a dict, not a JSON string: the jsonb
                    # codec registered in _init_connection does the encoding,
                    # and dumping it here first would store a JSON *string*
                    # that reads back as text instead of an object.
                    values.extend(
                        [
                            f"{source}#{index}",
                            source,
                            index,
                            chunk,
                            metadata,
                            to_vector(embeddings[index]),
                        ]
                    )

                    p = i * 6
                    rows.append(
                        f"(${p + 1}, ${p + 2}, ${p + 3}, ${p + 4}, ${p + 5}, ${p + 6}::vector)"
                    )

                await connection.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (id, source, chunk_index, content, metadata, embedding)
                    VALUES {", ".join(rows)}
                    ON CONFLICT (id) DO UPDATE
                      SET content = EXCLUDED.content,
                          metadata = EXCLUDED.metadata,
                          embedding = EXCLUDED.embedding
                    """,
                    *values,
                )


def _to_match(row: asyncpg.Record) -> dict[str, Any]:
    """One database row, in the shape the rest of the pipeline passes around."""
    metadata = dict(row["metadata"] or {})
    metadata.update({"source": row["source"], "chunkIndex": row["chunk_index"]})

    return {"text": row["content"], "score": float(row["score"]), "metadata": metadata}


async def query_chunks(
    embedder: Embedder, embedding: Sequence[float], top_k: int
) -> list[dict[str, Any]]:
    """The retrieval step: nearest chunks to a question embedding.

    Returns [{ text, score, metadata }], best match first.
    """
    if not await _ensure_schema(embedder):
        return []

    pool = await get_pool()

    # <=> is pgvector's cosine distance. Ordering by the same expression we
    # select is what lets the HNSW index answer the query.
    rows = await pool.fetch(
        f"""
        SELECT content, source, chunk_index, metadata,
               1 - (embedding <=> $1::vector) AS score
          FROM {TABLE_NAME}
         ORDER BY embedding <=> $1::vector
         LIMIT $2
        """,
        to_vector(embedding),
        top_k,
    )

    return [_to_match(row) for row in rows]


async def keyword_chunks(embedder: Embedder, question: str, top_k: int) -> list[dict[str, Any]]:
    """The other half of a hybrid search: literal word matching, ranked by
    Postgres' own text search.

    Embeddings are good at meaning and bad at exact strings — "WPS 365" and
    "WPS 366" sit almost on top of each other in vector space. When the user
    wants those characters found, this is the search that finds them.

    Returns the same [{ text, score, metadata }] shape as query_chunks, though
    the score is a text rank and not a cosine similarity, so the two are worth
    merging by rank and not by value.
    """
    if not await _ensure_schema(embedder):
        return []

    # Keep letters and digits only, then OR the words together. Stripping the
    # punctuation is what makes it safe to hand to to_tsquery, and OR rather
    # than AND is what keeps a long question from matching nothing at all.
    terms = [
        word
        for word in re.sub(r"[^a-z0-9\s]", " ", question.lower()).split()
        if len(word) > 2
    ]

    if not terms:
        return []

    pool = await get_pool()

    rows = await pool.fetch(
        f"""
        SELECT content, source, chunk_index, metadata,
               ts_rank(to_tsvector('english', content), query) AS score
          FROM {TABLE_NAME}, to_tsquery('english', $1) AS query
         WHERE to_tsvector('english', content) @@ query
         ORDER BY score DESC
         LIMIT $2
        """,
        " | ".join(terms),
        top_k,
    )

    return [_to_match(row) for row in rows]


async def close_store() -> None:
    """Close the pool so the process can exit cleanly."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
