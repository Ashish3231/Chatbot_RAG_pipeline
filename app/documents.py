"""Document loading and chunking: the front half of the pipeline.

PDF ────────┐
DOCX ───────┤
TXT ────────┤
CSV ────────┤
Website ────┤
Database ───┘
      │
      ▼
┌──────────────────┐
│ Document Loader  │  ← this file
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Document         │
│ + Metadata       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Text Chunking    │  ← and this one
└────────┬─────────┘
         ▼
    Embeddings ─► Vector Database ─► Retrieval ─► Context Injection ─► LLM
"""

import asyncio
from pathlib import Path
from typing import Any, Sequence

from pypdf import PdfReader

# ~500 characters keeps each chunk small enough to embed well, but big enough
# to carry meaning.
CHUNK_SIZE = 500


def _read_pdf(file_path: str) -> dict[str, Any]:
    reader = PdfReader(file_path)

    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    return {
        "content": text,
        "metadata": {
            "source": file_path,
            "type": "pdf",
            "pages": len(reader.pages),
        },
    }


async def load_pdf_document(file_path: str | Path) -> dict[str, Any]:
    """The whole text of a PDF, plus where it came from.

    Parsing is blocking and can take a while on a large document, so it runs in
    a worker thread rather than on the event loop.
    """
    return await asyncio.to_thread(_read_pdf, str(file_path))


def recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    separators: Sequence[str] = ("\n\n", "\n", " ", ""),
) -> list[str]:
    """Split on the coarsest separator that fits, falling back to finer ones.

    Paragraphs first, then lines, then words, then — for a single unbroken run
    of characters — anywhere at all.
    """
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]

    parts = list(text) if separator == "" else text.split(separator)

    chunks: list[str] = []
    current_chunk = ""

    for part in parts:
        candidate = current_chunk + separator + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # A single part can still be too big: split it again with a finer
            # separator.
            if len(part) > chunk_size and len(separators) > 1:
                chunks.extend(recursive_split(part, chunk_size, separators[1:]))
                current_chunk = ""
            else:
                current_chunk = part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
