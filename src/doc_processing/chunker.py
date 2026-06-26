"""Document chunking using RecursiveCharacterTextSplitter.

Based on both articles' consensus: 500-char chunks, 80-char overlap,
heading-aware separators. This is the "soul parameter" of the system.
"""

import hashlib
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default separators: heading > paragraph > sentence > word
DEFAULT_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n",
    "。",
    ".",
    "！",
    "？",
    "；",
    " ",
]


def create_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a RecursiveCharacterTextSplitter with Chinese-optimized settings.

    Args:
        chunk_size: Maximum characters per chunk (default 500).
        chunk_overlap: Overlap between adjacent chunks (default 80, ~16%).
        separators: Priority-ordered list of separators.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_document(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Split a document into chunks with metadata.

    Args:
        text: Document text content.
        metadata: Base metadata to attach to every chunk (file info, etc.).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.
        separators: Priority-ordered list of separators.

    Returns:
        List of chunk dicts, each with 'content' and 'metadata' keys.
    """
    splitter = create_splitter(chunk_size, chunk_overlap, separators)
    chunks = splitter.split_text(text)

    if metadata is None:
        metadata = {}

    result = []
    total = len(chunks)
    for i, chunk_text in enumerate(chunks):
        # Skip empty or whitespace-only chunks
        if not chunk_text.strip():
            continue
        content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
        chunk_meta = {
            **metadata,
            "chunk_index": i,
            "total_chunks": total,
            "chunk_id": content_hash[:12],
            "content_hash": content_hash,
            "chunk_size": len(chunk_text),
        }
        result.append({
            "content": chunk_text,
            "metadata": chunk_meta,
        })

    # Update total_chunks to reflect actual (non-empty) chunk count
    actual_total = len(result)
    for chunk in result:
        chunk["metadata"]["total_chunks"] = actual_total

    return result