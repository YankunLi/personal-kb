"""Document chunking using RecursiveCharacterTextSplitter + semantic chunking.

Based on both articles' consensus: 500-char chunks, 80-char overlap,
heading-aware separators. This is the "soul parameter" of the system.

Also provides semantic chunking that respects document structure:
headings → paragraphs → sentences → character-level splits.
"""

import hashlib
import re
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
        content_hash = hashlib.md5(chunk_text.encode("utf-8"), usedforsecurity=False).hexdigest()
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

    # Update total_chunks and re-index after filtering empty chunks
    actual_total = len(result)
    for new_idx, chunk in enumerate(result):
        chunk["metadata"]["chunk_index"] = new_idx
        chunk["metadata"]["total_chunks"] = actual_total

    return result


# --- Semantic chunking ---

# Regex for Chinese/English sentence boundaries
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？.!?])\s*")


def _split_by_headings(text: str) -> list[tuple[str, str | None]]:
    """Split text into (content, heading) pairs by markdown headings.

    Returns a list of (section_text, heading_title_or_None) tuples.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(text))

    if not matches:
        return [(text, None)]

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        # Preserve heading level: "## Introduction" instead of just "Introduction"
        heading = f"{m.group(1)} {m.group(2).strip()}"
        if section_text:
            sections.append((section_text, heading))

    # Also include text before the first heading
    if matches and matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.insert(0, (preamble, None))

    return sections


def _split_by_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by double newlines, merging short ones."""
    raw = re.split(r"\n\s*\n", text)
    # Filter out empty paragraphs
    return [p.strip() for p in raw if p.strip()]


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph into smaller pieces at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_BOUNDARY.split(text)
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= max_chars:
            current = (current + " " + sent).strip() if current else sent
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds max_chars, fall back to character split
            if len(sent) > max_chars:
                for i in range(0, len(sent), max_chars):
                    chunks.append(sent[i:i + max_chars])
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks


def _merge_short_chunks(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    """Merge adjacent short chunks together to avoid tiny fragments."""
    if not chunks:
        return chunks

    merged = []
    buffer = ""

    for chunk in chunks:
        combined = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        # If this chunk is below min_chars, always merge it with the buffer
        # even if it pushes the buffer over max_chars.  This prevents tiny
        # fragments (e.g. stray 10-char sentence fragments) from surviving.
        if len(combined) <= max_chars or (buffer and len(chunk) < min_chars):
            buffer = combined
        else:
            if buffer:
                merged.append(buffer)
            buffer = chunk

    if buffer:
        merged.append(buffer)

    return merged


def chunk_document_semantic(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """Split a document into semantically coherent chunks.

    Splitting order: markdown headings → paragraphs → sentences → characters.
    This preserves section context and avoids breaking sentences mid-way.

    Args:
        text: Document text content.
        metadata: Base metadata to attach to every chunk.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Overlap between adjacent chunks (applied as trailing
            context from the previous chunk).
        min_chunk_size: Minimum chunk size; smaller chunks are merged.

    Returns:
        List of chunk dicts with 'content' and 'metadata' keys.
    """
    if metadata is None:
        metadata = {}

    all_chunks: list[str] = []

    # Phase 1: Split by headings
    sections = _split_by_headings(text)

    for section_text, heading in sections:
        # Phase 2: Split each section by paragraphs
        paragraphs = _split_by_paragraphs(section_text)
        section_chunks: list[str] = []

        for para in paragraphs:
            # Phase 3: Split long paragraphs at sentence boundaries
            para_chunks = _split_long_paragraph(para, chunk_size)
            section_chunks.extend(para_chunks)

        # Phase 4: Merge short chunks within this section
        section_chunks = _merge_short_chunks(section_chunks, min_chunk_size, chunk_size)

        # Prepend heading context to chunks in this section
        if heading:
            section_chunks = [_prepend_heading(c, heading) for c in section_chunks]

        all_chunks.extend(section_chunks)

    # Phase 5: Add overlap context from previous chunk
    all_chunks = _add_overlap(all_chunks, chunk_overlap)

    # Build result with metadata
    total = len(all_chunks)
    result = []
    for i, chunk_text in enumerate(all_chunks):
        if not chunk_text.strip():
            continue
        content_hash = hashlib.md5(chunk_text.encode("utf-8"), usedforsecurity=False).hexdigest()
        chunk_meta = {
            **metadata,
            "chunk_index": i,
            "total_chunks": total,
            "chunk_id": content_hash[:12],
            "content_hash": content_hash,
            "chunk_size": len(chunk_text),
            "chunk_method": "semantic",
        }
        result.append({
            "content": chunk_text,
            "metadata": chunk_meta,
        })

    actual_total = len(result)
    for new_idx, chunk in enumerate(result):
        chunk["metadata"]["chunk_index"] = new_idx
        chunk["metadata"]["total_chunks"] = actual_total

    return result


def _prepend_heading(chunk: str, heading: str) -> str:
    """Prepend heading context to a chunk if not already present."""
    # Only skip if the heading is already at the start of the chunk
    if chunk.startswith(heading):
        return chunk
    return f"{heading}\n{chunk}"


def _add_overlap(chunks: list[str], overlap_chars: int) -> list[str]:
    """Add trailing context from the previous chunk as overlap prefix."""
    if not chunks or overlap_chars <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        # Take the last `overlap_chars` characters from the previous chunk
        overlap_text = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        result.append(overlap_text + "\n" + chunks[i])

    return result