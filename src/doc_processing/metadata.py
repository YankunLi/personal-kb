"""Metadata enrichment for document chunks.

Each chunk carries full provenance metadata for source tracking,
as recommended by both articles.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def extract_section(text: str) -> str | None:
    """Extract the nearest heading from chunk text as section name.

    Looks for markdown-style headings (# Title) or common patterns.
    """
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            # Remove leading # and whitespace
            section = re.sub(r"^#+\s*", "", line).strip()
            if section:
                return section
    return None


def build_base_metadata(
    file_path: str | Path,
    kb_name: str = "default",
    embedding_model: str = "BAAI/bge-m3",
    embedding_dim: int = 1024,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build base metadata dict for a file being processed.

    This metadata is shared across all chunks from the same file.

    Args:
        file_path: Path to the source document.
        kb_name: Name of the knowledge base.
        embedding_model: Embedding model name (version pinned).
        embedding_dim: Embedding vector dimensions.
        extra: Additional metadata to merge.

    Returns:
        Dict of base metadata.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    meta = {
        "source_file": str(file_path),
        "source_file_basename": file_path.name,
        "file_type": ext,
        "kb_name": kb_name,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }

    if extra:
        meta.update(extra)

    return meta


def enrich_chunk_metadata(
    chunk: dict[str, Any],
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Enrich a single chunk with metadata.

    Merges base metadata with chunk-specific metadata and extracts
    section information from the chunk content.

    Args:
        chunk: Chunk dict with 'content' and 'metadata' keys.
        base_metadata: Base metadata from the source file.

    Returns:
        Chunk dict with enriched metadata.
    """
    section = extract_section(chunk["content"])
    enriched = {
        **base_metadata,
        **chunk["metadata"],
        "section": section,
    }
    chunk["metadata"] = enriched
    return chunk


def enrich_chunks(
    chunks: list[dict[str, Any]],
    base_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enrich all chunks with metadata.

    Args:
        chunks: List of chunk dicts.
        base_metadata: Base metadata from the source file.

    Returns:
        List of chunks with enriched metadata.
    """
    return [enrich_chunk_metadata(c, base_metadata) for c in chunks]