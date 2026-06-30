"""Content-hash based deduplication for document chunks.

Based on Article A: "去重不是可选项——知识库里经常有重复内容，
不去重会导致检索结果被同一信息占据 top-K。"
"""

import hashlib
from typing import Any


class ChunkDeduplicator:
    """Deduplicate chunks based on content hash (MD5).

    Two levels of deduplication:
    1. Intra-batch: within the same import batch
    2. Cross-batch: against previously stored chunks (via content_hash)

    Uses the content_hash from chunk metadata when available (set by the chunker),
    falling back to computing the hash from content.
    """

    def __init__(self):
        self._seen_hashes: set[str] = set()

    def reset(self):
        """Reset the seen hashes set (for a new import batch)."""
        self._seen_hashes.clear()

    def seed_hashes(self, hashes: set[str]):
        """Pre-seed the deduplicator with existing hashes for cross-batch dedup."""
        self._seen_hashes.update(hashes)

    def compute_hash(self, content: str) -> str:
        """Compute MD5 hash of chunk content."""
        return hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _get_hash(self, chunk: dict) -> str:
        """Get the content hash from chunk metadata, or compute it."""
        meta = chunk.get("metadata", {})
        h = meta.get("content_hash")
        if h:
            return h
        return self.compute_hash(chunk.get("content", ""))

    def check_and_register(self, content: str) -> bool:
        """Check if content is new, and register it if so.

        Returns True if the content is new (not seen before).
        """
        h = self.compute_hash(content)
        if h in self._seen_hashes:
            return False
        self._seen_hashes.add(h)
        return True

    def deduplicate(
        self,
        chunks: list[dict[str, Any]],
        existing_hashes: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Remove duplicate chunks from a batch.

        Args:
            chunks: List of chunk dicts with 'content' key.
            existing_hashes: Set of content hashes already in the vector store.

        Returns:
            Tuple of (deduplicated_chunks, duplicate_count).
        """
        if existing_hashes:
            self._seen_hashes.update(existing_hashes)

        unique = []
        duplicates = 0
        for chunk in chunks:
            h = self._get_hash(chunk)
            if h in self._seen_hashes:
                duplicates += 1
            else:
                self._seen_hashes.add(h)
                unique.append(chunk)

        return unique, duplicates