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
    """

    def __init__(self):
        self._seen_hashes: set[str] = set()

    def reset(self):
        """Reset the seen hashes set (for a new import batch)."""
        self._seen_hashes.clear()

    def compute_hash(self, content: str) -> str:
        """Compute MD5 hash of chunk content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def is_duplicate(self, content: str) -> bool:
        """Check if a chunk's content has been seen before.

        Args:
            content: Chunk text content.

        Returns:
            True if the content hash has been seen before.
        """
        h = self.compute_hash(content)
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

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
            content = chunk.get("content", "")
            if self.is_duplicate(content):
                duplicates += 1
            else:
                unique.append(chunk)

        return unique, duplicates