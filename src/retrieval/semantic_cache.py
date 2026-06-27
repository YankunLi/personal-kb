"""Semantic cache for similar queries, per knowledge base.

Based on Article A: "语义缓存：减少 30-50% LLM 调用"
Caches (query_embedding, answer, sources, kb_name) pairs with cosine similarity matching.
"""

import time
from typing import Any

import numpy as np


class SemanticCache:
    """Cosine-similarity-based query cache, keyed by KB name.

    When a new query's embedding is similar enough (> threshold) to a cached
    query for the same KB, the cached answer is returned, skipping LLM generation.
    """

    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000):
        self.threshold = similarity_threshold
        self.max_size = max_size
        # cache: list of (embedding, answer, sources, kb_name, timestamp)
        self._entries: list[tuple[np.ndarray, str, list, str, float]] = []

    def get(self, query_embedding: np.ndarray, kb_name: str = "default") -> dict[str, Any] | None:
        """Check if a similar query for the same KB is cached.

        Uses vectorized dot product to compute all cosine similarities at once
        instead of a linear scan with per-entry computation.

        Returns:
            Dict with 'answer', 'sources' if found, None otherwise.
        """
        # Filter entries for the target KB
        kb_entries = [(i, e) for i, e in enumerate(self._entries) if e[3] == kb_name]
        if not kb_entries:
            return None

        indices = [i for i, _ in kb_entries]
        embeddings = np.stack([e[0] for _, e in kb_entries])

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return None

        # Vectorized cosine similarity: (N x D) @ (D,) / (N,) / scalar
        norms = np.linalg.norm(embeddings, axis=1)
        # Guard against zero-norm embeddings (should not happen with normalized
        # embeddings, but protects against corrupted cache entries)
        valid = norms > 0
        if not valid.any():
            return None
        sims = np.empty(len(embeddings))
        sims[~valid] = -1.0  # Zero-norm entries cannot match
        sims[valid] = np.dot(embeddings[valid], query_embedding) / (norms[valid] * query_norm)

        best_local = int(np.argmax(sims))
        if sims[best_local] > self.threshold:
            _, answer, sources, _, _ = self._entries[indices[best_local]]
            return {"answer": answer, "sources": sources}

        return None

    def set(
        self,
        query_embedding: np.ndarray,
        answer: str,
        sources: list[dict[str, Any]],
        kb_name: str = "default",
    ):
        """Store a query-answer pair in the cache, keyed by KB."""
        self._entries.append((query_embedding.copy(), answer, sources, kb_name, time.time()))

        # LRU eviction: remove oldest entry
        if len(self._entries) > self.max_size:
            oldest = min(self._entries, key=lambda x: x[4])
            self._entries.remove(oldest)

    def clear(self, kb_name: str | None = None):
        """Clear cached entries. If kb_name is given, only clear that KB's entries."""
        if kb_name is None:
            self._entries.clear()
        else:
            self._entries = [e for e in self._entries if e[3] != kb_name]

    def __len__(self) -> int:
        return len(self._entries)