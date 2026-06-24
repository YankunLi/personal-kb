"""Semantic cache for similar queries.

Based on Article A: "语义缓存：减少 30-50% LLM 调用"
Caches (query_embedding, answer) pairs with cosine similarity matching.
"""

import time
from typing import Any

import numpy as np


class SemanticCache:
    """Cosine-similarity-based query cache.

    When a new query's embedding is similar enough (> threshold) to a cached
    query, the cached answer is returned directly, skipping LLM generation.
    """

    def __init__(self, similarity_threshold: float = 0.95, max_size: int = 1000):
        self.threshold = similarity_threshold
        self.max_size = max_size
        # cache: list of (embedding, answer, sources, timestamp)
        self._entries: list[tuple[np.ndarray, str, list, float]] = []

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query_embedding: np.ndarray) -> dict[str, Any] | None:
        """Check if a similar query is cached.

        Returns:
            Dict with 'answer', 'sources' if found, None otherwise.
        """
        for emb, answer, sources, ts in self._entries:
            if self._cosine_sim(query_embedding, emb) > self.threshold:
                return {"answer": answer, "sources": sources}
        return None

    def set(
        self,
        query_embedding: np.ndarray,
        answer: str,
        sources: list[dict[str, Any]],
    ):
        """Store a query-answer pair in the cache."""
        self._entries.append((query_embedding.copy(), answer, sources, time.time()))

        # LRU eviction: remove oldest entry
        if len(self._entries) > self.max_size:
            oldest = min(self._entries, key=lambda x: x[3])
            self._entries.remove(oldest)

    def clear(self):
        """Clear all cached entries."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)