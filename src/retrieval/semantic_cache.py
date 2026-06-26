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

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_product = np.linalg.norm(a) * np.linalg.norm(b)
        if norm_product == 0:
            return 0.0
        return float(np.dot(a, b) / norm_product)

    def get(self, query_embedding: np.ndarray, kb_name: str = "default") -> dict[str, Any] | None:
        """Check if a similar query for the same KB is cached.

        Returns:
            Dict with 'answer', 'sources' if found, None otherwise.
        """
        for emb, answer, sources, cached_kb, ts in self._entries:
            if cached_kb != kb_name:
                continue
            if self._cosine_sim(query_embedding, emb) > self.threshold:
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