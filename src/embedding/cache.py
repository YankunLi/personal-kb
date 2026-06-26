"""Disk-based embedding cache with LRU eviction.

Avoids re-encoding the same text, saving CPU/GPU time.
Cache key: MD5(text + model_name + model_revision).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingCache:
    """Disk-based LRU cache for embedding vectors."""

    def __init__(self, cache_dir: str = "data/embedding_cache", max_entries: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict] = self._load_index()

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(self._index_path, "w") as f:
            json.dump(self._index, f)

    def _cache_key(self, text: str, model_name: str, model_revision: str) -> str:
        raw = f"{text}|{model_name}|{model_revision}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def get(
        self, text: str, model_name: str, model_revision: str
    ) -> np.ndarray | None:
        """Get cached embedding if available.

        Returns:
            Numpy array if cached, None otherwise.
        """
        key = self._cache_key(text, model_name, model_revision)
        cache_path = self._cache_path(key)

        if key in self._index and cache_path.exists():
            # Update access time for LRU
            self._index[key]["last_access"] = _now()
            self._save_index()
            return np.load(cache_path)

        return None

    def set(
        self,
        text: str,
        model_name: str,
        model_revision: str,
        embedding: np.ndarray,
    ):
        """Store embedding in cache."""
        key = self._cache_key(text, model_name, model_revision)
        cache_path = self._cache_path(key)

        np.save(cache_path, embedding)
        self._index[key] = {
            "text_hash": hashlib.md5(text.encode("utf-8")).hexdigest()[:12],
            "model_name": model_name,
            "model_revision": model_revision,
            "last_access": _now(),
        }

        # LRU eviction
        if len(self._index) > self.max_entries:
            self._evict()

        self._save_index()

    def _evict(self):
        """Evict the least recently used entry."""
        oldest_key = min(self._index, key=lambda k: self._index[k]["last_access"])
        cache_path = self._cache_path(oldest_key)
        if cache_path.exists():
            cache_path.unlink()
        del self._index[oldest_key]

    def clear(self):
        """Clear all cached embeddings."""
        for key in list(self._index.keys()):
            cache_path = self._cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
        self._index.clear()
        self._save_index()

    def __len__(self) -> int:
        return len(self._index)


def _now() -> float:
    import time
    return time.time()