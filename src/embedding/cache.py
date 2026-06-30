"""Disk-based embedding cache with LRU eviction.

Avoids re-encoding the same text, saving CPU/GPU time.
Cache key: MD5(text + model_name + model_revision).
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Disk-based LRU cache for embedding vectors.

    Thread-safe: all public methods acquire a re-entrant lock so the cache
    can be safely shared across threads (e.g. via the Pipeline singleton).
    """

    def __init__(self, cache_dir: str = "data/embedding_cache", max_entries: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._index_path = self.cache_dir / "index.json"
        self._lock = threading.RLock()
        self._index: dict[str, dict] = self._load_index()
        self._dirty_since_save: int = 0

    def _load_index(self) -> dict[str, dict]:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Embedding cache index corrupted, rebuilding from disk.")
        # Recover orphaned .npy files not in the index
        return self._recover_orphans()

    def _recover_orphans(self) -> dict[str, dict]:
        """Scan cache directory for orphaned .npy files and rebuild minimal index."""
        recovered = {}
        for npy_file in self.cache_dir.glob("*.npy"):
            key = npy_file.stem
            if len(key) == 32:  # MD5 hex digest
                recovered[key] = {
                    "text_hash": "",
                    "model_name": "",
                    "model_revision": "",
                    "last_access": npy_file.stat().st_mtime,
                }
        if recovered:
            logger.warning("Recovered %d orphaned cache entries.", len(recovered))
        return recovered

    def _save_index(self):
        """Persist the index atomically via tmp + rename.

        This prevents index corruption if the process crashes mid-write.
        """
        tmp_path = self._index_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._index, f)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self._index_path)

    def _cache_key(self, text: str, model_name: str, model_revision: str) -> str:
        raw = f"{text}|{model_name}|{model_revision}"
        return hashlib.md5(raw.encode("utf-8"), usedforsecurity=False).hexdigest()

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

        with self._lock:
            if key in self._index and cache_path.exists():
                # Update access time for LRU
                self._index[key]["last_access"] = _now()
                self._dirty_since_save += 1
                if self._dirty_since_save >= 100:
                    self._save_index()
                    self._dirty_since_save = 0
                try:
                    return np.load(cache_path)
                except (ValueError, OSError):
                    logger.warning("Corrupted embedding cache entry: %s", key)
                    cache_path.unlink(missing_ok=True)
                    self._index.pop(key, None)
                    self._save_index()
                    return None

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

        with self._lock:
            np.save(cache_path, embedding)
            self._index[key] = {
                "text_hash": hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:12],
                "model_name": model_name,
                "model_revision": model_revision,
                "last_access": _now(),
            }

            # LRU eviction
            if len(self._index) > self.max_entries:
                self._evict()

            self._dirty_since_save += 1
            if self._dirty_since_save >= 100:
                self._save_index()
                self._dirty_since_save = 0

    def flush(self):
        """Persist the index to disk immediately."""
        with self._lock:
            if self._dirty_since_save > 0:
                self._save_index()
                self._dirty_since_save = 0

    def _evict(self):
        """Evict the least recently used entry."""
        oldest_key = min(self._index, key=lambda k: self._index[k]["last_access"])
        cache_path = self._cache_path(oldest_key)
        if cache_path.exists():
            cache_path.unlink()
        del self._index[oldest_key]

    def clear(self):
        """Clear all cached embeddings."""
        with self._lock:
            for key in list(self._index.keys()):
                cache_path = self._cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            self._index.clear()
            self._save_index()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)


def _now() -> float:
    return time.time()
