"""Embedding model wrapper with version pinning.

Uses BGE-M3 (BAAI) for Chinese-optimized embeddings.
CRITICAL: Version pinning prevents embedding drift (see Article B's real accident).

Set HF_ENDPOINT=https://hf-mirror.com to use a mirror for downloading models.
"""

import os
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding.cache import EmbeddingCache


class Embedder:
    """Embedding model wrapper with version pinning and query instruction support.

    BGE models require a query prefix for asymmetric tasks (query vs document).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        model_revision: str = "main",
        normalize: bool = True,
        query_instruction: str = "为这个句子生成表示以用于检索相关文章：",
        batch_size: int = 32,
        cache_dir: str = "data/embedding_cache",
        cache_max_entries: int = 10000,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.batch_size = batch_size

        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._cache = EmbeddingCache(cache_dir=cache_dir, max_entries=cache_max_entries)

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use (thread-safe)."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = SentenceTransformer(
                        self.model_name,
                        revision=self.model_revision,
                        trust_remote_code=True,
                    )
        return self._model

    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        return self.model.get_embedding_dimension()

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query with the query instruction prefix.

        Args:
            query: User search query string.

        Returns:
            Normalized embedding vector as numpy array.
        """
        if self.query_instruction:
            query = self.query_instruction + query
        embedding = self.model.encode(
            query,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def encode_documents(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        """Encode document texts in batch.

        Uses disk cache to avoid re-encoding previously seen texts.

        Args:
            texts: List of document chunk texts.
            show_progress: Whether to show a progress bar.

        Returns:
            List of normalized embedding vectors.
        """
        result: list[np.ndarray] = [None] * len(texts)  # type: ignore
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text, self.model_name, self.model_revision)
            if cached is not None:
                result[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
                batch_size=self.batch_size,
            )
            for i, emb in zip(uncached_indices, embeddings):
                arr = np.array(emb, dtype=np.float32)
                result[i] = arr
                self._cache.set(texts[i], self.model_name, self.model_revision, arr)

        # Flush cache to disk after batch
        self._cache.flush()
        return result

    def get_version_info(self) -> dict:
        """Return version info for metadata tracking."""
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "dimensions": self.dim,
        }