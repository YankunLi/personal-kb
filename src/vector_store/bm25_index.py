"""BM25 sparse index for keyword-based retrieval.

Uses jieba for Chinese tokenization and rank_bm25 for scoring.
The index is persisted to disk via pickle.
"""

import pickle
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 keyword search index with Chinese tokenization support.

    Stores a corpus of document chunks and provides sparse retrieval.
    """

    def __init__(self, index_dir: str = "data/bm25"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._corpus: list[str] = []
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._loaded_kb: str | None = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Chinese text using jieba."""
        return list(jieba.cut(text))

    def build(self, chunks: list[dict[str, Any]]):
        """Build BM25 index from a list of chunks.

        Args:
            chunks: List of chunk dicts with 'content' and 'metadata' keys.
        """
        self._corpus = []
        self._chunk_ids = []
        self._metadatas = []

        tokenized_corpus = []
        for chunk in chunks:
            content = chunk.get("content", "")
            self._corpus.append(content)
            self._chunk_ids.append(chunk["metadata"].get("chunk_id", ""))
            self._metadatas.append(chunk["metadata"])
            tokenized_corpus.append(self._tokenize(content))

        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def add_chunks(self, chunks: list[dict[str, Any]]):
        """Add new chunks to the existing BM25 index (rebuilds entirely).

        For a personal KB with few documents, rebuilding is fast enough.
        """
        all_chunks = [
            {"content": c, "metadata": dict(m)}
            for c, m in zip(self._corpus, self._metadatas)
        ]
        all_chunks.extend(chunks)
        self.build(all_chunks)

    def search(
        self, query: str, top_k: int = 50
    ) -> list[dict[str, Any]]:
        """Search the BM25 index for relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with 'id', 'content', 'metadata', 'score'.
        """
        if self._bm25 is None or not self._corpus:
            return []

        if not query or not query.strip():
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self._chunk_ids[idx],
                "content": self._corpus[idx],
                "metadata": self._metadatas[idx],
                "score": float(scores[idx]),
            })

        return results

    def save(self, kb_name: str):
        """Persist the BM25 index to disk."""
        kb_dir = self.index_dir / kb_name
        kb_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "corpus": self._corpus,
            "chunk_ids": self._chunk_ids,
            "metadatas": self._metadatas,
        }
        with open(kb_dir / "bm25.pkl", "wb") as f:
            pickle.dump(data, f)

        self._loaded_kb = kb_name

    def load(self, kb_name: str) -> bool:
        """Load the BM25 index from disk. Returns True if load succeeded."""
        kb_dir = self.index_dir / kb_name
        index_path = kb_dir / "bm25.pkl"

        if not index_path.exists():
            return False

        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load BM25 index for '%s': %s", kb_name, e
            )
            return False

        self._corpus = data["corpus"]
        self._chunk_ids = data["chunk_ids"]
        self._metadatas = data["metadatas"]
        self._loaded_kb = kb_name

        # Rebuild BM25 object from tokenized corpus
        tokenized_corpus = [self._tokenize(c) for c in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        return True

    def reset(self):
        """Clear in-memory state without touching disk."""
        self._corpus = []
        self._chunk_ids = []
        self._metadatas = []
        self._bm25 = None
        self._loaded_kb = None

    def delete(self, kb_name: str):
        """Delete the BM25 index for a knowledge base."""
        kb_dir = self.index_dir / kb_name
        index_path = kb_dir / "bm25.pkl"
        if index_path.exists():
            index_path.unlink()

        # Remove empty parent directory
        try:
            kb_dir.rmdir()
        except OSError:
            pass  # Directory not empty or doesn't exist

        # Clear in-memory state only if the deleted KB is currently loaded
        if self._loaded_kb == kb_name:
            self.reset()

    def count(self) -> int:
        """Return the number of documents in the index."""
        return len(self._corpus)