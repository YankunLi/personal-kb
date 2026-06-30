"""BM25 sparse index for keyword-based retrieval.

Uses jieba for Chinese tokenization and rank_bm25 for scoring.
The index is persisted to disk via pickle.
"""

import os
import pickle
import re
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


# Characters allowed in KB names for path traversal prevention
_KB_NAME_RE = re.compile(r"^[a-zA-Z0-9\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff_.-]*$")


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
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._loaded_kb: str | None = None

    @staticmethod
    def _validate_kb_name(kb_name: str) -> str:
        """Validate kb_name to prevent path traversal.

        Returns the name if valid, raises ValueError otherwise.
        """
        if not kb_name or not _KB_NAME_RE.match(kb_name):
            raise ValueError(
                f"Invalid KB name: {kb_name!r}. "
                "Must start with alphanumeric or Chinese char, "
                "contain only alphanumeric, Chinese, underscores, dots, or hyphens."
            )
        if ".." in kb_name:
            raise ValueError(f"Path traversal detected in KB name: {kb_name!r}")
        return kb_name

    def _kb_dir(self, kb_name: str) -> Path:
        """Get the directory for a KB, validating the name first."""
        self._validate_kb_name(kb_name)
        return self.index_dir / kb_name

    def has_index(self, kb_name: str) -> bool:
        """Check if a BM25 index exists for a KB (validates kb_name first).

        Returns True immediately if the KB is already loaded in memory.
        """
        if self._loaded_kb == kb_name:
            return True
        return (self._kb_dir(kb_name) / "bm25.pkl").exists()

    def copy_index(self, source_kb: str, target_kb: str):
        """Copy a BM25 index from one KB to another (validates both names)."""
        src_dir = self._kb_dir(source_kb)
        dst_dir = self._kb_dir(target_kb)
        src_file = src_dir / "bm25.pkl"
        if src_file.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src_file, dst_dir / "bm25.pkl")

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
        self._tokenized_corpus = []

        for chunk in chunks:
            content = chunk.get("content", "")
            self._corpus.append(content)
            self._chunk_ids.append(chunk["metadata"].get("chunk_id", ""))
            self._metadatas.append(chunk["metadata"])
            self._tokenized_corpus.append(self._tokenize(content))

        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        self._loaded_kb = None  # Built from scratch, not yet persisted

    def add_chunks(self, chunks: list[dict[str, Any]]):
        """Add new chunks to the existing BM25 index.

        Only tokenizes the new chunks, avoiding costly re-tokenization of the
        entire corpus. The BM25Okapi object is rebuilt from the extended
        tokenized corpus (which is fast).
        """
        for chunk in chunks:
            content = chunk.get("content", "")
            self._corpus.append(content)
            self._chunk_ids.append(chunk["metadata"].get("chunk_id", ""))
            self._metadatas.append(chunk["metadata"])
            self._tokenized_corpus.append(self._tokenize(content))

        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def search(
        self, query: str, top_k: int = 50, kb_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Search the BM25 index for relevant chunks.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            kb_name: Expected KB name. If set, verifies the loaded index
                matches this KB to prevent serving stale state under
                concurrent access.

        Returns:
            List of result dicts with 'id', 'content', 'metadata', 'score'.
        """
        if self._bm25 is None or not self._corpus:
            return []

        if not query or not query.strip():
            return []

        # Guard: verify the loaded index matches the expected KB to prevent
        # serving stale state when BM25Index is shared across concurrent
        # operations.
        if kb_name is not None and self._loaded_kb is not None and self._loaded_kb != kb_name:
            import logging
            logging.getLogger(__name__).warning(
                "BM25 index mismatch: loaded '%s' but expected '%s'. "
                "Call load('%s') first.",
                self._loaded_kb, kb_name, kb_name,
            )
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices in descending score order
        top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            # Skip non-matching documents. BM25 scores can be 0 (no term
            # overlap) or slightly negative (IDF<0 when a term appears in
            # >50% of docs); such docs are not real matches and pollute
            # retrieval if returned.
            if scores[idx] <= 0:
                break  # scores are sorted descending; rest are <= 0
            results.append({
                "id": self._chunk_ids[idx],
                "content": self._corpus[idx],
                "metadata": self._metadatas[idx],
                "score": float(scores[idx]),
            })
            if len(results) >= top_k:
                break

        return results

    def save(self, kb_name: str):
        """Persist the BM25 index to disk."""
        kb_dir = self._kb_dir(kb_name)
        kb_dir.mkdir(parents=True, exist_ok=True)

        # Guard: prevent saving under a different KB name than what is loaded.
        if self._loaded_kb is not None and self._loaded_kb != kb_name:
            raise RuntimeError(
                f"Cannot save BM25 index: loaded KB '{self._loaded_kb}' "
                f"does not match target '{kb_name}'. Load the correct KB first."
            )

        data = {
            "corpus": self._corpus,
            "chunk_ids": self._chunk_ids,
            "metadatas": self._metadatas,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(kb_dir / "bm25.pkl", "wb") as f:
            pickle.dump(data, f)

        self._loaded_kb = kb_name

    def load(self, kb_name: str) -> bool:
        """Load the BM25 index from disk. Returns True if load succeeded.

        If the requested KB is already loaded in memory, skips the disk read
        to avoid redundant I/O on repeated queries to the same KB (e.g. during
        a chat session).
        """
        if self._loaded_kb == kb_name:
            return True

        kb_dir = self._kb_dir(kb_name)
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

        # Restore cached tokenized corpus if available, otherwise tokenize
        if "tokenized_corpus" in data:
            self._tokenized_corpus = data["tokenized_corpus"]
        else:
            self._tokenized_corpus = [self._tokenize(c) for c in self._corpus]

        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        # Set _loaded_kb AFTER BM25 rebuild so a crash leaves a clean state
        self._loaded_kb = kb_name

        return True

    def reset(self):
        """Clear in-memory state without touching disk."""
        self._corpus = []
        self._chunk_ids = []
        self._metadatas = []
        self._tokenized_corpus = []
        self._bm25 = None
        self._loaded_kb = None

    def delete(self, kb_name: str):
        """Delete the BM25 index for a knowledge base."""
        kb_dir = self._kb_dir(kb_name)
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