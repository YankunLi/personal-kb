"""Reranker using BGE-Reranker cross-encoder for fine-grained scoring.

Based on Article A: "重排序可以让答案质量提升 15-30%"
Based on Article B: "Re-rank 能再 +7-10%"

Graceful degradation: if the reranker model fails to load or compute scores,
the system falls back to raw retrieval scores instead of crashing.
"""

import logging
from typing import Any
import threading

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for refining top-k retrieval results.

    Uses BGE-Reranker-v2-m3 (BAAI) for Chinese-optimized reranking.
    Lazily loaded to avoid loading the model if reranking is not used.
    Falls back to raw retrieval scores on model failure.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", model_revision: str = "main"):
        self.model_name = model_name
        self.model_revision = model_revision
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._load_failed = False

    @property
    def model(self):
        """Lazy-load the reranker model (thread-safe).

        Returns None if the model failed to load (graceful degradation).
        """
        if self._model is None and not self._load_failed:
            with self._model_lock:
                if self._model is None and not self._load_failed:
                    try:
                        from FlagEmbedding import FlagReranker
                        self._model = FlagReranker(
                            self.model_name,
                            use_fp16=True,
                            revision=self.model_revision,
                        )
                    except Exception as e:
                        self._load_failed = True
                        logger.warning(
                            "Failed to load reranker model '%s': %s. "
                            "Falling back to raw retrieval scores.",
                            self.model_name, e,
                        )
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents using cross-encoder scoring.

        Falls back to raw retrieval scores if the reranker model is unavailable.

        Args:
            query: User query string.
            documents: List of document dicts with 'content' key.
            top_n: Number of results to return after reranking.

        Returns:
            Top-n documents sorted by reranker score descending.
        """
        # Work on a shallow copy to avoid mutating caller's documents
        docs = [{**d} for d in documents]
        if not docs:
            return []

        # Ensure all documents have a baseline rerank_score
        for doc in docs:
            if "rerank_score" not in doc or doc.get("rerank_score") is None:
                score = doc.get("score")
                doc["rerank_score"] = float(score) if score is not None else 0.0

        model = self.model
        if model is None:
            return sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_n]

        try:
            pairs = [[query, doc["content"]] for doc in docs]
            scores = model.compute_score(pairs, normalize=True)
        except Exception as e:
            logger.warning(
                "Reranker inference failed: %s. Falling back to raw retrieval scores.", e
            )
            return sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_n]

        # compute_score returns a float for single pair, list for multiple
        if isinstance(scores, float):
            scores = [scores]

        if len(scores) != len(docs):
            raise RuntimeError(
                f"Reranker returned {len(scores)} scores for {len(docs)} documents"
            )

        # Attach reranker scores
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        # Sort by score descending
        reranked = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_n]