"""Reranker using BGE-Reranker cross-encoder for fine-grained scoring.

Based on Article A: "重排序可以让答案质量提升 15-30%"
Based on Article B: "Re-rank 能再 +7-10%"
"""

from typing import Any


class Reranker:
    """Cross-encoder reranker for refining top-k retrieval results.

    Uses BGE-Reranker-v2-m3 (BAAI) for Chinese-optimized reranking.
    Lazily loaded to avoid loading the model if reranking is not used.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self):
        """Lazy-load the reranker model."""
        if self._model is None:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                use_fp16=True,
            )
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents using cross-encoder scoring.

        Args:
            query: User query string.
            documents: List of document dicts with 'content' key.
            top_n: Number of results to return after reranking.

        Returns:
            Top-n documents sorted by reranker score descending.
        """
        if len(documents) <= top_n:
            return documents

        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        # Attach reranker scores
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        # Sort by score descending
        reranked = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_n]