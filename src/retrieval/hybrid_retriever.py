"""Hybrid retriever combining Dense (vector) + Sparse (BM25) search with RRF fusion.

Based on Article A: "工业级的答案是混合检索（Hybrid Search）"
RRF (Reciprocal Rank Fusion): score(doc) = Σ 1/(k + rank_i(doc)), k defaults to 60.
"""

from typing import Any

import numpy as np

from src.embedding.embedder import Embedder
from src.vector_store.bm25_index import BM25Index
from src.vector_store.chroma_store import ChromaStore


class HybridRetriever:
    """Hybrid retriever: Dense + Sparse + RRF fusion + optional metadata filtering."""

    def __init__(
        self,
        embedder: Embedder,
        chroma_store: ChromaStore,
        bm25_index: BM25Index,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.chroma = chroma_store
        self.bm25 = bm25_index
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.rrf_k = rrf_k

    def dense_search(
        self, kb_name: str, query: str, top_k: int | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search via ChromaDB."""
        if top_k is None:
            top_k = self.dense_top_k
        if query_embedding is None:
            query_embedding = self.embedder.encode_query(query)
        return self.chroma.query(kb_name, query_embedding, top_k=top_k)

    def sparse_search(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """BM25 keyword search."""
        if top_k is None:
            top_k = self.sparse_top_k
        return self.bm25.search(query, top_k=top_k)

    def hybrid_search(
        self,
        kb_name: str,
        query: str,
        top_k: int | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search with RRF fusion of dense and sparse results.

        Args:
            kb_name: Knowledge base name.
            query: User query string.
            top_k: Number of results after fusion.
            query_embedding: Pre-computed query embedding for dense search.

        Returns:
            List of result dicts with RRF fusion scores.
        """
        if top_k is None:
            top_k = self.dense_top_k

        dense_results = self.dense_search(kb_name, query, top_k=self.dense_top_k, query_embedding=query_embedding)
        sparse_results = self.sparse_search(query, top_k=self.sparse_top_k)

        if not dense_results and not sparse_results:
            return []

        if not dense_results:
            return sparse_results[:top_k]
        if not sparse_results:
            return dense_results[:top_k]

        # RRF fusion
        rrf_scores: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(dense_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {**result, "rrf_score": rrf_score}
            else:
                rrf_scores[doc_id]["rrf_score"] += rrf_score

        for rank, result in enumerate(sparse_results):
            doc_id = result["id"]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {**result, "rrf_score": rrf_score}
            else:
                rrf_scores[doc_id]["rrf_score"] += rrf_score

        # Sort by RRF score descending
        sorted_results = sorted(
            rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True
        )
        return sorted_results[:top_k]

    def retrieve(
        self,
        kb_name: str,
        query: str,
        top_k: int | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Main retrieval entry point: hybrid search → top-k results.

        Args:
            kb_name: Knowledge base name.
            query: User query string.
            top_k: Number of results after fusion.
            query_embedding: Pre-computed query embedding. If provided, skips
                re-encoding the query for dense search.
        """
        return self.hybrid_search(kb_name, query, top_k=top_k, query_embedding=query_embedding)