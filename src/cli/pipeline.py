"""Core pipeline orchestrating document import, retrieval, and generation."""

import time
from pathlib import Path
from typing import Any

from src.config.loader import AppConfig, load_config
from src.doc_processing.loader import load_documents
from src.doc_processing.chunker import chunk_document
from src.doc_processing.metadata import build_base_metadata, enrich_chunks
from src.doc_processing.deduplicator import ChunkDeduplicator
from src.embedding.embedder import Embedder
from src.vector_store.chroma_store import ChromaStore
from src.vector_store.bm25_index import BM25Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.semantic_cache import SemanticCache
from src.generation.llm_adapter import LLMAdapter
from src.generation.prompt_builder import build_messages
from src.generation.hallucination import detect_hallucination, get_hallucination_risk_level
from src.kb_manager.manager import KBManager
from src.source_tracking.tracker import (
    RAGResponse,
    SourceInfo,
    extract_sources_from_contexts,
    format_sources_output,
)


# Global pipeline instance (lazy initialized)
_pipeline: "Pipeline | None" = None


def get_pipeline() -> "Pipeline":
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = Pipeline(config)
    return _pipeline


class Pipeline:
    """Orchestrates the full RAG pipeline: import, search, chat."""

    def __init__(self, config: AppConfig | None = None):
        if config is None:
            config = load_config()
        self.config = config

        # Initialize components
        self.embedder = Embedder(
            model_name=config.embedding.model_name,
            model_revision=config.embedding.model_revision,
            normalize=config.embedding.normalize,
            query_instruction=config.embedding.query_instruction,
            batch_size=config.embedding.batch_size,
        )
        self.chroma = ChromaStore(persist_dir=config.paths.chroma_db)
        self.bm25 = BM25Index(index_dir=config.paths.bm25_index_dir)
        self.kb_manager = KBManager(
            registry_path=config.paths.kb_registry,
            chroma_store=self.chroma,
            bm25_index=self.bm25,
        )
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            chroma_store=self.chroma,
            bm25_index=self.bm25,
            dense_top_k=config.retrieval.dense_top_k,
            sparse_top_k=config.retrieval.sparse_top_k,
            hybrid_top_k=config.retrieval.hybrid_top_k,
            rrf_k=config.retrieval.rrf_k,
        )
        self.reranker = Reranker(model_name=config.retrieval.reranker_model)
        self.semantic_cache = SemanticCache(
            similarity_threshold=config.retrieval.semantic_cache["similarity_threshold"],
            max_size=config.retrieval.semantic_cache["max_size"],
        )
        self.deduplicator = ChunkDeduplicator()

    def _get_llm_adapter(self, provider_name: str | None = None) -> LLMAdapter:
        """Get LLM adapter for the specified or default provider."""
        name = provider_name or self.config.defaults.provider
        provider_config = self.config.llm.providers.get(name)
        if provider_config is None:
            raise ValueError(f"Unknown provider: {name}. Available: {list(self.config.llm.providers.keys())}")
        return LLMAdapter(
            provider=provider_config,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            max_retries=self.config.llm.retry.max_attempts,
            backoff_seconds=self.config.llm.retry.backoff_seconds,
        )

    def import_documents(
        self,
        path: str,
        kb_name: str = "default",
        recursive: bool = True,
        dry_run: bool = False,
        progress_callback=None,
    ) -> dict[str, Any]:
        """Import documents into a knowledge base.

        Args:
            path: File or directory path.
            kb_name: Target knowledge base name.
            recursive: Whether to recurse into subdirectories.
            dry_run: If True, show what would be imported without actually indexing.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dict with import statistics.
        """
        path = Path(path).expanduser()

        # Ensure KB exists
        if not self.kb_manager.exists(kb_name):
            self.kb_manager.create(kb_name)

        # Load existing hashes for cross-batch dedup
        existing_hashes = self.chroma.get_existing_hashes(kb_name)
        self.deduplicator.reset()

        total_files = 0
        total_chunks = 0
        total_duplicates = 0
        all_chunks = []

        for doc in load_documents(path, recursive=recursive):
            total_files += 1
            file_path = doc["file_path"]

            if progress_callback:
                progress_callback("parse", file_path, total_files)

            if dry_run:
                continue

            # Build base metadata
            base_meta = build_base_metadata(
                file_path=file_path,
                kb_name=kb_name,
                embedding_model=self.config.embedding.model_name,
                embedding_dim=self.config.embedding.dimensions,
            )

            # Chunk the document
            chunks = chunk_document(
                doc["text"],
                metadata=None,
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
                separators=self.config.chunking.separators,
            )

            # Enrich with metadata
            chunks = enrich_chunks(chunks, base_meta)

            # Deduplicate
            if self.config.chunking.enable_deduplication:
                chunks, dups = self.deduplicator.deduplicate(chunks, existing_hashes)
                total_duplicates += dups

            total_chunks += len(chunks)
            all_chunks.extend(chunks)

        if dry_run:
            return {
                "files": total_files,
                "chunks": 0,
                "duplicates": 0,
                "dry_run": True,
            }

        if not all_chunks:
            return {
                "files": total_files,
                "chunks": 0,
                "duplicates": total_duplicates,
            }

        if progress_callback:
            progress_callback("embed", "", 0)

        # Embed all chunks
        texts = [c["content"] for c in all_chunks]
        embeddings = self.embedder.encode_documents(texts, show_progress=False)

        if progress_callback:
            progress_callback("index", "", 0)

        # Store in ChromaDB
        self.chroma.add_chunks(kb_name, all_chunks, embeddings)

        # Rebuild BM25 index (load existing + add new)
        self.bm25.load(kb_name)
        self.bm25.add_chunks(all_chunks)
        self.bm25.save(kb_name)

        # Update KB stats
        existing = self.kb_manager.get(kb_name)
        self.kb_manager.update_stats(
            kb_name,
            chunk_count=self.chroma.count(kb_name),
            file_count=existing.file_count + total_files,
        )

        return {
            "files": total_files,
            "chunks": total_chunks,
            "duplicates": total_duplicates,
        }

    def search(
        self,
        query: str,
        kb_name: str = "default",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search knowledge base without LLM generation.

        Args:
            query: Search query.
            kb_name: Knowledge base name.
            top_k: Number of results to return.

        Returns:
            List of result dicts with content, metadata, scores.
        """
        # Load BM25 index for this KB
        self.bm25.load(kb_name)

        # Hybrid search
        results = self.retriever.retrieve(kb_name, query, top_k=top_k)

        # Rerank
        results = self.reranker.rerank(query, results, top_n=top_k)

        return results

    async def chat(
        self,
        query: str,
        kb_name: str = "default",
        provider_name: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
        stream: bool = True,
    ) -> RAGResponse:
        """Full RAG pipeline: search + generate with source citations.

        Args:
            query: User question.
            kb_name: Knowledge base name.
            provider_name: LLM provider override.
            chat_history: Previous conversation turns.
            stream: Whether to stream tokens.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        start_time = time.time()

        # Load BM25 index for this KB
        self.bm25.load(kb_name)

        # Compute query embedding once for cache + retrieval
        query_emb = self.embedder.encode_query(query)

        # Check semantic cache
        if self.config.retrieval.semantic_cache["enabled"]:
            cached = self.semantic_cache.get(query_emb)
            if cached:
                latency = (time.time() - start_time) * 1000
                return RAGResponse(
                    answer=cached["answer"],
                    sources=cached["sources"],
                    hallucination_risk="low",
                    latency_ms=latency,
                )

        # Hybrid search
        results = self.retriever.retrieve(kb_name, query, query_embedding=query_emb)

        if not results:
            latency = (time.time() - start_time) * 1000
            return RAGResponse(
                answer="抱歉，在当前知识库中没有找到相关信息。请尝试更换关键词或导入相关文档。",
                sources=[],
                hallucination_risk="low",
                latency_ms=latency,
            )

        # Rerank
        results = self.reranker.rerank(query, results, top_n=self.config.retrieval.rerank_top_n)

        # Build prompt
        messages = build_messages(query, results, chat_history=chat_history)

        # Generate
        llm = None
        try:
            llm = self._get_llm_adapter(provider_name)
            if stream:
                answer = ""
                async for token in llm.chat_stream(messages):
                    answer += token
            else:
                answer = await llm.chat(messages)
        finally:
            if llm is not None:
                await llm.close()

        # Hallucination check
        is_risk, overlap = detect_hallucination(
            answer, results,
            entity_overlap_threshold=self.config.hallucination.entity_overlap_threshold,
        )
        risk_level = get_hallucination_risk_level(overlap)

        # Extract sources
        sources = extract_sources_from_contexts(results)

        # Update semantic cache
        if self.config.retrieval.semantic_cache["enabled"]:
            self.semantic_cache.set(query_emb, answer, sources)

        latency = (time.time() - start_time) * 1000

        return RAGResponse(
            answer=answer,
            sources=sources,
            hallucination_risk=risk_level,
            latency_ms=latency,
        )

    async def chat_stream(
        self,
        query: str,
        kb_name: str = "default",
        provider_name: str | None = None,
        chat_history: list[dict[str, str]] | None = None,
    ):
        """Streaming RAG pipeline: yields tokens as they arrive.

        Yields:
            Dicts with 'type': 'token'|'sources'|'done'|'error'.
        """
        start_time = time.time()

        # Load BM25 index
        self.bm25.load(kb_name)

        # Compute query embedding once for cache + retrieval
        query_emb = self.embedder.encode_query(query)

        # Check semantic cache
        if self.config.retrieval.semantic_cache["enabled"]:
            cached = self.semantic_cache.get(query_emb)
            if cached:
                yield {"type": "answer", "content": cached["answer"]}
                yield {"type": "sources", "sources": cached["sources"]}
                yield {
                    "type": "done",
                    "hallucination_risk": "low",
                    "latency_ms": (time.time() - start_time) * 1000,
                }
                return

        # Hybrid search
        results = self.retriever.retrieve(kb_name, query, query_embedding=query_emb)

        if not results:
            yield {
                "type": "answer",
                "content": "抱歉，在当前知识库中没有找到相关信息。",
            }
            yield {"type": "sources", "sources": []}
            yield {"type": "done"}
            return

        # Rerank
        results = self.reranker.rerank(query, results, top_n=self.config.retrieval.rerank_top_n)

        # Build prompt
        messages = build_messages(query, results, chat_history=chat_history)

        # Generate streaming
        llm = None
        try:
            llm = self._get_llm_adapter(provider_name)
            full_answer = ""
            async for token in llm.chat_stream(messages):
                full_answer += token
                yield {"type": "token", "content": token}
        finally:
            if llm is not None:
                await llm.close()

        # Hallucination check
        is_risk, overlap = detect_hallucination(
            full_answer, results,
            entity_overlap_threshold=self.config.hallucination.entity_overlap_threshold,
        )

        # Extract sources
        sources = extract_sources_from_contexts(results)

        # Update semantic cache
        if self.config.retrieval.semantic_cache["enabled"]:
            self.semantic_cache.set(query_emb, full_answer, sources)

        yield {"type": "sources", "sources": sources}
        yield {
            "type": "done",
            "hallucination_risk": get_hallucination_risk_level(overlap),
            "latency_ms": (time.time() - start_time) * 1000,
        }