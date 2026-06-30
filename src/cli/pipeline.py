"""Core pipeline orchestrating document import, retrieval, and generation."""

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config.loader import AppConfig, load_config

logger = logging.getLogger(__name__)


@contextmanager
def _phase(timing: dict[str, float], name: str):
    """Context manager to record elapsed time for a named phase."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing[name] = (time.perf_counter() - t0) * 1000


def _log_timing(operation: str, timing: dict[str, float]):
    """Log timing breakdown at DEBUG level."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    parts = ", ".join(f"{k}={v:.0f}ms" for k, v in timing.items())
    logger.debug("%s timing: %s", operation, parts)

from src.doc_processing.loader import load_documents
from src.doc_processing.chunker import chunk_document, chunk_document_semantic
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
from src.generation.hallucination import (
    detect_hallucination,
    get_hallucination_risk_level,
    verify_factual_accuracy,
)
from src.kb_manager.manager import KBManager
from src.source_tracking.tracker import (
    RAGResponse,
    extract_sources_from_contexts,
    format_sources_output,
    verify_citations,
)


# Global pipeline instance (lazy initialized)
_pipeline: "Pipeline | None" = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> "Pipeline":
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
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
            cache_dir=config.embedding.cache_dir,
            cache_max_entries=config.embedding.cache_max_entries,
        )
        self.chroma = ChromaStore(
            persist_dir=config.vector_store.persist_dir,
            distance_metric=config.vector_store.distance_metric,
        )
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
        self.reranker = Reranker(
            model_name=config.retrieval.reranker_model,
            model_revision=config.retrieval.reranker_model_revision,
        )
        self.semantic_cache = SemanticCache(
            similarity_threshold=config.retrieval.semantic_cache.get("similarity_threshold", 0.95),
            max_size=config.retrieval.semantic_cache.get("max_size", 1000),
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
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Import documents into a knowledge base.

        Processes documents in streaming batches to avoid accumulating all
        chunks in memory for large imports.

        Args:
            path: File or directory path.
            kb_name: Target knowledge base name.
            recursive: Whether to recurse into subdirectories.
            dry_run: If True, show what would be imported without actually indexing.
            progress_callback: Optional callback for progress updates.
            batch_size: Number of files to buffer before embedding + storing.

        Returns:
            Dict with import statistics.
        """
        path = Path(path).expanduser()
        start_time = time.time()

        # Ensure KB exists
        if not self.kb_manager.exists(kb_name):
            self.kb_manager.create(kb_name)

        # Load existing hashes for cross-batch dedup and seed deduplicator once
        existing_hashes = self.chroma.get_existing_hashes(kb_name)
        self.deduplicator.reset()
        if existing_hashes:
            self.deduplicator.seed_hashes(existing_hashes)

        total_files = 0
        total_chunks = 0
        total_duplicates = 0
        files_with_new_chunks = 0
        batch_chunks: list[dict[str, Any]] = []
        import_timing: dict[str, float] = {"embed": 0.0, "index": 0.0}
        bm25_loaded = False

        # Clear semantic cache upfront before any data is added, so queries
        # during the import don't get stale cached results from partial data.
        self.semantic_cache.clear(kb_name)

        def _flush_batch(batch: list[dict[str, Any]]):
            """Embed + store a batch of chunks, updating BM25 index."""
            nonlocal bm25_loaded
            if not batch:
                return

            if progress_callback:
                progress_callback("embed", "", 0)

            t0 = time.perf_counter()
            texts = [c["content"] for c in batch]
            embeddings = self.embedder.encode_documents(texts, show_progress=False)
            import_timing["embed"] += (time.perf_counter() - t0) * 1000

            if progress_callback:
                progress_callback("index", "", 0)

            t0 = time.perf_counter()
            self.chroma.add_chunks(kb_name, batch, embeddings)

            # Update BM25 incrementally: load once, add per batch, save checkpoint
            try:
                if not bm25_loaded:
                    if not self.bm25.load(kb_name):
                        self.bm25.build(batch)
                    else:
                        self.bm25.add_chunks(batch)
                    bm25_loaded = True
                else:
                    self.bm25.add_chunks(batch)
                # Save checkpoint after each successful batch so rollback
                # can always reload from the last good state.
                self.bm25.save(kb_name)
            except Exception:
                # Rollback: undo the just-added ChromaDB chunks. If this
                # deletion fails, log it and continue with BM25 reload so
                # we at least restore in-memory consistency.
                try:
                    chunk_ids = [c["metadata"]["chunk_id"] for c in batch]
                    self.chroma.delete_by_ids(kb_name, chunk_ids)
                except Exception as rollback_e:
                    logger.error(
                        "Failed to roll back ChromaDB chunks during import: %s",
                        rollback_e,
                    )
                # Rollback BM25 in-memory state: reload from last good checkpoint
                if not self.bm25.load(kb_name):
                    self.bm25.reset()
                # Rollback resets the loaded-state flag so the next batch
                # (if any) re-loads from scratch instead of assuming the
                # in-memory state is valid.
                bm25_loaded = False
                raise
            finally:
                import_timing["index"] += (time.perf_counter() - t0) * 1000

        try:
            doc_iter, failed_counter = load_documents(path, recursive=recursive)
            for doc in doc_iter:
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
                if self.config.chunking.chunk_method == "semantic":
                    chunks = chunk_document_semantic(
                        doc["text"],
                        metadata=None,
                        chunk_size=self.config.chunking.chunk_size,
                        chunk_overlap=self.config.chunking.chunk_overlap,
                        min_chunk_size=self.config.chunking.min_chunk_size,
                    )
                else:
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
                    chunks, dups = self.deduplicator.deduplicate(chunks)
                    total_duplicates += dups

                if len(chunks) > 0:
                    files_with_new_chunks += 1

                total_chunks += len(chunks)
                batch_chunks.extend(chunks)

                # Flush batch when we've accumulated enough files
                if total_files % batch_size == 0 and batch_chunks:
                    _flush_batch(batch_chunks)
                    batch_chunks.clear()

            total_failed = failed_counter[0]  # Read after iterator is consumed

            if dry_run:
                return {
                    "files": total_files,
                    "chunks": 0,
                    "duplicates": 0,
                    "failed": total_failed,
                    "dry_run": True,
                }

            # Flush remaining chunks
            if batch_chunks:
                _flush_batch(batch_chunks)
                batch_chunks.clear()

            if total_chunks == 0:
                import_timing["total"] = (time.time() - start_time) * 1000
                _log_timing("import", import_timing)
                return {
                    "files": total_files,
                    "chunks": 0,
                    "duplicates": total_duplicates,
                    "failed": total_failed,
                }

            # Update KB stats
            existing = self.kb_manager.get(kb_name)
            try:
                new_chunk_count = self.chroma.count(kb_name)
            except Exception:
                logger.warning("Failed to query chunk count for '%s'; stats may be stale.", kb_name, exc_info=True)
                new_chunk_count = existing.chunk_count
            self.kb_manager.update_stats(
                kb_name,
                chunk_count=new_chunk_count,
                file_count=existing.file_count + files_with_new_chunks,
            )

            import_timing["total"] = (time.time() - start_time) * 1000
            _log_timing("import", import_timing)

            return {
                "files": total_files,
                "chunks": total_chunks,
                "duplicates": total_duplicates,
                "failed": total_failed,
            }
        except Exception:
            logger.error("Import failed after processing %d files, %d chunks", total_files, total_chunks, exc_info=True)
            raise

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
        # Verify KB exists in registry first
        if not self.kb_manager.exists(kb_name):
            raise ValueError(f"Knowledge base '{kb_name}' does not exist. Create it with 'kb kb create {kb_name}' or import documents.")

        # Validate query
        if not query or not query.strip():
            return []
        if len(query) > 4096:
            return []

        # Load BM25 index for this KB
        if not self.bm25.has_index(kb_name):
            return []
        if not self.bm25.load(kb_name):
            return []

        # Hybrid search: retrieve the full candidate pool (hybrid_top_k, e.g.
        # 50) so the reranker can improve recall, then cut down to top_k.
        # Passing top_k here would cap candidates before reranking, defeating
        # the purpose of the reranker (only reorders, never promotes).
        results = self.retriever.retrieve(kb_name, query)

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
        # Validate query length to prevent abuse / OOM on embedding
        if not query or not query.strip():
            return RAGResponse(
                answer="请输入有效的问题。",
                sources=[],
                hallucination_risk="low",
                latency_ms=0.0,
            )
        if len(query) > 4096:
            return RAGResponse(
                answer=f"问题过长（{len(query)}字符），请精简到4096字符以内。",
                sources=[],
                hallucination_risk="low",
                latency_ms=0.0,
            )
        timing: dict[str, float] = {}
        start_time = time.perf_counter()

        # Load BM25 index for this KB
        with _phase(timing, "bm25_load"):
            if not self.bm25.has_index(kb_name):
                timing["total"] = (time.perf_counter() - start_time) * 1000
                _log_timing("chat", timing)
                return RAGResponse(
                    answer="知识库为空，请先导入文档。",
                    sources=[],
                    hallucination_risk="low",
                    latency_ms=timing["total"],
                )
            if not self.bm25.load(kb_name):
                timing["total"] = (time.perf_counter() - start_time) * 1000
                _log_timing("chat", timing)
                return RAGResponse(
                    answer="抱歉，BM25 索引损坏，请尝试重新导入文档。",
                    sources=[],
                    hallucination_risk="low",
                    latency_ms=timing["total"],
                )

        # Compute query embedding once for cache + retrieval
        with _phase(timing, "query_embed"):
            query_emb = self.embedder.encode_query(query)

        # Check semantic cache
        with _phase(timing, "cache_check"):
            if self.config.retrieval.semantic_cache.get("enabled", True):
                cached = self.semantic_cache.get(query_emb, kb_name=kb_name)
                if cached:
                    timing["total"] = (time.perf_counter() - start_time) * 1000
                    _log_timing("chat", timing)
                    return RAGResponse(
                        answer=cached["answer"],
                        sources=cached["sources"],
                        hallucination_risk="low",
                        latency_ms=timing["total"],
                    )

        # Hybrid search
        with _phase(timing, "hybrid_search"):
            try:
                results = self.retriever.retrieve(kb_name, query, query_embedding=query_emb)
            except Exception as e:
                logger.warning("Hybrid search failed: %s", e, exc_info=True)
                timing["total"] = (time.perf_counter() - start_time) * 1000
                _log_timing("chat", timing)
                return RAGResponse(
                    answer="检索知识库时出现错误，请稍后重试。",
                    sources=[],
                    hallucination_risk="low",
                    latency_ms=timing["total"],
                )

        if not results:
            timing["total"] = (time.perf_counter() - start_time) * 1000
            _log_timing("chat", timing)
            return RAGResponse(
                answer="抱歉，在当前知识库中没有找到相关信息。请尝试更换关键词或导入相关文档。",
                sources=[],
                hallucination_risk="low",
                latency_ms=timing["total"],
            )

        # Rerank
        with _phase(timing, "rerank"):
            try:
                results = self.reranker.rerank(query, results, top_n=self.config.retrieval.rerank_top_n)
            except Exception as e:
                logger.warning("Reranker failed, using raw retrieval scores: %s", e, exc_info=True)
                # Fall back to raw scores by sorting descending
                results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:self.config.retrieval.rerank_top_n]

        # Build prompt
        with _phase(timing, "build_prompt"):
            messages = build_messages(query, results, chat_history=chat_history)

        # Generate
        with _phase(timing, "llm_gen"):
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
        with _phase(timing, "hallucination"):
            is_risk, overlap = detect_hallucination(
                answer, results,
                entity_overlap_threshold=self.config.hallucination.entity_overlap_threshold,
            )
            risk_level = get_hallucination_risk_level(overlap)

            # Optional LLM-based verification for high-risk answers
            # (create a fresh adapter since the generation one was closed)
            if (
                self.config.hallucination.llm_verification
                and risk_level == "high"
            ):
                verify_llm = None
                try:
                    verify_llm = self._get_llm_adapter(provider_name)
                    verification = await verify_factual_accuracy(
                        answer, results, verify_llm,
                    )
                    if not verification["is_accurate"]:
                        logger.warning(
                            "LLM verification flagged inaccuracies: %s",
                            verification.get("issues", []),
                        )
                except Exception:
                    pass
                finally:
                    if verify_llm is not None:
                        await verify_llm.close()

        # Extract sources
        sources = extract_sources_from_contexts(results)

        # Citation verification: warn if the LLM omitted or mis-cited sources.
        # Soft signal only — does not alter the answer, just logs.
        try:
            cited_ok, citation_issues = verify_citations(answer, sources)
            if not cited_ok:
                logger.warning(
                    "Citation check failed for answer: %s", citation_issues
                )
        except Exception:
            logger.debug("Citation verification skipped due to error", exc_info=True)

        # Update semantic cache
        with _phase(timing, "cache_set"):
            if self.config.retrieval.semantic_cache.get("enabled", True):
                self.semantic_cache.set(query_emb, answer, sources, kb_name=kb_name)

        timing["total"] = (time.perf_counter() - start_time) * 1000
        _log_timing("chat", timing)

        return RAGResponse(
            answer=answer,
            sources=sources,
            hallucination_risk=risk_level,
            latency_ms=timing["total"],
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
            Dicts with 'type': 'token'|'answer'|'sources'|'done'|'error'.
        """
        timing: dict[str, float] = {}
        start_time = time.perf_counter()

        # Validate query length
        if not query or not query.strip():
            timing["total"] = (time.perf_counter() - start_time) * 1000
            yield {"type": "answer", "content": "请输入有效的问题。"}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "hallucination_risk": "low", "latency_ms": timing["total"]}
            return
        if len(query) > 4096:
            timing["total"] = (time.perf_counter() - start_time) * 1000
            yield {
                "type": "answer",
                "content": f"问题过长（{len(query)}字符），请精简到4096字符以内。",
            }
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "hallucination_risk": "low", "latency_ms": timing["total"]}
            return

        # Load BM25 index
        t0 = time.perf_counter()
        if not self.bm25.has_index(kb_name):
            timing["bm25_load"] = (time.perf_counter() - t0) * 1000
            timing["total"] = (time.perf_counter() - start_time) * 1000
            _log_timing("chat_stream", timing)
            yield {
                "type": "answer",
                "content": "知识库为空，请先导入文档。",
            }
            yield {"type": "sources", "sources": []}
            yield {
                "type": "done",
                "hallucination_risk": "low",
                "latency_ms": timing["total"],
            }
            return
        if not self.bm25.load(kb_name):
            timing["bm25_load"] = (time.perf_counter() - t0) * 1000
            timing["total"] = (time.perf_counter() - start_time) * 1000
            _log_timing("chat_stream", timing)
            yield {
                "type": "answer",
                "content": "抱歉，BM25 索引损坏，请尝试重新导入文档。",
            }
            yield {"type": "sources", "sources": []}
            yield {
                "type": "done",
                "hallucination_risk": "low",
                "latency_ms": timing["total"],
            }
            return
        timing["bm25_load"] = (time.perf_counter() - t0) * 1000

        # Compute query embedding once for cache + retrieval
        t0 = time.perf_counter()
        query_emb = self.embedder.encode_query(query)
        timing["query_embed"] = (time.perf_counter() - t0) * 1000

        # Check semantic cache
        t0 = time.perf_counter()
        if self.config.retrieval.semantic_cache.get("enabled", True):
            cached = self.semantic_cache.get(query_emb, kb_name=kb_name)
            if cached:
                timing["cache_check"] = (time.perf_counter() - t0) * 1000
                timing["total"] = (time.perf_counter() - start_time) * 1000
                _log_timing("chat_stream", timing)
                yield {"type": "answer", "content": cached["answer"]}
                yield {"type": "sources", "sources": cached["sources"]}
                yield {
                    "type": "done",
                    "hallucination_risk": "low",
                    "latency_ms": timing["total"],
                }
                return
        timing["cache_check"] = (time.perf_counter() - t0) * 1000

        # Hybrid search
        t0 = time.perf_counter()
        try:
            results = self.retriever.retrieve(kb_name, query, query_embedding=query_emb)
        except Exception as e:
            logger.warning("Hybrid search failed in chat_stream: %s", e, exc_info=True)
            timing["hybrid_search"] = (time.perf_counter() - t0) * 1000
            timing["total"] = (time.perf_counter() - start_time) * 1000
            _log_timing("chat_stream", timing)
            yield {
                "type": "error",
                "content": "检索知识库时出现错误，请稍后重试。",
            }
            yield {
                "type": "done",
                "hallucination_risk": "low",
                "latency_ms": timing["total"],
            }
            return
        timing["hybrid_search"] = (time.perf_counter() - t0) * 1000

        if not results:
            timing["total"] = (time.perf_counter() - start_time) * 1000
            _log_timing("chat_stream", timing)
            yield {
                "type": "answer",
                "content": "抱歉，在当前知识库中没有找到相关信息。请尝试更换关键词或导入相关文档。",
            }
            yield {"type": "sources", "sources": []}
            yield {
                "type": "done",
                "hallucination_risk": "low",
                "latency_ms": timing["total"],
            }
            return

        # Rerank
        t0 = time.perf_counter()
        try:
            results = self.reranker.rerank(query, results, top_n=self.config.retrieval.rerank_top_n)
        except Exception as e:
            logger.warning("Reranker failed in chat_stream, using raw scores: %s", e, exc_info=True)
            results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:self.config.retrieval.rerank_top_n]
        timing["rerank"] = (time.perf_counter() - t0) * 1000

        # Build prompt
        t0 = time.perf_counter()
        messages = build_messages(query, results, chat_history=chat_history)
        timing["build_prompt"] = (time.perf_counter() - t0) * 1000

        # Generate streaming
        t0 = time.perf_counter()
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
        timing["llm_gen"] = (time.perf_counter() - t0) * 1000

        # Hallucination check
        t0 = time.perf_counter()
        is_risk, overlap = detect_hallucination(
            full_answer, results,
            entity_overlap_threshold=self.config.hallucination.entity_overlap_threshold,
        )
        risk_level = get_hallucination_risk_level(overlap)

        # Optional LLM-based verification for high-risk answers (streaming:
        # re-open a short-lived adapter for the verification call)
        if (
            self.config.hallucination.llm_verification
            and risk_level == "high"
        ):
            verify_llm = None
            try:
                verify_llm = self._get_llm_adapter(provider_name)
                verification = await verify_factual_accuracy(
                    full_answer, results, verify_llm,
                )
                if not verification["is_accurate"]:
                    logger.warning(
                        "LLM verification flagged inaccuracies: %s",
                        verification.get("issues", []),
                    )
            except Exception:
                pass
            finally:
                if verify_llm is not None:
                    await verify_llm.close()
        timing["hallucination"] = (time.perf_counter() - t0) * 1000

        # Extract sources
        sources = extract_sources_from_contexts(results)

        # Citation verification: warn if the LLM omitted or mis-cited sources.
        # Soft signal only — does not alter the answer, just logs.
        try:
            cited_ok, citation_issues = verify_citations(full_answer, sources)
            if not cited_ok:
                logger.warning(
                    "Citation check failed for answer: %s", citation_issues
                )
        except Exception:
            logger.debug("Citation verification skipped due to error", exc_info=True)

        # Update semantic cache
        t0 = time.perf_counter()
        if self.config.retrieval.semantic_cache.get("enabled", True):
            self.semantic_cache.set(query_emb, full_answer, sources, kb_name=kb_name)
        timing["cache_set"] = (time.perf_counter() - t0) * 1000

        timing["total"] = (time.perf_counter() - start_time) * 1000
        _log_timing("chat_stream", timing)

        yield {"type": "sources", "sources": sources}
        yield {
            "type": "done",
            "hallucination_risk": risk_level,
            "latency_ms": timing["total"],
        }