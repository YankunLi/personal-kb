"""ChromaDB vector store wrapper.

Handles: add documents, query by vector, delete by filter, metadata operations.
One collection per knowledge base.
"""

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings


class ChromaStore:
    """ChromaDB-backed vector store for document chunks.

    Each knowledge base is a separate ChromaDB collection.
    """

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def collection_name(self, kb_name: str) -> str:
        """Convert KB name to a valid ChromaDB collection name."""
        # ChromaDB collection names must be 3-512 chars, alphanumeric + . _ -
        # Must start and end with alphanumeric.
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in kb_name)
        safe = safe.strip("_-.")
        if not safe:
            safe = "default"
        # Ensure starts with alphanumeric
        if not safe[0].isalnum():
            safe = "kb" + safe
        # Ensure ends with alphanumeric
        if not safe[-1].isalnum():
            safe = safe + "0"
        return f"kb_{safe}"

    def _collection_name(self, kb_name: str) -> str:
        """Backward-compatible alias for collection_name."""
        return self.collection_name(kb_name)

    def get_or_create_collection(self, kb_name: str):
        """Get or create a ChromaDB collection for a knowledge base."""
        name = self._collection_name(kb_name)
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, kb_name: str):
        """Delete a knowledge base's collection."""
        name = self._collection_name(kb_name)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass  # Collection doesn't exist or already deleted

    def copy_collection(self, source_kb: str, target_kb: str, batch_size: int = 1000):
        """Copy all data from one collection to another using batching.

        Args:
            source_kb: Source KB name.
            target_kb: Target KB name.
            batch_size: Number of items per batch.

        Raises:
            ValueError: If source collection doesn't exist.
            RuntimeError: If copy fails verification.
        """
        source_name = self._collection_name(source_kb)
        target_name = self._collection_name(target_kb)

        try:
            source_coll = self._client.get_collection(source_name)
        except ValueError:
            raise ValueError(f"Source collection '{source_kb}' does not exist")

        total = source_coll.count()
        if total == 0:
            # Just ensure target exists
            self.get_or_create_collection(target_kb)
            return

        target_coll = self._client.get_or_create_collection(
            name=target_name,
            metadata={"hnsw:space": "cosine"},
        )

        copied = 0
        offset = 0
        while offset < total:
            batch = source_coll.get(
                offset=offset,
                limit=batch_size,
                include=["documents", "metadatas", "embeddings"],
            )
            if batch["ids"]:
                target_coll.add(
                    ids=batch["ids"],
                    documents=batch["documents"],
                    metadatas=batch["metadatas"],
                    embeddings=batch["embeddings"],
                )
                copied += len(batch["ids"])
            offset += batch_size

        if copied < total:
            self._client.delete_collection(target_name)
            raise RuntimeError(
                f"Failed to copy all chunks during rename "
                f"(expected {total}, got {copied})"
            )

    def collection_exists(self, kb_name: str) -> bool:
        """Check if a collection exists for a KB."""
        name = self._collection_name(kb_name)
        try:
            self._client.get_collection(name)
            return True
        except ValueError:
            return False

    def add_chunks(
        self,
        kb_name: str,
        chunks: list[dict[str, Any]],
        embeddings: list[np.ndarray],
    ):
        """Add chunks with embeddings to the vector store.

        Args:
            kb_name: Knowledge base name.
            chunks: List of chunk dicts with 'content' and 'metadata' keys.
            embeddings: List of embedding vectors corresponding to chunks.
        """
        if not chunks:
            return

        collection = self.get_or_create_collection(kb_name)

        ids = [c["metadata"]["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [_sanitize_metadata(c["metadata"]) for c in chunks]
        embeddings_list = [e.tolist() for e in embeddings]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )

    def query(
        self,
        kb_name: str,
        query_embedding: np.ndarray,
        top_k: int = 50,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store for similar chunks.

        Args:
            kb_name: Knowledge base name.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            where: Optional metadata filter dict.

        Returns:
            List of result dicts with 'id', 'content', 'metadata', 'score'.
        """
        collection = self.get_or_create_collection(kb_name)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns cosine distance; convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 - distance  # Cosine similarity from cosine distance

                output.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": score,
                })

        return output

    def get_existing_hashes(self, kb_name: str) -> set[str]:
        """Get all content hashes currently in the collection (for dedup)."""
        try:
            collection = self.get_or_create_collection(kb_name)
            results = collection.get(include=["metadatas"])
            hashes = set()
            if results["metadatas"]:
                for meta in results["metadatas"]:
                    h = meta.get("content_hash", "")
                    if h:
                        hashes.add(h)
            return hashes
        except Exception:
            return set()

    def count(self, kb_name: str) -> int:
        """Return the number of chunks in a knowledge base."""
        try:
            collection = self.get_or_create_collection(kb_name)
            return collection.count()
        except Exception:
            return 0

    def delete_by_file(self, kb_name: str, source_file: str):
        """Delete all chunks from a specific source file."""
        collection = self.get_or_create_collection(kb_name)
        results = collection.get(
            where={"source_file": source_file},
            include=["metadatas"],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])

    def list_collections(self) -> list[str]:
        """List all knowledge base collection names."""
        collections = self._client.list_collections()
        return [c.name.replace("kb_", "", 1) for c in collections if c.name.startswith("kb_")]


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """ChromaDB requires metadata values to be str, int, float, or bool."""
    sanitized = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized