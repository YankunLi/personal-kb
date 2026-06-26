"""Multi-KB manager: create, list, delete, switch knowledge bases."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.kb_manager.models import KBInfo
from src.vector_store.chroma_store import ChromaStore
from src.vector_store.bm25_index import BM25Index


class KBManager:
    """Manages multiple knowledge bases with persistent registry.

    Each KB is an independent ChromaDB collection + BM25 index.
    The registry is stored in JSON at data/kb_registry.json.
    """

    def __init__(
        self,
        registry_path: str = "data/kb_registry.json",
        chroma_store: ChromaStore | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma = chroma_store or ChromaStore()
        self.bm25 = bm25_index or BM25Index()
        self._registry: dict[str, KBInfo] = self._load_registry()

        # Ensure default KB exists
        if "default" not in self._registry:
            self.create("default", topic="通用知识库")

    def _load_registry(self) -> dict[str, KBInfo]:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: KBInfo.from_dict(v) for k, v in data.items()}
        return {}

    def _save_registry(self):
        data = {k: v.to_dict() for k, v in self._registry.items()}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create(self, name: str, topic: str = "") -> KBInfo:
        """Create a new knowledge base.

        Args:
            name: Unique KB name.
            topic: Optional description of the KB's topic.

        Returns:
            KBInfo for the created KB.

        Raises:
            ValueError: If KB with this name already exists.
        """
        if name in self._registry:
            raise ValueError(f"Knowledge base '{name}' already exists")

        info = KBInfo(name=name, topic=topic)
        self._registry[name] = info
        self._save_registry()

        # Ensure ChromaDB collection exists
        self.chroma.get_or_create_collection(name)
        return info

    def delete(self, name: str, force: bool = False):
        """Delete a knowledge base and all its data.

        Args:
            name: KB name to delete.
            force: If True, skip confirmation.

        Raises:
            ValueError: If KB doesn't exist or is 'default'.
        """
        if name == "default":
            raise ValueError("Cannot delete the default knowledge base")
        if name not in self._registry:
            raise ValueError(f"Knowledge base '{name}' does not exist")

        # Delete vector data
        self.chroma.delete_collection(name)
        self.bm25.delete(name)

        # Remove from registry
        del self._registry[name]
        self._save_registry()

    def list(self) -> list[KBInfo]:
        """List all knowledge bases with stats."""
        result = []
        for name, info in self._registry.items():
            info.chunk_count = self.chroma.count(name)
            result.append(info)
        return result

    def get(self, name: str) -> KBInfo:
        """Get KB info by name.

        Raises:
            ValueError: If KB doesn't exist.
        """
        if name not in self._registry:
            raise ValueError(f"Knowledge base '{name}' does not exist")
        info = self._registry[name]
        info.chunk_count = self.chroma.count(name)
        return info

    def exists(self, name: str) -> bool:
        """Check if a KB exists."""
        return name in self._registry

    def update_stats(self, name: str, chunk_count: int | None = None, file_count: int | None = None):
        """Update KB statistics after import."""
        if name in self._registry:
            if chunk_count is not None:
                self._registry[name].chunk_count = chunk_count
            if file_count is not None:
                self._registry[name].file_count = file_count
            self._save_registry()

    def rename(self, old_name: str, new_name: str):
        """Rename a knowledge base.

        Copies ChromaDB collection data and BM25 index to the new name,
        then deletes the old data.

        Raises:
            ValueError: If old_name doesn't exist or new_name already exists.
        """
        if old_name not in self._registry:
            raise ValueError(f"Knowledge base '{old_name}' does not exist")
        if new_name in self._registry:
            raise ValueError(f"Knowledge base '{new_name}' already exists")

        # Copy ChromaDB data: get all chunks from old collection,
        # add them to a new collection, then delete the old.
        old_collection = self.chroma._collection_name(old_name)
        new_collection = self.chroma._collection_name(new_name)

        try:
            old_data = self.chroma._client.get_collection(old_collection).get(
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception:
            # Old collection doesn't exist (no data yet), just ensure new exists
            self.chroma.get_or_create_collection(new_name)
        else:
            if old_data["ids"]:
                new_coll = self.chroma._client.get_or_create_collection(
                    name=new_collection,
                    metadata={"hnsw:space": "cosine"},
                )
                new_coll.add(
                    ids=old_data["ids"],
                    documents=old_data["documents"] or [],
                    metadatas=old_data["metadatas"] or [],
                    embeddings=old_data["embeddings"] or [],
                )
                # Verify copy succeeded before deleting old data
                if new_coll.count() >= len(old_data["ids"]):
                    self.chroma.delete_collection(old_name)
                else:
                    self.chroma.delete_collection(new_name)
                    raise RuntimeError(
                        f"Failed to copy all chunks during rename "
                        f"(expected {len(old_data['ids'])}, got {new_coll.count()})"
                    )
            else:
                self.chroma.delete_collection(old_name)

        # Copy BM25 index
        old_bm25 = self.bm25.index_dir / old_name / "bm25.pkl"
        if old_bm25.exists():
            new_bm25_dir = self.bm25.index_dir / new_name
            new_bm25_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(old_bm25, new_bm25_dir / "bm25.pkl")
            self.bm25.delete(old_name)

        # Update registry
        info = self._registry.pop(old_name)
        info.name = new_name
        self._registry[new_name] = info
        self._save_registry()