"""Multi-KB manager: create, list, delete, switch knowledge bases."""

import json
import os
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
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {k: KBInfo.from_dict(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    "KB registry corrupted (%s), starting with empty registry.", e
                )
        return {}

    def _save_registry(self):
        data = {k: v.to_dict() for k, v in self._registry.items()}
        tmp_path = self.registry_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.registry_path)

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
        if not name or not name.strip():
            raise ValueError("Knowledge base name cannot be empty or whitespace")

        info = KBInfo(name=name, topic=topic)

        # Ensure ChromaDB collection exists before persisting registry entry
        self.chroma.get_or_create_collection(name)

        self._registry[name] = info
        self._save_registry()
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
        then deletes the old data.Atomic: old data is only deleted after
        all copies succeed.

        Raises:
            ValueError: If old_name doesn't exist or new_name already exists.
        """
        if old_name not in self._registry:
            raise ValueError(f"Knowledge base '{old_name}' does not exist")
        if new_name in self._registry:
            raise ValueError(f"Knowledge base '{new_name}' already exists")

        # Phase 1: Copy all data to new name
        chroma_copied = False
        bm25_copied = False
        try:
            try:
                self.chroma.copy_collection(old_name, new_name)
            except ValueError:
                # Source collection has no data yet, just ensure target exists
                self.chroma.get_or_create_collection(new_name)
            chroma_copied = True

            # Copy BM25 index
            self.bm25.copy_index(old_name, new_name)
            bm25_copied = True
        except Exception as e:
            # Rollback: clean up any partially copied data
            if chroma_copied:
                self.chroma.delete_collection(new_name)
            if bm25_copied:
                self.bm25.delete(new_name)
            raise RuntimeError(f"Failed to rename '{old_name}' to '{new_name}'") from e

        # Phase 2: Delete old data (only after copies are confirmed)
        self.chroma.delete_collection(old_name)
        self.bm25.delete(old_name)

        # Phase 3: Update registry
        info = self._registry.pop(old_name)
        info.name = new_name
        self._registry[new_name] = info
        self._save_registry()