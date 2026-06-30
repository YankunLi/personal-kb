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

    def _save_registry(self, registry: dict[str, KBInfo] | None = None):
        """Persist the registry atomically (tmp + fsync + rename).

        If ``registry`` is given, persist that state without first mutating
        ``self._registry`` — callers can use this to make a rename atomic:
        persist the new state, and only swap in-memory after the disk write
        succeeds.
        """
        reg = registry if registry is not None else self._registry
        data = {k: v.to_dict() for k, v in reg.items()}
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
            ValueError: If KB with this name already exists, or name
                contains invalid characters.
        """
        if name in self._registry:
            raise ValueError(f"Knowledge base '{name}' already exists")
        if not name or not name.strip():
            raise ValueError("Knowledge base name cannot be empty or whitespace")

        # Validate name against BM25's allowed character set so a KB that
        # ChromaDB accepts doesn't become unusable for keyword search.
        import re
        if not re.match(r"^[a-zA-Z0-9\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff_.-]*$", name):
            raise ValueError(
                f"Invalid KB name: {name!r}. "
                "Must start with alphanumeric or Chinese char, "
                "contain only alphanumeric, Chinese, underscores, dots, or hyphens."
            )
        if ".." in name:
            raise ValueError(f"Path traversal detected in KB name: {name!r}")

        info = KBInfo(name=name, topic=topic)

        # Ensure ChromaDB collection exists (idempotent; an orphaned empty
        # collection from a failed create is harmless and reused on retry).
        self.chroma.get_or_create_collection(name)

        # Persist the new registry state to disk BEFORE mutating in-memory
        # state, so a write failure leaves no trace in self._registry.
        new_registry = {**self._registry, name: info}
        self._save_registry(new_registry)
        self._registry = new_registry
        return info

    def delete(self, name: str):
        """Delete a knowledge base and all its data.

        Args:
            name: KB name to delete.

        Raises:
            ValueError: If KB doesn't exist or is 'default'.
        """
        if name == "default":
            raise ValueError("Cannot delete the default knowledge base")
        if name not in self._registry:
            raise ValueError(f"Knowledge base '{name}' does not exist")

        # Phase 1: Delete vector + keyword data BEFORE updating the registry.
        # If this fails, the registry is untouched and the KB still exists —
        # the user can retry the delete.
        self.chroma.delete_collection(name)
        self.bm25.delete(name)

        # Phase 2: Persist the updated registry atomically. At this point the
        # data is already gone, so even if this write fails we only have an
        # orphaned registry entry (recoverable) instead of orphaned data that
        # could silently contaminate a re-created KB.
        new_registry = {k: v for k, v in self._registry.items() if k != name}
        self._save_registry(new_registry)
        self._registry = new_registry

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
            ValueError: If old_name doesn't exist, new_name already exists,
                or new_name contains invalid characters.
            RuntimeError: If data copy fails (rolled back internally).
        """
        if old_name not in self._registry:
            raise ValueError(f"Knowledge base '{old_name}' does not exist")
        if new_name in self._registry:
            raise ValueError(f"Knowledge base '{new_name}' already exists")

        # Validate new_name against BM25's allowed character set (same as create).
        import re
        if not re.match(r"^[a-zA-Z0-9\u4e00-\u9fff][a-zA-Z0-9\u4e00-\u9fff_.-]*$", new_name):
            raise ValueError(
                f"Invalid KB name: {new_name!r}. "
                "Must start with alphanumeric or Chinese char, "
                "contain only alphanumeric, Chinese, underscores, dots, or hyphens."
            )
        if ".." in new_name:
            raise ValueError(f"Path traversal detected in KB name: {new_name!r}")

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

        # Phase 2: Persist the new registry state to disk BEFORE mutating
        # in-memory state or deleting old data. If the write fails (or the
        # process crashes), old_name's data and registry entry stay intact
        # and the rename is simply aborted — no broken state.
        old_info = self._registry[old_name]
        new_info = old_info.model_copy(update={"name": new_name})
        new_registry = {k: v for k, v in self._registry.items() if k != old_name}
        new_registry[new_name] = new_info
        self._save_registry(new_registry)

        # Phase 3: Disk is safely updated — now swap the in-memory state.
        self._registry = new_registry

        # Phase 4: Delete old data. A failure here leaves orphaned old data
        # on disk (recoverable), but the registry correctly points to new_name.
        self.chroma.delete_collection(old_name)
        self.bm25.delete(old_name)