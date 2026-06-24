"""Knowledge base metadata models."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class KBInfo(BaseModel):
    """Metadata for a single knowledge base."""
    name: str
    topic: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    chunk_count: int = 0
    file_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KBInfo":
        return cls(**data)