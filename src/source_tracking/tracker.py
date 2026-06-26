"""End-to-end source citation tracking.

Based on Article A: "每个 chunk 带 chunk_id，最终答案可以追溯到源文档的哪一段"
Based on Article B: "100% 必须有引用来源"
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceInfo:
    """Structured source citation information."""
    chunk_id: str
    source_file: str
    section: str | None = None
    page: int | None = None
    relevance_score: float = 0.0
    text_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "section": self.section,
            "page": self.page,
            "relevance_score": self.relevance_score,
            "text_preview": self.text_preview,
        }


@dataclass
class RAGResponse:
    """Complete RAG response with sources and metadata."""
    answer: str
    sources: list[SourceInfo] = field(default_factory=list)
    hallucination_risk: str = "low"
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "hallucination_risk": self.hallucination_risk,
            "latency_ms": self.latency_ms,
        }


def _extract_page_from_text(text: str) -> int | None:
    """Extract page number from PDF page markers like [第12页]."""
    m = re.search(r"\[第(\d+)页\]", text)
    if m:
        return int(m.group(1))
    return None


def _get_score(doc: dict[str, Any]) -> float:
    """Extract relevance score, handling None values."""
    score = doc.get("rerank_score")
    if score is None:
        score = doc.get("score", 0)
    if score is None:
        return 0.0
    return float(score)


def extract_sources_from_contexts(
    contexts: list[dict[str, Any]],
    max_preview_chars: int = 200,
) -> list[SourceInfo]:
    """Extract structured source info from retrieved context documents.

    Args:
        contexts: List of retrieved document dicts.
        max_preview_chars: Max characters for text preview.

    Returns:
        List of SourceInfo objects.
    """
    sources = []
    for doc in contexts:
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")

        sources.append(SourceInfo(
            chunk_id=metadata.get("chunk_id", ""),
            source_file=metadata.get("source_file_basename", metadata.get("source_file", "未知")),
            section=metadata.get("section"),
            page=metadata.get("page") if metadata.get("page") is not None else _extract_page_from_text(content),
            relevance_score=_get_score(doc),
            text_preview=content[:max_preview_chars],
        ))

    return sources


CITATION_RE = re.compile(r"\[来源(\d+)\]")


def extract_citations(answer: str) -> list[str]:
    """Extract source citation indices from the answer text.

    Finds patterns like [来源1], [来源2], etc.
    Returns list of citation strings.
    """
    return [f"[来源{m}]" for m in CITATION_RE.findall(answer)]


def verify_citations(
    answer: str,
    sources: list[SourceInfo],
) -> tuple[bool, list[str]]:
    """Verify that source citations in the answer reference valid sources.

    Args:
        answer: Generated answer text.
        sources: List of source info used for generation.

    Returns:
        Tuple of (all_valid: bool, issues: list[str]).
        If no citations are found, returns (False, ["no_citations"]) to flag
        that the LLM did not cite any sources.
    """
    cited = [int(m) for m in CITATION_RE.findall(answer)]
    max_index = len(sources)

    if not cited:
        return False, ["no_citations"]

    invalid = []
    for idx in cited:
        if idx < 1 or idx > max_index:
            invalid.append(f"[来源{idx}]")

    return len(invalid) == 0, invalid


def format_sources_output(sources: list[SourceInfo]) -> str:
    """Format sources for user display.

    Args:
        sources: List of SourceInfo objects.

    Returns:
        Formatted string for CLI output.
    """
    lines = ["\n📚 来源:"]
    for i, src in enumerate(sources, 1):
        parts = [f"[来源{i}] {src.source_file}"]
        if src.section:
            parts.append(f" | {src.section}")
        if src.page is not None:
            parts.append(f" | 第{src.page}页")
        parts.append(f" | 相关度: {src.relevance_score:.2f}")
        lines.append("  " + "".join(parts))
    return "\n".join(lines)