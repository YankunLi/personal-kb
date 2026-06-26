"""Hallucination detection via entity overlap check.

Based on Article A: "检查回答中的关键实体是否在上下文中出现"
Simple but effective: checks if numbers and key entities in the answer
exist in the retrieved contexts.
"""

import re
from typing import Any


def extract_numbers(text: str) -> set[str]:
    """Extract numbers (including percentages) from text."""
    return set(re.findall(r"\d+\.?\d*%?", text))


def extract_entities(text: str, min_length: int = 2) -> set[str]:
    """Extract potential named entities from text.

    Uses jieba for Chinese word segmentation instead of regex substring matching,
    which prevents false entity inflation from overlapping substrings.
    Falls back to regex if jieba is not available.
    """
    entities: set[str] = set()

    # Chinese: use jieba for proper word segmentation
    try:
        import jieba
        words = jieba.cut(text)
        for w in words:
            w = w.strip()
            if len(w) >= min_length and re.search(r"[\u4e00-\u9fff]", w):
                entities.add(w)
    except ImportError:
        # Fallback: match 2-4 char Chinese sequences (avoids substring explosion)
        entities.update(re.findall(r"[\u4e00-\u9fff]{2,4}", text))

    # English: words of 3+ letters
    entities.update(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text))

    return entities


def detect_hallucination(
    answer: str,
    contexts: list[dict[str, Any]],
    entity_overlap_threshold: float = 0.7,
) -> tuple[bool, float]:
    """Check if the answer contains entities not found in the retrieved contexts.

    Checks numbers and entities separately, returning the entity overlap ratio
    as the primary signal. Number mismatches are reported as a warning signal
    but do not override the entity overlap ratio.

    Args:
        answer: Generated answer text.
        contexts: Retrieved document chunks used for generation.
        entity_overlap_threshold: Minimum fraction of answer entities that must
            exist in the contexts.

    Returns:
        Tuple of (is_hallucination_risk: bool, overlap_ratio: float).
    """
    context_text = " ".join([c.get("content", "") for c in contexts])

    # Check entities (primary signal)
    answer_entities = extract_entities(answer)
    if answer_entities:
        context_entities = extract_entities(context_text)
        matched = answer_entities & context_entities
        overlap = len(matched) / len(answer_entities)
        if overlap < entity_overlap_threshold:
            return True, overlap
        return False, overlap

    # Check numbers for factual accuracy (secondary signal)
    answer_numbers = extract_numbers(answer)
    if answer_numbers:
        context_numbers = extract_numbers(context_text)
        unmatched = answer_numbers - context_numbers
        number_ratio = 1.0 - len(unmatched) / len(answer_numbers)
        if number_ratio < 0.5:
            return True, number_ratio

    return False, 1.0


def get_hallucination_risk_level(overlap_ratio: float) -> str:
    """Convert overlap ratio to a risk level string."""
    if overlap_ratio >= 0.9:
        return "low"
    elif overlap_ratio >= 0.7:
        return "medium"
    else:
        return "high"