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
    """Extract potential named entities: Chinese words of 2+ chars, English words."""
    # Chinese: sequences of 2+ Chinese characters
    chinese = set(re.findall(r"[\u4e00-\u9fff]{" + str(min_length) + r",}", text))
    # English: words of 3+ letters
    english = set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text))
    return chinese | english


def detect_hallucination(
    answer: str,
    contexts: list[dict[str, Any]],
    entity_overlap_threshold: float = 0.7,
) -> tuple[bool, float]:
    """Check if the answer contains entities not found in the retrieved contexts.

    Args:
        answer: Generated answer text.
        contexts: Retrieved document chunks used for generation.
        entity_overlap_threshold: Minimum fraction of answer entities that must
            exist in the contexts.

    Returns:
        Tuple of (is_hallucination_risk: bool, overlap_ratio: float).
    """
    context_text = " ".join([c.get("content", "") for c in contexts])

    # Check numbers (critical for factual accuracy)
    answer_numbers = extract_numbers(answer)
    if answer_numbers:
        context_numbers = extract_numbers(context_text)
        unmatched = answer_numbers - context_numbers
        number_ratio = 1.0 - len(unmatched) / len(answer_numbers)
        if number_ratio < entity_overlap_threshold:
            return True, number_ratio

    # Check entities
    answer_entities = extract_entities(answer)
    if answer_entities:
        context_entities = extract_entities(context_text)
        if not answer_entities:
            return False, 1.0
        matched = answer_entities & context_entities
        overlap = len(matched) / len(answer_entities)
        if overlap < entity_overlap_threshold:
            return True, overlap
        return False, overlap

    return False, 1.0


def get_hallucination_risk_level(overlap_ratio: float) -> str:
    """Convert overlap ratio to a risk level string."""
    if overlap_ratio >= 0.9:
        return "low"
    elif overlap_ratio >= 0.7:
        return "medium"
    else:
        return "high"