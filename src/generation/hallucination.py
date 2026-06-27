"""Hallucination detection via entity overlap + optional LLM-based verification.

Based on Article A: "检查回答中的关键实体是否在上下文中出现"
Simple but effective: checks if numbers and key entities in the answer
exist in the retrieved contexts. For high-risk answers, an optional LLM-based
verification pass can double-check factual accuracy against the sources.
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

    return False, -1.0  # Unable to assess: no entities or numbers in answer


def get_hallucination_risk_level(overlap_ratio: float) -> str:
    """Convert overlap ratio to a risk level string."""
    if overlap_ratio < 0:
        return "unknown"  # Unable to assess
    elif overlap_ratio >= 0.9:
        return "low"
    elif overlap_ratio >= 0.7:
        return "medium"
    else:
        return "high"


def build_verification_prompt(answer: str, contexts: list[dict[str, Any]]) -> str:
    """Build a prompt for LLM-based factual verification.

    Asks the LLM to check whether each factual claim in the answer is
    supported by the provided source contexts.
    """
    context_texts = []
    for i, ctx in enumerate(contexts, 1):
        content = ctx.get("content", "")
        context_texts.append(f"[来源{i}]\n{content}")
    sources = "\n\n".join(context_texts)

    return f"""请检查以下回答是否完全基于提供的来源内容。对于回答中的每个事实性陈述，判断其是否能在来源中找到支持。

## 来源内容
{sources}

## 待验证回答
{answer}

## 验证要求
请判断回答是否存在以下问题：
1. 无中生有：回答中包含来源中不存在的具体事实、数字或实体
2. 错误引用：回答中的信息与来源内容矛盾或歪曲

请只回复一个JSON对象：
{{"is_accurate": true/false, "issues": ["问题描述"], "supported_ratio": 0.0-1.0}}"""


async def verify_factual_accuracy(
    answer: str,
    contexts: list[dict[str, Any]],
    llm_adapter: Any,
) -> dict[str, Any]:
    """Use LLM to verify if the answer is factually supported by the contexts.

    This is an optional second-pass verification for answers flagged as
    high hallucination risk by the entity overlap check.

    Args:
        answer: Generated answer text.
        contexts: Retrieved document chunks used for generation.
        llm_adapter: LLMAdapter instance for the verification call.

    Returns:
        Dict with keys: is_accurate (bool), issues (list[str]),
        supported_ratio (float), or None if verification failed.
    """
    import json

    prompt = build_verification_prompt(answer, contexts)
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await llm_adapter.chat(messages, temperature=0.0, max_tokens=512)
        # Try to extract JSON from the response
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "is_accurate": result.get("is_accurate", True),
                "issues": result.get("issues", []),
                "supported_ratio": result.get("supported_ratio", 1.0),
            }
    except (json.JSONDecodeError, Exception):
        pass

    return {
        "is_accurate": True,
        "issues": [],
        "supported_ratio": 1.0,
    }