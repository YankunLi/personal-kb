"""Prompt builder for RAG: system prompt + context assembly + source formatting.

Based on Article A: System prompt must require source citation and forbid fabrication.
"""

from typing import Any


SYSTEM_PROMPT_TEMPLATE = """你是一个个人知识库助手。严格遵循以下规则：

1. **只基于提供的文档回答**。文档中没有的信息，直接说"我目前的知识库中没有相关信息"。
2. **引用来源**。每个回答必须标注引用了哪个 [来源N]。
3. **不要编造**。不要添加文档中没有的具体数字、日期、人名。
4. **如果信息不充分**，明确告诉用户你缺少哪些信息。

已检索到的参考文档：
{context}"""


def format_context(
    documents: list[dict[str, Any]],
    max_chars_per_doc: int = 1500,
) -> str:
    """Format retrieved documents into context blocks with source info.

    Args:
        documents: List of document dicts with 'content', 'metadata', 'score' etc.
        max_chars_per_doc: Max characters per context block to prevent lost-in-the-middle.

    Returns:
        Formatted context string with source labels.
    """
    blocks = []
    for i, doc in enumerate(documents, 1):
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        score = doc.get("rerank_score")
        if score is None:
            score = doc.get("score", 0)
        if score is None:
            score = 0.0

        # Truncate long content to prevent lost-in-the-middle effect
        truncated = content[:max_chars_per_doc] if len(content) > max_chars_per_doc else content

        source_file = metadata.get("source_file_basename", metadata.get("source_file", "未知"))
        section = metadata.get("section", "")

        header = f"[来源{i} | 相关度: {score:.2f} | 文件: {source_file}"
        if section:
            header += f" | 章节: {section}"
        header += "]"

        blocks.append(f"{header}\n{truncated}")

    return "\n\n---\n\n".join(blocks)


def build_messages(
    query: str,
    contexts: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
    max_context_chars: int = 1500,
) -> list[dict[str, str]]:
    """Build the full message list for LLM generation.

    Args:
        query: User's question.
        contexts: Retrieved and reranked document chunks.
        chat_history: Previous conversation turns (max 3 rounds kept).
        max_context_chars: Max chars per context block.

    Returns:
        List of message dicts for the LLM API.
    """
    context_str = format_context(contexts, max_chars_per_doc=max_context_chars)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_str)

    messages = [{"role": "system", "content": system_prompt}]

    # Keep only the last 3 rounds (6 messages) of chat history,
    # ensuring the first message is always a user role for API compatibility.
    if chat_history:
        truncated = chat_history[-6:]
        if truncated and truncated[0]["role"] == "assistant":
            truncated = truncated[1:]
        messages.extend(truncated)

    messages.append({"role": "user", "content": query})
    return messages