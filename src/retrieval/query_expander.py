"""Lightweight query expansion for improved BM25 keyword recall.

For short or informal queries, extracts key terms and generates a keyword-rich
variant for BM25 sparse search. Dense search always uses the original query.
"""

import re


def _get_keywords(text: str) -> list[str]:
    """Extract meaningful Chinese keywords from text using jieba."""
    try:
        import jieba
    except ImportError:
        return []

    # Remove common stop words and short tokens
    stop_words = {"的", "了", "是", "我", "你", "他", "她", "它", "们", "在",
                  "和", "与", "或", "也", "就", "都", "而", "及", "把", "被",
                  "让", "给", "对", "从", "到", "上", "下", "中", "里", "外",
                  "要", "会", "能", "可以", "应该", "需要", "不", "没", "有",
                  "这", "那", "哪", "什么", "怎么", "如何", "为什么", "吗",
                  "呢", "吧", "啊", "哦", "嗯", "哈", "呀", "噢", "唉", "嘛",
                  "一", "二", "三", "个", "种", "些", "很", "非常", "比较",
                  "更", "最", "大", "小", "多", "少", "还", "只", "请", "说",
                  "讲", "告诉", "问", "知道", "想", "看", "听", "找", "查",
                  "帮", "用", "做", "去", "来", "过", "着"}

    keywords = []
    words = jieba.cut(text)
    for w in words:
        w = w.strip()
        if len(w) >= 2 and w not in stop_words and re.search(r"[\u4e00-\u9fff]", w):
            keywords.append(w)
    return keywords


def expand_query(query: str, min_length: int = 5) -> str:
    """Expand a short query with extracted keywords for BM25.

    For queries shorter than `min_length` characters, extracts keywords
    and appends them to help BM25 match more relevant documents.
    Chinese queries of 5+ characters typically contain enough context
    for effective BM25 retrieval without expansion.

    Args:
        query: User's original query string.
        min_length: Queries shorter than this get expanded.

    Returns:
        Expanded query string (or original if long enough).
    """
    query = query.strip()
    if len(query) >= min_length:
        return query

    keywords = _get_keywords(query)
    if not keywords:
        return query

    # For very short queries, use keywords directly as the sparse query
    # For medium-length queries, append keywords
    return " ".join(keywords)