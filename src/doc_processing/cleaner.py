"""Text cleaning and normalization for extracted document content."""

import html
import re
import unicodedata


def clean_html_text(text: str) -> str:
    """Clean text extracted from HTML documents.

    Two-phase cleaning as recommended in the design:
    Phase 1: Remove HTML artifacts (already done by parser, but catch stragglers)
    Phase 2: Normalize whitespace, punctuation, and Unicode.
    """
    # Phase 1: Remove any remaining HTML/XML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode all HTML entities to their Unicode equivalents
    text = html.unescape(text)

    # Phase 2: Normalize
    text = normalize_text(text)
    return text


def normalize_text(text: str) -> str:
    """Normalize text for consistent processing.

    - Unicode NFC normalization
    - Collapse multiple newlines to double newlines
    - Collapse multiple spaces (preserving newlines)
    - Strip leading/trailing whitespace per line
    - Remove zero-width and control characters
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters and control characters (except newline/tab)
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\uFEFF]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple newlines (3+) to double newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces/tabs (preserving newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start and end
    text = text.strip()

    return text


def clean_text(text: str, file_type: str = "txt") -> str:
    """Clean text based on its source format.

    Args:
        text: Raw extracted text.
        file_type: Source file extension (e.g., 'html', 'txt', 'pdf').

    Returns:
        Cleaned and normalized text.
    """
    if file_type in (".html", ".htm"):
        return clean_html_text(text)
    else:
        return normalize_text(text)