"""Per-format document parsers. Each parser takes a file path and returns extracted text."""

import re
from pathlib import Path


def _read_with_fallback(file_path: Path) -> str:
    """Read a file trying UTF-8 first, then GBK/GB2312."""
    for encoding in ("utf-8", "gbk", "gb2312"):
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode file: {file_path}")


def parse_txt(file_path: Path) -> str:
    """Parse plain text file, trying UTF-8 first, then GBK."""
    return _read_with_fallback(file_path)


def parse_md(file_path: Path) -> str:
    """Parse markdown file. Uses markdown-it-py to convert to plain text while
    preserving heading structure."""
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    text = _read_with_fallback(file_path)

    # Convert markdown to tokens, then extract text content
    tokens = md.parse(text)
    lines = []
    for token in tokens:
        if token.type == "heading_open":
            level = int(token.tag[1])  # h1 -> 1, h2 -> 2
            lines.append("\n" + "#" * level + " ")
        elif token.type == "inline" and token.content:
            lines.append(token.content)
            lines.append("\n")
        elif token.type == "fence" and token.content:
            lines.append(token.content)
            lines.append("\n")
        elif token.type == "paragraph_open":
            pass  # handled by inline
        elif token.type == "hardbreak":
            lines.append("\n")
        elif token.type == "blockquote_open":
            lines.append("\n> ")
        elif token.type == "list_item_open":
            lines.append("\n- ")

    result = "".join(lines)
    if not result.strip():
        # Fallback: just return raw text
        return text
    return result


def parse_pdf(file_path: Path) -> str:
    """Parse PDF file, extracting text page by page."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            # Add page marker for source tracking
            pages.append(f"[第{i+1}页]\n{text.strip()}")
    return "\n\n".join(pages)


def parse_docx(file_path: Path) -> str:
    """Parse Word (.docx) file, extracting text from paragraphs."""
    from docx import Document

    doc = Document(str(file_path))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            # Detect heading style
            if para.style and para.style.name and para.style.name.startswith("Heading"):
                level = para.style.name.split()[-1]
                try:
                    level = int(level)
                except ValueError:
                    level = 1
                paragraphs.append("#" * level + " " + text)
            else:
                paragraphs.append(text)
    return "\n\n".join(paragraphs)


def parse_html(file_path: Path) -> str:
    """Parse HTML file, extracting text content with structure preservation."""
    from bs4 import BeautifulSoup

    html = _read_with_fallback(file_path)
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Convert headings to markdown-style
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            tag.insert_before(soup.new_string("\n" + "#" * level + " "))
            tag.insert_after(soup.new_string("\n"))

    # Also handle common content containers
    for tag in soup.find_all(["article", "main", "section"]):
        # Ensure proper spacing around sections
        tag.insert_before(soup.new_string("\n"))
        tag.insert_after(soup.new_string("\n"))

    text = soup.get_text(separator="\n")
    return text


# Registry of parsers by file extension
PARSER_REGISTRY = {
    ".txt": parse_txt,
    ".md": parse_md,
    ".markdown": parse_md,
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".html": parse_html,
    ".htm": parse_html,
}


def get_parser(file_path: Path):
    """Get the appropriate parser function for a file based on its extension."""
    ext = file_path.suffix.lower()
    parser = PARSER_REGISTRY.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file format: {ext}. Supported: {list(PARSER_REGISTRY.keys())}")
    return parser


def parse_file(file_path: Path) -> str:
    """Parse a single file using the appropriate parser."""
    parser = get_parser(file_path)
    return parser(file_path)