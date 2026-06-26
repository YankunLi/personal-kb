"""File discovery and format dispatch for document loading."""

import logging
import os
from pathlib import Path
from typing import Iterator

from .parsers import PARSER_REGISTRY, parse_file
from .cleaner import clean_text

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = set(PARSER_REGISTRY.keys())


def discover_files(
    path: Path,
    recursive: bool = True,
) -> list[Path]:
    """Discover all supported document files in a directory or single file.

    Args:
        path: File or directory path.
        recursive: If True, recurse into subdirectories.

    Returns:
        List of file paths to supported documents.
    """
    path = Path(os.path.expanduser(path))

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    if path.is_dir():
        files = []
        pattern = "**/*" if recursive else "*"
        for f in path.glob(pattern):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(f)
        return sorted(files)

    raise FileNotFoundError(f"Path not found: {path}")


def load_document(file_path: Path) -> dict:
    """Load a single document: parse and clean.

    Args:
        file_path: Path to the document file.

    Returns:
        Dict with keys: 'file_path', 'file_type', 'text', 'file_name'.
    """
    ext = file_path.suffix.lower()
    raw_text = parse_file(file_path)
    cleaned_text = clean_text(raw_text, ext)

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "file_type": ext,
        "text": cleaned_text,
    }


def load_documents(
    path: Path,
    recursive: bool = True,
    progress_callback=None,
) -> tuple[Iterator[dict], list[int]]:
    """Load all documents from a path, yielding parsed and cleaned results.

    Args:
        path: File or directory path.
        recursive: If True, recurse into subdirectories.
        progress_callback: Optional callback(file_path, index, total) for progress.

    Returns:
        Tuple of (iterator over parsed docs, count of failed files).
    """
    files = discover_files(path, recursive=recursive)
    total = len(files)
    failed = [0]  # Use list for mutable closure capture

    def _iter():
        for i, file_path in enumerate(files):
            if progress_callback:
                progress_callback(file_path, i, total)
            try:
                yield load_document(file_path)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", file_path.name, e)
                failed[0] += 1
                continue

    return _iter(), failed