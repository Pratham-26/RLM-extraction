"""Chunker - Split documents into processable chunks.

Handles:
- Text documents: Fixed-size slices at nearest space
- PDFs: Text extraction using pypdf
"""

import os
from dataclasses import dataclass

from pypdf import PdfReader

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

# Maximum characters to search backward for a word boundary
MAX_WORD_BOUNDARY_SEARCH = 100


@dataclass
class Chunk:
    """A single chunk of a document."""
    idx: int
    content: str
    start: int | None = None
    end: int | None = None


def chunk_text(text: str, chunk_size: int = 2000) -> list[Chunk]:
    """Split text into fixed-size chunks at nearest spaces."""
    if not text:
        return []

    chunks = []
    idx = 0
    position = 0

    while position < len(text):
        end = min(position + chunk_size, len(text))

        if end < len(text):
            search_start = max(position, end - MAX_WORD_BOUNDARY_SEARCH)
            space_pos = text.rfind(" ", search_start, end)
            if space_pos != -1:
                end = space_pos + 1

        content = text[position:end].strip()
        if content:
            chunks.append(Chunk(idx=idx, content=content, start=position, end=end))
            idx += 1

        position = end

    return chunks


def _load_text_file(file_path: str) -> str:
    """Load text content from a .txt or .md file."""
    from pathlib import Path

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(abs_path)[1].lower()
    if ext not in {".txt", ".md"}:
        raise ValueError(
            f"Invalid text file type: {file_path}. "
            f"Supported: .txt, .md"
        )

    # Use utf-8 with error replacement for robustness
    return Path(abs_path).read_text(encoding="utf-8", errors="replace")


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using pypdf."""
    abs_path = os.path.abspath(pdf_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        reader = PdfReader(abs_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


def chunk_file(file_path: str, chunk_size: int = 2000) -> list[Chunk]:
    """Load file and chunk it based on extension."""
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(abs_path)[1].lower()

    if ext in {".txt", ".md"}:
        content = _load_text_file(file_path)
        return chunk_text(content, chunk_size)
    elif ext == ".pdf":
        content = _extract_pdf_text(file_path)
        return chunk_text(content, chunk_size)
    else:
        raise ValueError(
            f"Unsupported file type: {ext}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
