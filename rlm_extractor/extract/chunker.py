"""Chunker - Split documents into processable chunks.

Handles:
- Text documents: Fixed-size slices at nearest space
- PDFs: Text extraction using pypdf

Uses the Strategy pattern for extensibility - different chunking strategies
can be added by implementing the ChunkingStrategy ABC.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pypdf import PdfReader

if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports
    pass

# Supported text file extensions
VALID_TEXT_EXTENSIONS = {".txt", ".md"}

# Supported PDF extension
VALID_PDF_EXTENSION = {".pdf"}

# All supported file extensions
ALL_SUPPORTED_EXTENSIONS = VALID_TEXT_EXTENSIONS | VALID_PDF_EXTENSION

# Chunking constants
# Maximum characters to search backward for a word boundary.
# Set to 100 as a balance between finding clean breaks and limiting search overhead.
# Most sentences have spaces well within this range, so this avoids splitting words
# in the middle while maintaining reasonable performance.
MAX_WORD_BOUNDARY_SEARCH = 100


class ChunkingStrategy(abc.ABC):
    """Abstract base class for chunking strategies.

    Subclasses implement specific chunking behaviors for different document types.
    """

    @abc.abstractmethod
    def chunk(self, input_data, **kwargs) -> list[Chunk]:
        """Chunk the input data into a list of Chunks.

        Args:
            input_data: The data to chunk (type depends on strategy)
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of Chunk objects
        """
        pass


class TextChunkStrategy(ChunkingStrategy):
    """Strategy for chunking text documents into fixed-size chunks."""

    def __init__(self, chunk_size: int = 2000):
        """Initialize text chunking strategy.

        Args:
            chunk_size: Target characters per chunk
        """
        self.chunk_size = chunk_size

    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """Split text into fixed-size chunks at nearest spaces.

        Args:
            text: Document text to chunk

        Returns:
            List of Chunk objects with idx, content, start, end
        """
        if not text:
            return []

        chunks = []
        idx = 0
        position = 0

        while position < len(text):
            # Calculate end position
            end = min(position + self.chunk_size, len(text))

            # If not at end of text, find nearest space to avoid cutting words
            if end < len(text):
                # Look for space within last MAX_WORD_BOUNDARY_SEARCH chars
                search_start = max(position, end - MAX_WORD_BOUNDARY_SEARCH)
                space_pos = text.rfind(" ", search_start, end)

                if space_pos != -1:
                    end = space_pos + 1  # Include the space

            # Extract chunk content
            content = text[position:end].strip()

            if content:  # Only add non-empty chunks
                chunks.append(Chunk(idx=idx, content=content, start=position, end=end))
                idx += 1

            position = end

        return chunks


@dataclass
class Chunk:
    """A single chunk of a document."""

    # Chunk index
    idx: int

    # Chunk content (text string)
    content: str

    # Start position in original document (text mode only)
    start: int | None = None

    # End position in original document (text mode only)
    end: int | None = None


class Chunker:
    """Split documents into processable chunks for RLM extraction.

    Uses the Strategy pattern - delegates to TextChunkStrategy
    for actual chunking operations.
    """

    def __init__(self, chunk_size: int = 2000):
        """Initialize chunker.

        Args:
            chunk_size: Target characters per text chunk (splits at nearest space)
        """
        self._text_strategy = TextChunkStrategy(chunk_size)

    @property
    def chunk_size(self) -> int:
        """Get the current chunk size for text chunking."""
        return self._text_strategy.chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int) -> None:
        """Set a new chunk size for text chunking."""
        self._text_strategy = TextChunkStrategy(value)

    def chunk_text(self, text: str) -> list[Chunk]:
        """Split text into fixed-size chunks at nearest spaces.

        Delegates to TextChunkStrategy.

        Args:
            text: Document text to chunk

        Returns:
            List of Chunk objects with idx, content, start, end
        """
        return self._text_strategy.chunk(text)

    def count_chunks(self, document: str | list[str]) -> int:
        """Count how many chunks a document will produce.

        Args:
            document: Text string or list of file paths

        Returns:
            Number of chunks
        """
        if isinstance(document, str):
            return len(self.chunk_text(document))
        elif isinstance(document, list):
            # List of file paths - count chunks for each
            total = 0
            for path in document:
                if isinstance(path, str):
                    try:
                        content = self.load_text_file(path) if self._get_ext(path) in VALID_TEXT_EXTENSIONS else self.extract_pdf_text(path)
                        total += len(self.chunk_text(content))
                    except Exception:
                        pass
            return total
        return 0

    def _get_ext(self, path: str) -> str:
        """Get file extension from path."""
        return os.path.splitext(path)[1].lower()

    def load_text_file(self, file_path: str) -> str:
        """Load text content from a .txt or .md file.

        Args:
            file_path: Path to the text file

        Returns:
            File content as a string

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file has an invalid text extension or cannot be decoded
        """
        from pathlib import Path

        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in VALID_TEXT_EXTENSIONS:
            raise ValueError(
                f"Invalid text file type: {file_path}. "
                f"Supported text extensions: {', '.join(sorted(VALID_TEXT_EXTENSIONS))}"
            )

        # Try multiple encodings
        path = Path(abs_path)
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        errors = []

        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as e:
                errors.append(f"{encoding}: {e}")
                continue

        raise ValueError(
            f"Could not decode file with any supported encoding. "
            f"Tried: {', '.join(encodings)}. Errors: {'; '.join(errors)}"
        )

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using pypdf.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as a string

        Raises:
            ValueError: If file has wrong extension
            FileNotFoundError: If PDF file does not exist
            RuntimeError: If PDF text extraction fails
        """
        # Check extension first before checking file existence
        ext = os.path.splitext(pdf_path)[1].lower()
        if ext not in VALID_PDF_EXTENSION:
            raise ValueError(f"Invalid PDF file type: {pdf_path}. Expected .pdf extension.")

        abs_path = os.path.abspath(pdf_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            reader = PdfReader(abs_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    def chunk_file(
        self,
        file_path: str,
    ) -> list[Chunk]:
        """Route file to appropriate chunking method based on extension.

        Args:
            file_path: Path to the file (.txt, .md, or .pdf)

        Returns:
            List of Chunk objects

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file type is not supported
        """
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(abs_path)[1].lower()

        if ext in VALID_TEXT_EXTENSIONS:
            # Load text file and chunk it
            content = self.load_text_file(file_path)
            return self.chunk_text(content)

        elif ext in VALID_PDF_EXTENSION:
            # Extract text and chunk as text
            text_content = self.extract_pdf_text(file_path)
            return self.chunk_text(text_content)

        else:
            raise ValueError(
                f"Unsupported file type: {ext}\n"
                f"Supported types: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
            )


def create_text_chunker(chunk_size: int = 2000) -> Chunker:
    """Factory function to create a text chunker."""
    return Chunker(chunk_size=chunk_size)
