"""Chunker - Split documents into processable chunks.

Handles:
- Text documents: Fixed-size slices at nearest space
- Images: Store PIL Images for DSPy vision model processing

Uses the Strategy pattern for extensibility - different chunking strategies
can be added by implementing the ChunkingStrategy ABC.
"""

from __future__ import annotations

import abc
import io
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dspy
import fitz  # PyMuPDF
from PIL import Image

if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports
    pass

if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports
    pass

# Supported image file extensions
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

# Supported text file extensions
VALID_TEXT_EXTENSIONS = {".txt", ".md"}

# Supported PDF extension
VALID_PDF_EXTENSION = {".pdf"}

# All supported file extensions
ALL_SUPPORTED_EXTENSIONS = VALID_IMAGE_EXTENSIONS | VALID_TEXT_EXTENSIONS | VALID_PDF_EXTENSION

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


class ImageChunkStrategy(ChunkingStrategy):
    """Strategy for storing PIL Images directly for DSPy vision processing."""

    def chunk(self, images: list[Image.Image], **kwargs) -> list[Chunk]:
        """Store PIL Images directly for DSPy vision model processing.

        DSPy's Image type handles encoding and formatting internally.

        Args:
            images: List of PIL Image objects

        Returns:
            List of Chunk objects with PIL Image content
        """
        chunks = []

        for idx, image in enumerate(images):
            chunks.append(Chunk(idx=idx, content=image))

        return chunks


@dataclass
class Chunk:
    """A single chunk of a document."""

    # Chunk index
    idx: int

    # Chunk content (text string, PIL Image, or dspy.Image)
    content: str | Image.Image | dspy.Image

    # Start position in original document (text mode only)
    start: int | None = None

    # End position in original document (text mode only)
    end: int | None = None


class Chunker:
    """Split documents into processable chunks for RLM extraction.

    Uses the Strategy pattern - delegates to TextChunkStrategy and ImageChunkStrategy
    for actual chunking operations. This makes it easy to add new chunking strategies
    for other document types.
    """

    def __init__(self, chunk_size: int = 2000):
        """Initialize chunker.

        Args:
            chunk_size: Target characters per text chunk (splits at nearest space)
        """
        self._text_strategy = TextChunkStrategy(chunk_size)
        self._image_strategy = ImageChunkStrategy()

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

    def encode_images(self, images: list[Image.Image]) -> list[Chunk]:
        """Store PIL Images for DSPy vision model processing.

        Delegates to ImageChunkStrategy which stores PIL Images directly.
        DSPy's Image type handles encoding and formatting internally.

        Args:
            images: List of PIL Image objects

        Returns:
            List of Chunk objects with PIL Image content
        """
        return self._image_strategy.chunk(images)

    def chunk_image_files(self, image_paths: list[str]) -> list[Chunk]:
        """Load image files and store PIL Images for DSPy vision processing.

        Args:
            image_paths: List of image file paths

        Returns:
            List of Chunk objects with PIL Image content

        Raises:
            FileNotFoundError: If a file does not exist
            ValueError: If a file has an invalid image extension
        """
        images = []
        for path in image_paths:
            # Normalize and validate path
            abs_path = os.path.abspath(path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Image file not found: {path}")

            # Validate file extension
            ext = os.path.splitext(abs_path)[1].lower()
            if ext not in VALID_IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Invalid image file type: {path}. "
                    f"Supported extensions: {', '.join(sorted(VALID_IMAGE_EXTENSIONS))}"
                )

            try:
                images.append(Image.open(abs_path))
            except Exception as e:
                raise ValueError(f"Failed to load image {path}: {e}")

        return self.encode_images(images)

    def count_chunks(self, document: str | list[Image.Image] | list[str]) -> int:
        """Count how many chunks a document will produce.

        Args:
            document: Text string, list of Images, or list of image paths

        Returns:
            Number of chunks
        """
        if isinstance(document, str):
            return len(self.chunk_text(document))
        elif isinstance(document, list):
            if document and isinstance(document[0], Image.Image):
                return len(document)
            elif document and isinstance(document[0], str):
                # Assume image paths
                return len(document)
        return 0

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

    def convert_pdf_to_images(self, pdf_path: str, pdf_config) -> list[Image.Image]:
        """Convert PDF to list of PIL Images using PyMuPDF.

        Args:
            pdf_path: Path to PDF file
            pdf_config: PDFConfig object with conversion settings

        Returns:
            List of PIL Image objects (one per page)

        Raises:
            FileNotFoundError: If PDF file does not exist
            RuntimeError: If PDF conversion fails
        """

        abs_path = os.path.abspath(pdf_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in VALID_PDF_EXTENSION:
            raise ValueError(f"Invalid PDF file type: {pdf_path}. Expected .pdf extension.")

        try:
            # Open PDF document
            doc = fitz.open(abs_path)

            # Determine page range to process
            all_pages = range(len(doc))

            if pdf_config.page_range is not None:
                if isinstance(pdf_config.page_range, tuple):
                    # (start, end) - 1-indexed, inclusive
                    start, end = pdf_config.page_range
                    page_indices = range(start - 1, min(end, len(all_pages)))
                elif isinstance(pdf_config.page_range, list):
                    # [1, 3, 5] - specific page numbers (1-indexed)
                    page_indices = [i - 1 for i in pdf_config.page_range if 1 <= i <= len(doc)]
                else:
                    page_indices = all_pages
            else:
                page_indices = all_pages

            # Apply first_page_only filter
            if pdf_config.first_page_only and page_indices:
                page_indices = [page_indices[0]]

            # Convert pages to images
            images = []
            dpi_scale = pdf_config.dpi / 72.0

            for page_idx in page_indices:
                page = doc.load_page(page_idx)

                # Render page to pixmap
                mat = fitz.Matrix(dpi_scale, dpi_scale)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_bytes = pix.tobytes(output=pdf_config.fmt.upper())
                img = Image.open(io.BytesIO(img_bytes))

                images.append(img)

            doc.close()
            return images

        except Exception as e:
            raise RuntimeError(f"Failed to process PDF: {e}")

    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text as a string

        Raises:
            FileNotFoundError: If PDF file does not exist
            RuntimeError: If PDF text extraction fails
        """

        abs_path = os.path.abspath(pdf_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in VALID_PDF_EXTENSION:
            raise ValueError(f"Invalid PDF file type: {pdf_path}. Expected .pdf extension.")

        try:
            # Open PDF and extract text from all pages
            doc = fitz.open(abs_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text

        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    def chunk_file(
        self,
        file_path: str,
        pdf_config=None,
        pdf_mode="auto",
    ) -> list[Chunk]:
        """Route file to appropriate chunking method based on extension.

        Args:
            file_path: Path to the file (.txt, .md, .pdf, or image)
            pdf_config: PDFConfig object for PDF conversion
            pdf_mode: PDF processing mode ('text', 'image', or 'auto')
                       Default is 'auto' which chooses text for text/markdown files,
                       image for PDFs, or based on file extension.

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
            # Determine PDF processing mode
            if pdf_mode == "auto":
                # Auto: use text for text/markdown-like content, otherwise image
                # For simplicity, treat PDFs as image mode in auto
                mode = "image"
            else:
                mode = pdf_mode

            if mode == "text":
                # Extract text and chunk as text
                if pdf_config is None:
                    from rlm_extractor.config import PDFConfig

                    pdf_config = PDFConfig()
                text_content = self.extract_pdf_text(file_path)
                return self.chunk_text(text_content)
            else:
                # Convert PDF to images and chunk them
                if pdf_config is None:
                    from rlm_extractor.config import PDFConfig

                    pdf_config = PDFConfig()
                images = self.convert_pdf_to_images(file_path, pdf_config)
                return self.encode_images(images)

        elif ext in VALID_IMAGE_EXTENSIONS:
            # Load image file
            return self.chunk_image_files([file_path])

        else:
            raise ValueError(
                f"Unsupported file type: {ext}\n"
                f"Supported types: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
            )


def create_text_chunker(chunk_size: int = 2000) -> Chunker:
    """Factory function to create a text chunker."""
    return Chunker(chunk_size=chunk_size)


def create_image_chunker() -> Chunker:
    """Factory function to create an image chunker (chunk_size ignored for images)."""
    return Chunker()
