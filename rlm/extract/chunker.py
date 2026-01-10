"""Chunker - Split documents into processable chunks.

Handles:
- Text documents: Fixed-size slices at nearest space
- Images: Convert to base64 encoded strings
"""

import base64
import io
import os
from dataclasses import dataclass
from typing import Union

from PIL import Image


# Supported image file extensions
VALID_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"
}

# Supported text file extensions
VALID_TEXT_EXTENSIONS = {".txt", ".md"}

# Supported PDF extension
VALID_PDF_EXTENSION = {".pdf"}

# All supported file extensions
ALL_SUPPORTED_EXTENSIONS = (
    VALID_IMAGE_EXTENSIONS | VALID_TEXT_EXTENSIONS | VALID_PDF_EXTENSION
)

# Chunking constants
# Maximum characters to search backward for a word boundary
MAX_WORD_BOUNDARY_SEARCH = 100


@dataclass
class Chunk:
    """A single chunk of a document."""

    # Chunk index
    idx: int

    # Chunk content (text or base64 image)
    content: str

    # Start position in original document (text mode only)
    start: int | None = None

    # End position in original document (text mode only)
    end: int | None = None


class Chunker:
    """Split documents into processable chunks for RLM extraction."""

    def __init__(self, chunk_size: int = 2000):
        """Initialize chunker.

        Args:
            chunk_size: Target characters per text chunk (splits at nearest space)
        """
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> list[Chunk]:
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

    def encode_images(self, images: list[Image.Image]) -> list[Chunk]:
        """Convert PIL Images to base64 encoded chunks.

        Args:
            images: List of PIL Image objects

        Returns:
            List of Chunk objects with base64 content
        """
        chunks = []

        for idx, image in enumerate(images):
            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()
            base64_str = base64.b64encode(img_bytes).decode("utf-8")

            chunks.append(Chunk(idx=idx, content=base64_str))

        return chunks

    def chunk_image_files(self, image_paths: list[str]) -> list[Chunk]:
        """Load image files and convert to base64 chunks.

        Args:
            image_paths: List of image file paths

        Returns:
            List of Chunk objects with base64 content

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

    def count_chunks(self, document: Union[str, list[Image.Image], list[str]]) -> int:
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
        """Convert PDF to list of PIL Images using pdf2image.

        Args:
            pdf_path: Path to the PDF file
            pdf_config: PDFConfig object with conversion settings

        Returns:
            List of PIL Image objects (one per page)

        Raises:
            FileNotFoundError: If the PDF file does not exist
            RuntimeError: If poppler is not installed or PDF conversion fails
        """
        abs_path = os.path.abspath(pdf_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        ext = os.path.splitext(abs_path)[1].lower()
        if ext not in VALID_PDF_EXTENSION:
            raise ValueError(
                f"Invalid PDF file type: {pdf_path}. "
                f"Expected .pdf extension."
            )

        from pdf2image import convert_from_path
        from pdf2image.exceptions import (
            PDFInfoNotInstalledError,
            PDFPageCountError,
            PDFSyntaxError,
        )

        try:
            # Convert PDF to images
            images = convert_from_path(
                abs_path,
                dpi=pdf_config.dpi,
                fmt=pdf_config.fmt,
                thread_count=pdf_config.thread_count,
            )

            # Apply page range filters
            if pdf_config.first_page_only and images:
                return images[:1]

            if pdf_config.page_range is not None:
                if isinstance(pdf_config.page_range, tuple):
                    # (start, end) - 1-indexed, inclusive
                    start, end = pdf_config.page_range
                    images = images[start - 1 : end]
                elif isinstance(pdf_config.page_range, list):
                    # [1, 3, 5] - specific page numbers (1-indexed)
                    page_indices = [p - 1 for p in pdf_config.page_range]
                    images = [images[i] for i in page_indices if 0 <= i < len(images)]

            return images

        except PDFInfoNotInstalledError as e:
            raise RuntimeError(
                "PDF processing requires Poppler to be installed.\n"
                "Install with:\n"
                "  macOS: brew install poppler\n"
                "  Ubuntu: sudo apt-get install poppler-utils\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n"
                f"Original error: {e}"
            )
        except (PDFPageCountError, PDFSyntaxError) as e:
            raise RuntimeError(f"Failed to process PDF: {e}")

    def chunk_file(
        self,
        file_path: str,
        pdf_config=None,
    ) -> list[Chunk]:
        """Route file to appropriate chunking method based on extension.

        Args:
            file_path: Path to the file (.txt, .md, .pdf, or image)
            pdf_config: PDFConfig object for PDF conversion

        Returns:
            List of Chunk objects

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file type is not supported
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
            # Convert PDF to images and chunk them
            if pdf_config is None:
                from rlm.config import PDFConfig

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
