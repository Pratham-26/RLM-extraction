"""Chunker - Split documents into processable chunks.

Handles:
- Text documents: Fixed-size slices at nearest space
- Images: Convert to base64 encoded strings
"""

import base64
import io
from dataclasses import dataclass
from typing import Union

from PIL import Image


@dataclass
class Chunk:
    """A single chunk of a document."""

    idx: int
    """Chunk index."""

    content: str
    """Chunk content (text or base64 image)."""

    start: int | None = None
    """Start position in original document (text mode only)."""

    end: int | None = None
    """End position in original document (text mode only)."""


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
                # Look for space within last 100 chars
                search_start = max(position, end - 100)
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
        """
        images = []
        for path in image_paths:
            try:
                images.append(Image.open(path))
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


def create_text_chunker(chunk_size: int = 2000) -> Chunker:
    """Factory function to create a text chunker."""
    return Chunker(chunk_size=chunk_size)


def create_image_chunker() -> Chunker:
    """Factory function to create an image chunker (chunk_size ignored for images)."""
    return Chunker()
