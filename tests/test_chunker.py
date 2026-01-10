"""Tests for Chunker."""

import pytest

from rlm.extract.chunker import Chunk, Chunker


class TestChunker:
    """Test document chunking functionality."""

    def test_init(self):
        """Test chunker initialization."""
        chunker = Chunker(chunk_size=1000)
        assert chunker.chunk_size == 1000

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = Chunker()
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        chunker = Chunker(chunk_size=2000)
        text = "Short text"
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"
        assert chunks[0].idx == 0
        assert chunks[0].start == 0

    def test_chunk_text_long(self):
        """Test chunking long text."""
        chunker = Chunker(chunk_size=100)
        # Create text 300 chars long
        text = "word " * 60  # ~300 chars

        chunks = chunker.chunk_text(text)

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Check chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.idx == i
            assert chunk.content  # Non-empty
            assert chunk.start is not None
            assert chunk.end is not None

    def test_chunk_splits_at_space(self):
        """Test that chunks split at nearest space."""
        chunker = Chunker(chunk_size=50)

        # Text where chunk boundary falls in middle of word
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunker.chunk_text(text)

        # Check that no chunk starts with a space (trimmed)
        for chunk in chunks:
            assert not chunk.content.startswith(" ")
            assert not chunk.content.startswith("  ")

    def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        chunker = Chunker(chunk_size=100)
        text = "a" * 500

        chunks = chunker.chunk_text(text)
        indices = [chunk.idx for chunk in chunks]

        assert indices == list(range(len(chunks)))

    def test_count_chunks_text(self):
        """Test counting chunks for text."""
        chunker = Chunker(chunk_size=100)
        text = "a" * 500

        count = chunker.count_chunks(text)
        chunks = chunker.chunk_text(text)

        assert count == len(chunks)

    def test_count_chunks_images(self):
        """Test counting chunks for images."""
        from PIL import Image

        chunker = Chunker()

        # Create dummy images
        images = [Image.new("RGB", (100, 100)) for _ in range(3)]

        count = chunker.count_chunks(images)
        assert count == 3

    def test_encode_images(self):
        """Test encoding images to base64."""
        from PIL import Image

        chunker = Chunker()

        # Create dummy images
        images = [
            Image.new("RGB", (100, 100), color="red"),
            Image.new("RGB", (100, 100), color="blue"),
        ]

        chunks = chunker.encode_images(images)

        assert len(chunks) == 2
        assert chunks[0].idx == 0
        assert chunks[1].idx == 1
        assert chunks[0].content  # Base64 string
        assert chunks[1].content

    def test_chunk_image_files(self):
        """Test loading and chunking image files."""
        from PIL import Image
        import tempfile

        chunker = Chunker()

        # Create temporary image files
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i, color in enumerate(["red", "blue", "green"]):
                path = f"{tmpdir}/img{i}.png"
                img = Image.new("RGB", (50, 50), color=color)
                img.save(path)
                paths.append(path)

            chunks = chunker.chunk_image_files(paths)

            assert len(chunks) == 3
            for i, chunk in enumerate(chunks):
                assert chunk.idx == i
                assert chunk.content
