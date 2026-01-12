"""Tests for Chunker."""

import pytest

from rlm_extractor.extract.chunker import Chunk, Chunker


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
