"""Tests for Chunker."""

import pytest

from rlm_extractor.extract.chunker import Chunk, chunk_text


class TestChunker:
    """Test document chunking functionality."""

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=2000)
        assert chunks == []

    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=2000)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"
        assert chunks[0].idx == 0
        assert chunks[0].start == 0

    def test_chunk_text_long(self):
        """Test chunking long text."""
        # Create text 300 chars long
        text = "word " * 60  # ~300 chars

        chunks = chunk_text(text, chunk_size=100)

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
        # Text where chunk boundary falls in middle of word
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunk_text(text, chunk_size=50)

        # Check that no chunk starts with a space (trimmed)
        for chunk in chunks:
            assert not chunk.content.startswith(" ")
            assert not chunk.content.startswith("  ")

    def test_chunk_indices_sequential(self):
        """Test that chunk indices are sequential."""
        text = "a" * 500

        chunks = chunk_text(text, chunk_size=100)
        indices = [chunk.idx for chunk in chunks]

        assert indices == list(range(len(chunks)))

    def test_default_chunk_size(self):
        """Test default chunk size of 2000."""
        text = "a" * 3000
        chunks = chunk_text(text)  # Uses default 2000
        assert len(chunks) == 2
