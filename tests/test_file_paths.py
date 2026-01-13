"""Tests for file path support in RLM extraction."""

import os
import tempfile

import pytest

from rlm_extractor.config import RLMConfig
from rlm_extractor.extract.chunker import SUPPORTED_EXTENSIONS, chunk_file


class TestExtensionConstants:
    """Test extension constants are properly defined."""

    def test_supported_extensions(self):
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS


class TestChunkFile:
    """Test chunk_file module function."""

    def test_chunk_file_txt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Short text")
            f.flush()
            temp_path = f.name

        try:
            chunks = chunk_file(temp_path)
            assert len(chunks) == 1
            assert chunks[0].content == "Short text"
            assert chunks[0].idx == 0
        finally:
            os.unlink(temp_path)

    def test_chunk_file_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Markdown\n\nContent here.")
            f.flush()
            temp_path = f.name

        try:
            chunks = chunk_file(temp_path)
            assert len(chunks) == 1
            assert "Markdown" in chunks[0].content
        finally:
            os.unlink(temp_path)

    def test_chunk_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            chunk_file("nonexistent.txt")

    def test_chunk_file_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Some content")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                chunk_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestRLMConfig:
    """Test RLMConfig initialization."""

    def test_rlm_config_init(self):
        config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
        )
        assert config.root_model == "openai/gpt-4o"
        assert config.worker_text_model == "openai/gpt-4o-mini"
