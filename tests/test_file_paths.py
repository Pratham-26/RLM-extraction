"""Tests for file path support in RLM extraction."""

import os
import tempfile

import pytest

from rlm_extractor.config import RLMConfig
from rlm_extractor.extract.chunker import (
    ALL_SUPPORTED_EXTENSIONS,
    VALID_PDF_EXTENSION,
    VALID_TEXT_EXTENSIONS,
    Chunker,
)


class TestExtensionConstants:
    """Test extension constants are properly defined."""

    def test_text_extensions(self):
        assert ".txt" in VALID_TEXT_EXTENSIONS
        assert ".md" in VALID_TEXT_EXTENSIONS

    def test_pdf_extension(self):
        assert ".pdf" in VALID_PDF_EXTENSION

    def test_all_supported_includes_all(self):
        assert VALID_TEXT_EXTENSIONS.issubset(ALL_SUPPORTED_EXTENSIONS)
        assert VALID_PDF_EXTENSION.issubset(ALL_SUPPORTED_EXTENSIONS)


class TestLoadTextFile:
    """Test text file loading functionality."""

    def test_load_txt_file(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world!\nThis is a test.")
            f.flush()
            temp_path = f.name

        try:
            content = chunker.load_text_file(temp_path)
            assert content == "Hello world!\nThis is a test."
        finally:
            os.unlink(temp_path)

    def test_load_markdown_file(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Header\n\nSome markdown content.")
            f.flush()
            temp_path = f.name

        try:
            content = chunker.load_text_file(temp_path)
            assert content == "# Header\n\nSome markdown content."
        finally:
            os.unlink(temp_path)

    def test_load_file_not_found(self):
        chunker = Chunker()
        with pytest.raises(FileNotFoundError, match="File not found"):
            chunker.load_text_file("nonexistent.txt")

    def test_load_file_invalid_extension(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            f.write("Not really a PDF")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid text file type"):
                chunker.load_text_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_file_utf8_with_bom(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            # UTF-8 with BOM
            f.write(b"\xef\xbb\xbfHello world!")
            f.flush()
            temp_path = f.name

        try:
            content = chunker.load_text_file(temp_path)
            # utf-8-sig removes BOM
            assert "Hello world!" in content
        finally:
            os.unlink(temp_path)

    def test_load_file_latin1_encoding(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
            # Latin-1 encoded text
            f.write("Café".encode("latin-1"))
            f.flush()
            temp_path = f.name

        try:
            content = chunker.load_text_file(temp_path)
            assert content == "Café"
        finally:
            os.unlink(temp_path)


class TestExtractPDFText:
    """Test PDF text extraction using pypdf."""

    def test_extract_pdf_file_not_found(self):
        chunker = Chunker()
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            chunker.extract_pdf_text("nonexistent.pdf")

    def test_extract_pdf_invalid_extension(self):
        chunker = Chunker()
        with pytest.raises(ValueError, match="Invalid PDF file type"):
            chunker.extract_pdf_text("test.txt")


class TestChunkFile:
    """Test chunk_file router method."""

    def test_chunk_file_txt(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Short text")
            f.flush()
            temp_path = f.name

        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) == 1
            assert chunks[0].content == "Short text"
            assert chunks[0].idx == 0
        finally:
            os.unlink(temp_path)

    def test_chunk_file_markdown(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Markdown\n\nContent here.")
            f.flush()
            temp_path = f.name

        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) == 1
            assert "Markdown" in chunks[0].content
        finally:
            os.unlink(temp_path)

    def test_chunk_file_not_found(self):
        chunker = Chunker()
        with pytest.raises(FileNotFoundError, match="File not found"):
            chunker.chunk_file("nonexistent.txt")

    def test_chunk_file_unsupported_extension(self):
        chunker = Chunker()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("Some content")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                chunker.chunk_file(temp_path)
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
