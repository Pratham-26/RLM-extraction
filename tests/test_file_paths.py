"""Tests for file path support in RLM extraction."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from rlm.config import PDFConfig, RLMConfig
from rlm.extract.chunker import (
    ALL_SUPPORTED_EXTENSIONS,
    Chunker,
    VALID_IMAGE_EXTENSIONS,
    VALID_PDF_EXTENSION,
    VALID_TEXT_EXTENSIONS,
)

# Import pdf2image if available for PDF tests
try:
    import pdf2image
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


class TestExtensionConstants:
    """Test extension constants are properly defined."""

    def test_text_extensions(self):
        assert ".txt" in VALID_TEXT_EXTENSIONS
        assert ".md" in VALID_TEXT_EXTENSIONS

    def test_pdf_extension(self):
        assert ".pdf" in VALID_PDF_EXTENSION

    def test_image_extensions(self):
        assert ".png" in VALID_IMAGE_EXTENSIONS
        assert ".jpg" in VALID_IMAGE_EXTENSIONS
        assert ".jpeg" in VALID_IMAGE_EXTENSIONS

    def test_all_supported_includes_all(self):
        assert VALID_TEXT_EXTENSIONS.issubset(ALL_SUPPORTED_EXTENSIONS)
        assert VALID_PDF_EXTENSION.issubset(ALL_SUPPORTED_EXTENSIONS)
        assert VALID_IMAGE_EXTENSIONS.issubset(ALL_SUPPORTED_EXTENSIONS)


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
            # utf-8-sig removes the BOM
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


@pytest.mark.skipif(not PDF2IMAGE_AVAILABLE, reason="pdf2image not installed")
class TestConvertPDF:
    """Test PDF to image conversion."""

    def test_convert_pdf_file_not_found(self):
        chunker = Chunker()
        pdf_config = PDFConfig()

        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            chunker.convert_pdf_to_images("nonexistent.pdf", pdf_config)

    def test_convert_pdf_invalid_extension(self):
        chunker = Chunker()
        pdf_config = PDFConfig()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Not a PDF")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid PDF file type"):
                chunker.convert_pdf_to_images(temp_path, pdf_config)
        finally:
            os.unlink(temp_path)

    @patch("rlm.extract.chunker.convert_from_path")
    def test_convert_pdf_success(self, mock_convert):
        # Mock the pdf2image conversion
        mock_image = MagicMock(spec=Image.Image)
        mock_convert.return_value = [mock_image]

        chunker = Chunker()
        pdf_config = PDFConfig()

        # Create a temporary file with .pdf extension
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            f.flush()
            temp_path = f.name

        try:
            result = chunker.convert_pdf_to_images(temp_path, pdf_config)
            assert result == [mock_image]
            mock_convert.assert_called_once()
        finally:
            os.unlink(temp_path)

    @patch("rlm.extract.chunker.convert_from_path")
    def test_convert_pdf_first_page_only(self, mock_convert):
        mock_images = [MagicMock(spec=Image.Image) for _ in range(5)]
        mock_convert.return_value = mock_images

        chunker = Chunker()
        pdf_config = PDFConfig(first_page_only=True)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            f.flush()
            temp_path = f.name

        try:
            result = chunker.convert_pdf_to_images(temp_path, pdf_config)
            assert len(result) == 1
            assert result[0] == mock_images[0]
        finally:
            os.unlink(temp_path)

    @patch("rlm.extract.chunker.convert_from_path")
    def test_convert_pdf_page_range_tuple(self, mock_convert):
        mock_images = [MagicMock(spec=Image.Image) for _ in range(5)]
        mock_convert.return_value = mock_images

        chunker = Chunker()
        pdf_config = PDFConfig(page_range=(2, 4))  # Pages 2-4

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            f.flush()
            temp_path = f.name

        try:
            result = chunker.convert_pdf_to_images(temp_path, pdf_config)
            assert len(result) == 3  # Pages 2, 3, 4
            assert result[0] == mock_images[1]
            assert result[1] == mock_images[2]
            assert result[2] == mock_images[3]
        finally:
            os.unlink(temp_path)

    @patch("rlm.extract.chunker.convert_from_path")
    def test_convert_pdf_page_range_list(self, mock_convert):
        mock_images = [MagicMock(spec=Image.Image) for _ in range(5)]
        mock_convert.return_value = mock_images

        chunker = Chunker()
        pdf_config = PDFConfig(page_range=[1, 3, 5])  # Specific pages

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            f.flush()
            temp_path = f.name

        try:
            result = chunker.convert_pdf_to_images(temp_path, pdf_config)
            assert len(result) == 3
            assert result[0] == mock_images[0]
            assert result[1] == mock_images[2]
            assert result[2] == mock_images[4]
        finally:
            os.unlink(temp_path)

    @patch("rlm.extract.chunker.convert_from_path")
    def test_convert_pdf_poppler_not_installed(self, mock_convert):
        mock_convert.side_effect = PDFInfoNotInstalledError("Poppler not found")

        chunker = Chunker()
        pdf_config = PDFConfig()

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="PDF processing requires Poppler"):
                chunker.convert_pdf_to_images(temp_path, pdf_config)
        finally:
            os.unlink(temp_path)


class TestChunkFile:
    """Test the chunk_file router method."""

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

    @pytest.mark.skipif(not PDF2IMAGE_AVAILABLE, reason="pdf2image not installed")
    @patch("rlm.extract.chunker.convert_from_path")
    def test_chunk_file_pdf(self, mock_convert):
        mock_image = MagicMock(spec=Image.Image)
        mock_convert.return_value = [mock_image]

        chunker = Chunker()

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4")
            f.flush()
            temp_path = f.name

        try:
            chunks = chunker.chunk_file(temp_path)
            assert len(chunks) == 1
            assert chunks[0].idx == 0
        finally:
            os.unlink(temp_path)


class TestPDFConfig:
    """Test PDFConfig dataclass."""

    def test_default_values(self):
        config = PDFConfig()
        assert config.dpi == 200
        assert config.page_range is None
        assert config.first_page_only is False
        assert config.fmt == "png"
        assert config.thread_count == 1

    def test_custom_values(self):
        config = PDFConfig(
            dpi=300,
            page_range=(1, 5),
            first_page_only=True,
            fmt="jpeg",
            thread_count=4,
        )
        assert config.dpi == 300
        assert config.page_range == (1, 5)
        assert config.first_page_only is True
        assert config.fmt == "jpeg"
        assert config.thread_count == 4


class TestRLMConfigWithPDF:
    """Test RLMConfig includes PDFConfig."""

    def test_rlm_config_has_pdf_config(self):
        config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
            worker_vision_model="openai/gpt-4o",
        )
        assert hasattr(config, "pdf_config")
        assert isinstance(config.pdf_config, PDFConfig)

    def test_rlm_config_custom_pdf_config(self):
        pdf_config = PDFConfig(dpi=300, first_page_only=True)
        config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
            worker_vision_model="openai/gpt-4o",
            pdf_config=pdf_config,
        )
        assert config.pdf_config.dpi == 300
        assert config.pdf_config.first_page_only is True
