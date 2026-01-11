"""Tests for user context feature in RLM extraction."""

from unittest.mock import MagicMock, patch

import pytest

from rlm_extractor.config import RLMConfig
from rlm_extractor.extract.chunker import Chunker
from rlm_extractor.extract.extractor import RLMExtractor
from rlm_extractor.repl import REPLState


class TestREPLStateGuidance:
    """Test REPLState condensed_guidance field."""

    def test_default_condensed_guidance(self):
        state = REPLState()
        assert state.condensed_guidance == ""

    def test_set_condensed_guidance(self):
        state = REPLState()
        guidance = "Extract dates in DD/MM/YYYY format only."
        state.set_condensed_guidance(guidance)
        assert state.condensed_guidance == guidance

    def test_set_condensed_guidance_overwrites(self):
        state = REPLState()
        state.set_condensed_guidance("First guidance")
        state.set_condensed_guidance("Second guidance")
        assert state.condensed_guidance == "Second guidance"


class TestChunkProcessorGuidance:
    """Test ChunkProcessor with condensed_guidance."""

    def test_processor_accepts_guidance(self):
        from rlm_extractor.extract.processor import ChunkProcessor

        mock_lm = MagicMock()
        processor = ChunkProcessor(
            worker_lm=mock_lm,
            condensed_guidance="Follow these instructions",
        )
        assert processor.condensed_guidance == "Follow these instructions"

    def test_processor_default_empty_guidance(self):
        from rlm_extractor.extract.processor import ChunkProcessor

        mock_lm = MagicMock()
        processor = ChunkProcessor(worker_lm=mock_lm)
        assert processor.condensed_guidance == ""

    @patch("rlm_extractor.extract.processor.ChunkProcessor._get_predictor")
    def test_process_chunk_passes_guidance(self, mock_get_predictor):
        from rlm_extractor.extract.processor import ChunkProcessor
        from rlm_extractor.extract.chunker import Chunk

        # Setup mocks
        mock_predictor = MagicMock()
        mock_result = MagicMock()
        mock_result.gist = "Test gist"
        mock_result.extracted = "{}"
        mock_result.confidence = "high"
        mock_result.missing_fields = "[]"
        mock_predictor.return_value = mock_result
        mock_get_predictor.return_value = mock_predictor

        mock_lm = MagicMock()
        processor = ChunkProcessor(
            worker_lm=mock_lm,
            condensed_guidance="Use DD/MM/YYYY date format",
        )

        chunk = Chunk(idx=0, content="Test content")

        # Process chunk
        result = processor.process_chunk(chunk, "schema: test")

        # Verify condensed_guidance was passed to predictor
        mock_predictor.assert_called_once()
        call_kwargs = mock_predictor.call_args.kwargs
        assert "condensed_guidance" in call_kwargs
        assert call_kwargs["condensed_guidance"] == "Use DD/MM/YYYY date format"


class TestContextCondensationSignature:
    """Test the ContextCondensationSignature."""

    def test_signature_exists(self):
        from rlm_extractor.signatures import ContextCondensationSignature

        assert hasattr(ContextCondensationSignature, "__name__")
        assert "ContextCondensation" in ContextCondensationSignature.__name__

    def test_signature_has_user_context_input(self):
        from rlm_extractor.signatures import ContextCondensationSignature

        # DSPy signatures store fields in __fields__ (Pydantic model)
        field_names = list(ContextCondensationSignature.model_fields.keys())
        assert "user_context" in field_names

    def test_signature_has_yaml_schema_input(self):
        from rlm_extractor.signatures import ContextCondensationSignature

        field_names = list(ContextCondensationSignature.model_fields.keys())
        assert "yaml_schema" in field_names

    def test_signature_has_condensed_guidance_output(self):
        from rlm_extractor.signatures import ContextCondensationSignature

        field_names = list(ContextCondensationSignature.model_fields.keys())
        assert "condensed_guidance" in field_names


class TestWorkerExtractionSignatureGuidance:
    """Test that WorkerExtractionSignature includes condensed_guidance."""

    def test_worker_signature_has_guidance_input(self):
        from rlm_extractor.signatures import WorkerExtractionSignature

        field_names = list(WorkerExtractionSignature.model_fields.keys())
        assert "condensed_guidance" in field_names

    def test_worker_guidance_has_default(self):
        from rlm_extractor.signatures import WorkerExtractionSignature

        # condensed_guidance should be a field with default value
        field_names = list(WorkerExtractionSignature.model_fields.keys())
        assert "condensed_guidance" in field_names


class TestExtractorUserContext:
    """Test RLMExtractor with user_context parameter."""

    def setup_method(self):
        # Create a mock config
        self.config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
            worker_vision_model="openai/gpt-4o",
        )

    def test_extract_method_has_user_context_param(self):
        """Verify extract() method accepts user_context parameter."""
        import inspect

        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        # Check method signature
        sig = inspect.signature(extractor.extract)
        params = list(sig.parameters.keys())
        assert "user_context" in params

        # Check it has a default value of None
        assert sig.parameters["user_context"].default is None

    def test_condense_user_context_method_exists(self):
        """Verify _condense_user_context method exists."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        assert hasattr(extractor, "_condense_user_context")
        assert callable(extractor._condense_user_context)


class TestIntegrationUserContextFlow:
    """Integration tests for user context flow through the system."""

    def test_user_context_flows_to_repl_state(self):
        """Test that condensed guidance ends up in REPLState."""
        state = REPLState()
        guidance = "Extract dates in DD/MM/YYYY format only."

        state.set_condensed_guidance(guidance)
        assert state.condensed_guidance == guidance

    def test_guidance_flows_from_extractor_to_processor(self):
        """Test that guidance flows from extractor to ChunkProcessor."""
        from rlm_extractor.extract.processor import ChunkProcessor

        mock_lm = MagicMock()
        guidance = "Use ISO 8601 date format"

        processor = ChunkProcessor(
            worker_lm=mock_lm,
            condensed_guidance=guidance,
        )

        assert processor.condensed_guidance == guidance

    @patch("rlm_extractor.extract.processor.ChunkProcessor._get_predictor")
    def test_guidance_reaches_worker_lm(self, mock_get_predictor):
        """Test that guidance reaches the worker LM during processing."""
        from rlm_extractor.extract.processor import ChunkProcessor
        from rlm_extractor.extract.chunker import Chunk

        mock_predictor = MagicMock()
        mock_result = MagicMock()
        mock_result.gist = "Summary"
        mock_result.extracted = "{}"
        mock_result.confidence = "high"
        mock_result.missing_fields = "[]"
        mock_predictor.return_value = mock_result
        mock_get_predictor.return_value = mock_predictor

        mock_lm = MagicMock()
        guidance = "BP means blood pressure"

        processor = ChunkProcessor(worker_lm=mock_lm, condensed_guidance=guidance)
        chunk = Chunk(idx=0, content="Patient has BP of 120/80")

        processor.process_chunk(chunk, "vitals: {}")

        # Verify the guidance was passed
        call_kwargs = mock_predictor.call_args.kwargs
        assert call_kwargs["condensed_guidance"] == guidance


class TestUserContextValidation:
    """Test user_context validation."""

    def setup_method(self):
        self.config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
            worker_vision_model="openai/gpt-4o",
        )

    def test_validate_user_context_accepts_valid_input(self):
        """Test that valid user_context passes validation."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        # Should not raise
        extractor._validate_user_context("This is a valid user context with enough characters")

    def test_validate_user_context_rejects_empty(self):
        """Test that empty user_context is rejected."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor._validate_user_context("")

    def test_validate_user_context_rejects_too_short(self):
        """Test that too-short user_context is rejected."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        with pytest.raises(ValueError, match="too short"):
            extractor._validate_user_context("short")

    def test_validate_user_context_rejects_too_long(self):
        """Test that too-long user_context is rejected."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        long_context = "x" * 10001  # Default max is 10,000
        with pytest.raises(ValueError, match="too long"):
            extractor._validate_user_context(long_context)

    def test_validate_user_context_respects_config_limit(self):
        """Test that validation respects custom max_user_context_chars."""
        config = RLMConfig(
            root_model="openai/gpt-4o",
            worker_text_model="openai/gpt-4o-mini",
            worker_vision_model="openai/gpt-4o",
            max_user_context_chars=100,
        )
        mock_lm = MagicMock()
        with patch.object(config, "configure_dspy"):
            with patch.object(config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(config)

        # Should fail with custom limit
        long_context = "x" * 101
        with pytest.raises(ValueError, match="Maximum: 100"):
            extractor._validate_user_context(long_context)

    def test_validate_user_context_rejects_non_string(self):
        """Test that non-string user_context is rejected."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        with pytest.raises(TypeError, match="must be a string"):
            extractor._validate_user_context(123)  # type: ignore

    def test_sanitize_user_context_removes_control_chars(self):
        """Test that sanitization removes control characters."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        # Contains control character \x00
        dirty = "Valid context\x00with null byte"
        clean = extractor._sanitize_user_context(dirty)

        assert "\x00" not in clean
        assert clean == "Valid contextwith null byte"

    def test_sanitize_user_context_limits_newlines(self):
        """Test that sanitization limits consecutive newlines."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        dirty = "Line one\n\n\n\nLine two"  # 4 consecutive newlines
        clean = extractor._sanitize_user_context(dirty)

        assert clean == "Line one\n\nLine two"

    def test_sanitize_user_context_trims_whitespace(self):
        """Test that sanitization trims leading/trailing whitespace."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        dirty = "  Valid context with spaces  "
        clean = extractor._sanitize_user_context(dirty)

        assert clean == "Valid context with spaces"

    def test_sanitize_user_context_preserves_tabs(self):
        """Test that sanitization preserves tab characters."""
        mock_lm = MagicMock()
        with patch.object(self.config, "configure_dspy"):
            with patch.object(self.config, "get_root_lm", return_value=mock_lm):
                extractor = RLMExtractor(self.config)

        context = "Line one\n\tLine two"
        clean = extractor._sanitize_user_context(context)

        assert "\t" in clean
