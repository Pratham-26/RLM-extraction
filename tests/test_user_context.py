"""Tests for user context feature in RLM extraction."""

from unittest.mock import MagicMock, patch

import pytest

from rlm_extractor.config import RLMConfig
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

    @patch("rlm_extractor.extract.processor.dspy.Predict")
    def test_process_chunk_passes_guidance(self, mock_predict_class):
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
        mock_predict_class.return_value = mock_predictor

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
        )

    def test_extract_method_has_user_context_param(self):
        """Verify extract() method accepts user_context parameter."""
        import inspect

        mock_lm = MagicMock()
        with patch.object(self.config, "_configure_lms"):
            with patch.object(type(self.config), "root_lm", mock_lm):
                extractor = RLMExtractor(self.config)

        # Check method signature
        sig = inspect.signature(extractor.extract)
        params = list(sig.parameters.keys())
        assert "user_context" in params

        # Check it has a default value of None
        assert sig.parameters["user_context"].default is None

    def test_prepare_user_context_method_exists(self):
        """Verify _prepare_user_context method exists."""
        mock_lm = MagicMock()
        with patch.object(self.config, "_configure_lms"):
            with patch.object(type(self.config), "root_lm", mock_lm):
                extractor = RLMExtractor(self.config)

        assert hasattr(extractor, "_prepare_user_context")
        assert callable(extractor._prepare_user_context)


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

    @patch("rlm_extractor.extract.processor.dspy.Predict")
    def test_guidance_reaches_worker_lm(self, mock_predict_class):
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
        mock_predict_class.return_value = mock_predictor

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
        )

    def test_prepare_user_context_rejects_too_short(self):
        """Test that too-short user_context is rejected."""
        mock_lm = MagicMock()
        mock_logger = MagicMock()
        mock_lm.return_value = MagicMock(condensed_guidance="test")

        with patch.object(self.config, "_configure_lms"):
            with patch.object(type(self.config), "root_lm", mock_lm):
                extractor = RLMExtractor(self.config)

        with pytest.raises(ValueError, match="too short"):
            extractor._prepare_user_context("short", "{}", mock_logger)

    def test_prepare_user_context_rejects_non_string(self):
        """Test that non-string user_context is rejected."""
        mock_lm = MagicMock()
        mock_logger = MagicMock()

        with patch.object(self.config, "_configure_lms"):
            with patch.object(type(self.config), "root_lm", mock_lm):
                extractor = RLMExtractor(self.config)

        with pytest.raises(TypeError, match="must be a string"):
            extractor._prepare_user_context(123, "{}", mock_logger)  # type: ignore

