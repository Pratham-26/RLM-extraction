"""Tests for RLMConfig."""

from rlm_extractor.config import RLMConfig


class TestRLMConfig:
    """Test RLM configuration."""

    def test_init_minimal(self):
        """Test minimal config initialization."""
        # Mock out DSPy to avoid actual LM initialization
        config = RLMConfig(
            root_model="test/root",
            worker_text_model="test/worker-text",
            worker_vision_model="test/worker-vision",
        )

        assert config.root_model == "test/root"
        assert config.worker_text_model == "test/worker-text"
        assert config.worker_vision_model == "test/worker-vision"

    def test_init_with_defaults(self):
        """Test config with default values."""
        config = RLMConfig(
            root_model="test/root",
            worker_text_model="test/worker",
            worker_vision_model="test/vision",
        )

        assert config.chunk_size == 2000
        assert config.summary_level == "standard"
        assert config.parallel_first_pass is True
        assert config.parallel_retry is False
        assert config.max_parallel_workers == 5
        assert config.max_turns == 20
        assert config.code_execution_timeout == 30
        assert config.max_retries == 3

    def test_init_custom_values(self):
        """Test config with custom values."""
        config = RLMConfig(
            root_model="test/root",
            worker_text_model="test/worker",
            worker_vision_model="test/vision",
            chunk_size=3000,
            summary_level="verbose",
            parallel_first_pass=False,
            parallel_retry=True,
            max_parallel_workers=10,
            max_turns=50,
            max_retries=5,
        )

        assert config.chunk_size == 3000
        assert config.summary_level == "verbose"
        assert config.parallel_first_pass is False
        assert config.parallel_retry is True
        assert config.max_parallel_workers == 10
        assert config.max_turns == 50
        assert config.max_retries == 5

    def test_summary_level_validation(self):
        """Test that summary_level accepts valid values."""
        # This test just verifies the type annotation works
        # Runtime validation would need to be added
        config = RLMConfig(
            root_model="test/root",
            worker_text_model="test/worker",
            worker_vision_model="test/vision",
            summary_level="minimal",
        )
        assert config.summary_level == "minimal"
