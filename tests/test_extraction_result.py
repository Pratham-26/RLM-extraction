"""Tests for ExtractionResult."""

import pytest

from rlm_extractor.extract.extractor import ExtractionResult


class TestExtractionResult:
    """Test extraction result dataclass."""

    def test_init(self):
        """Test result initialization."""
        result = ExtractionResult(
            data={"name": "Test"},
            chunk_gists=[{"idx": 0, "gist": "Test"}],
            failures=[],
            turns=1,
            token_usage={"total": 1000},
        )

        assert result.data == {"name": "Test"}
        assert len(result.chunk_gists) == 1
        assert len(result.failures) == 0
        assert result.turns == 1
        assert result.token_usage["total"] == 1000

    def test_is_complete_true(self):
        """Test is_complete when no failures."""
        result = ExtractionResult(
            data={},
            chunk_gists=[{"idx": 0}],
            failures=[],
        )

        assert result.is_complete() is True

    def test_is_complete_false(self):
        """Test is_complete when there are failures."""
        result = ExtractionResult(
            data={},
            chunk_gists=[{"idx": 0}],
            failures=[{"chunk_idx": 1, "error": "Test error"}],
        )

        assert result.is_complete() is False

    def test_get_failure_rate_zero(self):
        """Test failure rate with no failures."""
        result = ExtractionResult(
            data={},
            chunk_gists=[{"idx": 0}, {"idx": 1}],
            failures=[],
        )

        assert result.get_failure_rate() == 0.0

    def test_get_failure_rate_partial(self):
        """Test failure rate with some failures."""
        result = ExtractionResult(
            data={},
            chunk_gists=[{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}],
            failures=[{"chunk_idx": 1}, {"chunk_idx": 2}],
        )

        assert result.get_failure_rate() == 0.5

    def test_get_failure_rate_all(self):
        """Test failure rate with all failures."""
        result = ExtractionResult(
            data={},
            chunk_gists=[{"idx": 0}, {"idx": 1}],
            failures=[{"chunk_idx": 0}, {"chunk_idx": 1}],
        )

        assert result.get_failure_rate() == 1.0

    def test_get_failure_rate_no_chunks(self):
        """Test failure rate with no chunks."""
        result = ExtractionResult(
            data={},
            chunk_gists=[],
            failures=[],
        )

        assert result.get_failure_rate() == 0.0
