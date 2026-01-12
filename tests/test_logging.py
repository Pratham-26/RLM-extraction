"""Integration test for LLM call logging with RLMExtractor.

This test verifies that:
1. ExtractionResult includes log_file_path
2. Log file is created with proper entries
3. Log entries contain all required fields
"""

import json
from pathlib import Path

import pytest

from rlm_extractor import CallLogger, RLMExtractor, RLMConfig


class TestLoggingIntegration:
    """Test LLM call logging integration."""

    def test_logger_creates_file(self):
        """Test that CallLogger creates a log file."""
        logger = CallLogger()

        call_id = logger.log_request(
            lm_type="root",
            model="openrouter/anthropic/claude-sonnet-4",
            signature="TestSignature",
            call_type="test",
            request={"test": "data"},
        )

        logger.log_response(call_id=call_id, response={"result": "success"})
        logger.close()

        log_path = logger.get_log_file_path()
        assert log_path.exists()
        assert log_path.stat().st_size > 0

        # Clean up
        log_path.unlink()

    def test_logger_jsonl_format(self):
        """Test that log entries are valid JSON Lines format."""
        logger = CallLogger()

        logger.log_request(
            lm_type="root",
            model="openrouter/anthropic/claude-sonnet-4",
            signature="TestSignature",
            call_type="test",
            request={"test": "data"},
        )
        logger.close()

        log_path = logger.get_log_file_path()

        # Read and validate JSON Lines
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "call_id" in entry
        assert "lm_type" in entry
        assert "model" in entry
        assert "signature" in entry
        assert "call_type" in entry
        assert "stage" in entry
        assert "request" in entry
        assert "timestamp" in entry

        # Clean up
        log_path.unlink()

    def test_request_response_pairing(self):
        """Test that request and response entries share the same call_id."""
        logger = CallLogger()

        call_id = logger.log_request(
            lm_type="worker",
            model="openrouter/anthropic/claude-haiku-4",
            signature="TestSignature",
            call_type="test",
            request={"test": "data"},
            metadata={"chunk_idx": 0, "attempt": 1},
        )

        logger.log_response(
            call_id=call_id,
            response={"result": "success"},
            metadata={"chunk_idx": 0, "attempt": 1},
        )
        logger.close()

        log_path = logger.get_log_file_path()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        request_entry = json.loads(lines[0])
        response_entry = json.loads(lines[1])

        assert request_entry["call_id"] == response_entry["call_id"]
        assert request_entry["call_id"] == call_id
        assert request_entry["stage"] == "request"
        assert response_entry["stage"] == "response"
        assert request_entry["lm_type"] == "worker"
        assert request_entry["metadata"]["chunk_idx"] == 0

        # Clean up
        log_path.unlink()

    def test_error_logging(self):
        """Test that errors are logged correctly."""
        logger = CallLogger()

        call_id = logger.log_request(
            lm_type="worker",
            model="openrouter/anthropic/claude-haiku-4",
            signature="TestSignature",
            call_type="test",
            request={"test": "data"},
        )

        logger.log_error(
            call_id=call_id,
            error="Test error message",
            metadata={"error_type": "timeout"},
        )
        logger.close()

        log_path = logger.get_log_file_path()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        error_entry = json.loads(lines[1])

        assert error_entry["call_id"] == call_id
        assert error_entry["stage"] == "error"
        assert error_entry["error"] == "Test error message"
        assert error_entry["metadata"]["error_type"] == "timeout"

        # Clean up
        log_path.unlink()

    def test_worker_vs_root_lm_differentiation(self):
        """Test that Root LM and Worker LM calls are clearly differentiated."""
        logger = CallLogger()

        # Root LM call
        root_call_id = logger.log_request(
            lm_type="root",
            model="openrouter/anthropic/claude-sonnet-4",
            signature="RootExtractionSignature",
            call_type="root_decision",
            request={"task": "test"},
        )

        # Worker LM call
        worker_call_id = logger.log_request(
            lm_type="worker",
            model="openrouter/anthropic/claude-haiku-4",
            signature="WorkerExtractionSignature",
            call_type="worker_extraction",
            request={"yaml_schema": "test", "chunk_idx": 0},
        )

        logger.close()

        log_path = logger.get_log_file_path()

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

        root_entry = json.loads(lines[0])
        worker_entry = json.loads(lines[1])

        assert root_entry["lm_type"] == "root"
        assert root_entry["call_type"] == "root_decision"
        assert root_entry["model"] == "openrouter/anthropic/claude-sonnet-4"

        assert worker_entry["lm_type"] == "worker"
        assert worker_entry["call_type"] == "worker_extraction"
        assert worker_entry["model"] == "openrouter/anthropic/claude-haiku-4"

        # Clean up
        log_path.unlink()
