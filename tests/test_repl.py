"""Tests for REPLState."""


from rlm.repl import REPLState


class TestREPLState:
    """Test REPL state management."""

    def test_init(self):
        """Test REPL state initialization."""
        state = REPLState()

        assert state.INPUT == ""
        assert state.total_chunks == 0
        assert state.completed_chunks == set()
        assert state.failed_chunks == {}
        assert state.chunk_retry_counts == {}
        assert state.chunk_summaries == []
        assert state.results_so_far == {}

    def test_get_pending_chunks_empty(self):
        """Test getting pending chunks when none exist."""
        state = REPLState()
        state.total_chunks = 0

        pending = state.get_pending_chunks()
        assert pending == []

    def test_get_pending_chunks_some(self):
        """Test getting pending chunks."""
        state = REPLState()
        state.total_chunks = 5
        state.completed_chunks = {0, 2, 4}

        pending = state.get_pending_chunks()
        assert pending == [1, 3]

    def test_get_completion_rate(self):
        """Test completion rate calculation."""
        state = REPLState()
        state.total_chunks = 10
        state.completed_chunks = {0, 1, 2, 3, 4}

        rate = state.get_completion_rate()
        assert rate == 0.5

    def test_get_completion_rate_zero_chunks(self):
        """Test completion rate with zero chunks."""
        state = REPLState()
        state.total_chunks = 0

        rate = state.get_completion_rate()
        assert rate == 0.0

    def test_update_chunk_result(self):
        """Test updating chunk result."""
        state = REPLState()

        state.update_chunk_result(
            idx=0,
            gist="Test chunk",
            extracted={"name": "Test"},
            confidence="high",
            fields_found=["name"],
        )

        assert 0 in state.completed_chunks
        assert len(state.chunk_summaries) == 1
        assert state.chunk_summaries[0]["idx"] == 0
        assert state.chunk_summaries[0]["gist"] == "Test chunk"
        assert state.results_so_far == {"name": "Test"}

    def test_update_multiple_chunks(self):
        """Test updating multiple chunks."""
        state = REPLState()

        state.update_chunk_result(0, "Chunk 0", {"a": 1}, "high", ["a"])
        state.update_chunk_result(1, "Chunk 1", {"b": 2}, "medium", ["b"])

        assert state.completed_chunks == {0, 1}
        assert len(state.chunk_summaries) == 2
        assert state.results_so_far == {"a": 1, "b": 2}

    def test_merge_extracted_dict(self):
        """Test merging extracted dictionaries."""
        state = REPLState()

        state._merge_extracted({"name": "John"})
        assert state.results_so_far == {"name": "John"}

        state._merge_extracted({"age": 30})
        assert state.results_so_far == {"name": "John", "age": 30}

    def test_mark_failed(self):
        """Test marking chunk as failed."""
        state = REPLState()

        state.mark_failed(0, "Test error")
        state.mark_failed(1, "Another error")

        assert state.failed_chunks == {0: "Test error", 1: "Another error"}

    def test_get_state_summary(self):
        """Test getting state summary."""
        state = REPLState()
        state.total_chunks = 10
        state.completed_chunks = {0, 1, 2, 3, 4, 5, 6, 7}
        state.failed_chunks = {8: "Error"}

        summary = state.get_state_summary()

        assert "Completed: 8/10" in summary
        # Pending = total - completed = 10 - 8 = 2 (chunk 8 failed + chunk 9 untouched)
        assert "Pending: 2" in summary
        assert "Failed: 1" in summary

    def test_get_results_preview_empty(self):
        """Test results preview when empty."""
        state = REPLState()

        preview = state.get_results_preview()
        assert "No results yet" in preview

    def test_get_results_preview_with_data(self):
        """Test results preview with data."""
        state = REPLState()
        state.results_so_far = {"name": "John", "age": 30}

        preview = state.get_results_preview()
        assert "John" in preview
        assert "30" in preview

    def test_get_results_preview_truncation(self):
        """Test results preview truncation."""
        state = REPLState()
        # Create large data
        state.results_so_far = {"field_" + str(i): i for i in range(100)}

        preview = state.get_results_preview(max_chars=200)
        assert len(preview) < 300  # Should be truncated
        assert "truncated" in preview.lower()

    def test_get_chunk_summaries_preview(self):
        """Test chunk summaries preview."""
        state = REPLState()
        state.update_chunk_result(0, "First chunk", {"a": 1}, "high", ["a"])
        state.update_chunk_result(1, "Second chunk", {"b": 2}, "medium", ["b"])

        preview = state.get_chunk_summaries_preview()

        assert "Chunk 0" in preview
        assert "Chunk 1" in preview
        assert "First chunk" in preview
        assert "Second chunk" in preview

    def test_get_chunk_summaries_preview_limit(self):
        """Test chunk summaries preview with limit."""
        state = REPLState()
        for i in range(20):
            state.update_chunk_result(i, f"Chunk {i}", {f"f{i}": i}, "high", [])

        preview = state.get_chunk_summaries_preview(max_count=5)

        # Should show first 5 and mention more
        assert "Chunk 0" in preview
        assert "Chunk 4" in preview
        assert "more" in preview.lower()

    def test_reset_for_task_text(self):
        """Test resetting for a new text task."""
        state = REPLState()
        state.completed_chunks = {0, 1, 2}
        state.failed_chunks = {3: "error"}
        state.results_so_far = {"old": "data"}
        state.chunk_retry_counts[0] = 2

        state.reset_for_task("new document text", "schema")

        assert state.INPUT == "new document text"
        assert state.yaml_schema == "schema"
        assert state.completed_chunks == set()
        assert state.failed_chunks == {}
        assert state.results_so_far == {}
        assert state.chunk_retry_counts == {}

    def test_reset_for_task_images(self):
        """Test resetting for a new image task."""
        state = REPLState()
        images = ["img1", "img2", "img3"]

        state.reset_for_task(images, "schema")

        assert state.INPUT == images
        assert state.total_chunks == 3
        assert state.completed_chunks == set()

    def test_chunk_retry_counts(self):
        """Test that retry counts are tracked per chunk."""
        state = REPLState()

        # First processing
        state.update_chunk_result(0, "First try", {"a": 1}, "low", ["a"])
        assert state.chunk_retry_counts[0] == 1

        # Re-extraction
        state.update_chunk_result(0, "Second try", {"a": 1}, "high", ["a"])
        assert state.chunk_retry_counts[0] == 2

        # Different chunk
        state.update_chunk_result(1, "Chunk 1", {"b": 2}, "medium", ["b"])
        assert state.chunk_retry_counts[1] == 1

        # Multiple chunks
        state.update_chunk_result(2, "Chunk 2", {"c": 3}, "low", ["c"])
        assert state.chunk_retry_counts[2] == 1

    def test_chunk_retry_counts_in_summary(self):
        """Test that retry counts appear in chunk summaries."""
        state = REPLState()

        state.update_chunk_result(0, "Chunk 0", {"a": 1}, "low", ["a"])
        state.update_chunk_result(0, "Retry 1", {"a": 1}, "medium", ["a"])

        summary = state.get_chunk_summaries_preview()
        assert "retry 2" in summary.lower()

    def test_get_retry_summary(self):
        """Test retry summary formatting."""
        state = REPLState()

        # No retries
        assert "No re-extractions yet" in state.get_retry_summary()

        # Single chunk processed once
        state.update_chunk_result(0, "Chunk 0", {"a": 1}, "high", ["a"])
        assert "No chunks re-extracted yet" in state.get_retry_summary()

        # Chunk re-extracted
        state.update_chunk_result(0, "Retry 1", {"a": 1}, "medium", ["a"])
        summary = state.get_retry_summary()
        assert "Chunk 0: 2 total attempts" in summary

        # Multiple chunks re-extracted
        state.update_chunk_result(1, "Chunk 1", {"b": 2}, "low", ["b"])
        state.update_chunk_result(1, "Retry 1", {"b": 2}, "high", ["b"])
        summary = state.get_retry_summary()
        assert "Chunk 0: 2 total attempts" in summary
        assert "Chunk 1: 2 total attempts" in summary

    def test_set_total_chunks(self):
        """Test setting total chunks."""
        state = REPLState()
        state.set_total_chunks(42)

        assert state.total_chunks == 42
