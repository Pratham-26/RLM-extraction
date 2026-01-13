"""LLM Call Logger for RLM.

Logs all LLM requests and responses to JSON Lines files for fine-tuning.
Captures both Root LM (orchestrator) and Worker LM (extraction) calls with
clear identification of the model type and call context.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class CallLogger:
    """Logger for LLM calls and responses.

    Logs each LLM interaction as a JSON line including:
    - Timestamp and unique call ID
    - LM type (root/worker) and model name
    - DSPy signature and call type
    - Full request and response payloads
    - Metadata (chunk_idx, turn, attempt, etc.)
    """

    def __init__(self, log_dir: str | Path | None = None):
        """Initialize the call logger.

        Args:
            log_dir: Directory to store log files. Defaults to current working directory.
        """
        if log_dir is None:
            log_dir = Path.cwd()
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = log_dir / f"rlm_extraction_{timestamp}.jsonl"
        self._file_handle = None

    def _open(self) -> None:
        """Open the log file if not already open."""
        if self._file_handle is None:
            self._file_handle = open(self.log_file_path, "a", encoding="utf-8")
            return

    def _ensure_open(self) -> bool:
        """Ensure the log file is open.

        Returns:
            True if file is open, False otherwise
        """
        if self._file_handle is None:
            self._file_handle = open(self.log_file_path, "a", encoding="utf-8")
        return True

    def _close(self) -> None:
        """Close the log file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def _write_log_entry(self, entry: dict) -> None:
        """Write a log entry to the file.

        Args:
            entry: Dictionary containing log data
        """
        if self._file_handle is None:
            self._file_handle = open(self.log_file_path, "a", encoding="utf-8")

        def json_default(obj):
            if hasattr(obj, "__str__"):
                return str(obj)
            return "<non-serializable>"

        self._file_handle.write(json.dumps(entry, ensure_ascii=False, default=json_default) + "\n")
        self._file_handle.flush()

    def _prepare_call_id(self) -> str:
        """Generate a unique call ID.

        Returns:
            UUID string for this call
        """
        return str(uuid.uuid4())

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            ISO 8601 formatted timestamp string
        """
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def log_request(
        self,
        lm_type: str,
        model: str,
        signature: str,
        call_type: str,
        request: dict,
        metadata: dict | None = None,
    ) -> str:
        """Log an LLM request.

        Args:
            lm_type: Type of LM - "root" or "worker"
            model: Model name (e.g., "openrouter/anthropic/claude-sonnet-4")
            signature: DSPy signature class name
            call_type: Specific operation type (e.g., "context_condensation")
            request: Full request payload
            metadata: Optional metadata (chunk_idx, turn, attempt, etc.)

        Returns:
            Call ID for pairing with response
        """
        call_id = self._prepare_call_id()

        entry = {
            "timestamp": self._get_timestamp(),
            "call_id": call_id,
            "lm_type": lm_type,
            "model": model,
            "signature": signature,
            "call_type": call_type,
            "stage": "request",
            "request": request,
            "metadata": metadata or {},
        }

        self._write_log_entry(entry)
        return call_id

    def log_response(
        self,
        call_id: str,
        response: dict,
        metadata: dict | None = None,
    ) -> None:
        """Log an LLM response.

        Args:
            call_id: Call ID from corresponding request
            response: Full response payload
            metadata: Optional additional metadata
        """
        entry = {
            "timestamp": self._get_timestamp(),
            "call_id": call_id,
            "stage": "response",
            "response": response,
            "metadata": metadata or {},
        }

        self._write_log_entry(entry)

    def log_error(
        self,
        call_id: str,
        error: str,
        metadata: dict | None = None,
    ) -> None:
        """Log an LLM error.

        Args:
            call_id: Call ID from corresponding request
            error: Error message
            metadata: Optional additional metadata
        """
        entry = {
            "timestamp": self._get_timestamp(),
            "call_id": call_id,
            "stage": "error",
            "error": error,
            "metadata": metadata or {},
        }

        self._write_log_entry(entry)

    def get_log_file_path(self) -> Path:
        """Get the path to the log file.

        Returns:
            Path object pointing to the log file
        """
        return self.log_file_path

    def close(self) -> None:
        """Close the logger and flush all data."""
        self._close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
