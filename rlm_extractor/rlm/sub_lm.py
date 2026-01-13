"""Sub-LM handler for recursive RLM calls.

Manages recursive invocations of the RLM system with:
- Depth tracking to prevent infinite recursion
- Result caching
- Parallel execution support
- State isolation between calls
"""

import concurrent.futures
import threading
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from rlm_extractor.rlm.output_parser import OutputParser, ParsedOutput, RLMResult


@dataclass
class SubLMCall:
    """Record of a sub-LM call."""

    # Unique identifier for this call
    call_id: str

    # The prompt given to the sub-LM
    prompt: str

    # Depth at which this call was made (0 = root)
    depth: int

    # The result returned
    result: str | None = None

    # Number of turns this sub-call took
    turns: int = 0

    # Whether this call succeeded
    success: bool = True

    # Error message if failed
    error: str | None = None

    # Parent call ID (None for root calls)
    parent_id: str | None = None

    # Timestamps
    start_time: float | None = None
    end_time: float | None = None

    # Child calls made from this call
    child_ids: list[str] = field(default_factory=list)


class SubLMHandler:
    """Manages recursive LM calls for RLM.

    The LM() function creates new RLM instances that:
    1. Run independently with their own REPL state
    2. Share the same base configuration
    3. Track recursion depth
    4. Can execute code on their INPUT (the prompt)
    """

    def __init__(
        self,
        max_depth: int = 2,
        enable_cache: bool = True,
        max_parallel: int = 5,
    ):
        """Initialize sub-LM handler.

        Args:
            max_depth: Maximum recursion depth (1 = root + 1 level of sub-calls)
            enable_cache: Whether to cache sub-LM results
            max_parallel: Maximum parallel sub-calls
        """
        self.max_depth = max_depth
        self.enable_cache = enable_cache
        self.max_parallel = max_parallel

        # Thread safety
        self._lock = threading.RLock()

        # State tracking
        self.current_depth: int = 0
        self.call_history: list[SubLMCall] = []
        self.call_tree: dict[str, SubLMCall] = {}

        # Result cache: {prompt_hash: result}
        self._cache: dict[str, str] = {}

        # The executor function - set externally
        self._executor: Callable[[str, int], RLMResult] | None = None

    def set_executor(self, executor: Callable[[str, int], RLMResult]) -> None:
        """Set the function that executes RLM calls.

        Args:
            executor: Function that takes (prompt, depth) and returns RLMResult
        """
        self._executor = executor

    def create_lm_function(self) -> Callable[[str], str]:
        """Create the LM() function for injection into REPL.

        Returns:
            A callable that can be used as LM(prompt) in RLM code
        """

        def lm_func(prompt: str) -> str:
            """Recursive LM call.

            Args:
                prompt: The prompt to send to the sub-LM (becomes its INPUT)

            Returns:
                The final answer from the sub-LM

            Raises:
                RuntimeError: If recursion depth exceeded or no executor set
            """
            return self._call_lm(prompt)

        return lm_func

    def _call_lm(self, prompt: str) -> str:
        """Internal method to handle a single LM() call.

        Args:
            prompt: The prompt for the sub-LM

        Returns:
            The final answer from the sub-LM

        Raises:
            RuntimeError: If max depth exceeded or no executor
        """
        if self._executor is None:
            raise RuntimeError("SubLMHandler: No executor set. Call set_executor() first.")

        # Check depth
        with self._lock:
            if self.current_depth >= self.max_depth:
                raise RuntimeError(
                    f"Maximum recursion depth ({self.max_depth}) exceeded. "
                    f"Current depth: {self.current_depth}"
                )

            # Increment depth for this call
            self.current_depth += 1
            depth_at_call = self.current_depth

        # Check cache
        if self.enable_cache:
            import hashlib
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            if prompt_hash in self._cache:
                with self._lock:
                    self.current_depth -= 1
                return self._cache[prompt_hash]

        # Create call record
        call_id = str(uuid4())[:8]
        call = SubLMCall(
            call_id=call_id,
            prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            depth=depth_at_call,
            start_time=None,  # Would need time.time() import
        )

        # Execute the sub-call
        try:
            result = self._executor(prompt, depth_at_call)
            call.result = result.answer
            call.turns = result.turns
            call.success = result.success
            call.error = result.error

            # Cache the result
            if self.enable_cache and result.success:
                self._cache[prompt_hash] = result.answer

            return result.answer

        except Exception as e:
            call.success = False
            call.error = str(e)
            raise

        finally:
            # Decrement depth
            with self._lock:
                self.current_depth -= 1
                self.call_history.append(call)
                self.call_tree[call_id] = call

    def call_parallel(self, prompts: list[str]) -> list[str]:
        """Make multiple LM() calls in parallel.

        Args:
            prompts: List of prompts for parallel sub-LMs

        Returns:
            List of answers in the same order as prompts

        Raises:
            RuntimeError: If max depth would be exceeded
        """
        if self._executor is None:
            raise RuntimeError("SubLMHandler: No executor set. Call set_executor() first.")

        # Check depth
        with self._lock:
            if self.current_depth >= self.max_depth:
                raise RuntimeError(f"Maximum recursion depth ({self.max_depth}) exceeded")

            # Increment depth for these calls
            self.current_depth += 1
            depth_at_call = self.current_depth

        results = [None] * len(prompts)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            future_to_idx = {
                executor.submit(self._execute_single, prompt, depth_at_call, idx): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {e}"

        # Decrement depth
        with self._lock:
            self.current_depth -= 1

        return results

    def _execute_single(self, prompt: str, depth: int, idx: int) -> str:
        """Execute a single parallel sub-call.

        Args:
            prompt: The prompt for the sub-LM
            depth: Current recursion depth
            idx: Index in the parallel batch

        Returns:
            The answer from the sub-LM
        """
        # Check cache
        if self.enable_cache:
            import hashlib
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            if prompt_hash in self._cache:
                return self._cache[prompt_hash]

        result = self._executor(prompt, depth)

        # Cache result
        if self.enable_cache and result.success:
            self._cache[prompt_hash] = result.answer

        return result.answer

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about sub-LM calls.

        Returns:
            Dict with call statistics
        """
        with self._lock:
            total_calls = len(self.call_history)
            successful = sum(1 for c in self.call_history if c.success)
            failed = total_calls - successful
            max_depth_reached = max((c.depth for c in self.call_history), default=0)
            total_turns = sum(c.turns for c in self.call_history)

            return {
                "total_calls": total_calls,
                "successful": successful,
                "failed": failed,
                "max_depth_reached": max_depth_reached,
                "total_turns": total_turns,
                "current_depth": self.current_depth,
                "cache_size": len(self._cache),
            }

    def reset(self) -> None:
        """Reset handler state (clear cache, history, depth)."""
        with self._lock:
            self.current_depth = 0
            self.call_history.clear()
            self.call_tree.clear()
            self._cache.clear()
