"""RLM Executor - The main orchestrator for Recursive Language Model execution.

This module provides the core RLM execution engine that:
1. Manages the REPL environment
2. Handles the main execution loop
3. Manages sub-LM calls with depth tracking
4. Integrates with DSPy for LM calls
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import dspy

from rlm_extractor.rlm.repl import RLMREPL, REPLTurn
from rlm_extractor.rlm.sub_lm import SubLMHandler
from rlm_extractor.rlm.output_parser import OutputParser, RLMResult
from rlm_extractor.rlm.safety import SafetyError


@dataclass
class RLMConfig:
    """Configuration for RLM execution."""

    # Maximum turns before giving up
    max_turns: int = 20

    # Maximum recursion depth for LM() calls
    max_depth: int = 2

    # Code execution timeout (seconds)
    code_timeout: int = 30

    # Maximum parallel sub-LM calls
    max_parallel: int = 5

    # Enable result caching for sub-LM calls
    enable_cache: bool = True

    # Allowed modules for import
    allow_imports: set[str] = field(default_factory=lambda: {
        "math", "random", "re", "string", "json",
        "collections", "itertools", "functools", "datetime",
        "decimal", "fractions", "typing", "dataclasses",
    })


class RLMExecutor:
    """Main RLM execution engine.

    This is the core class that implements the Recursive Language Model
    paradigm from the paper. It manages:

    1. The REPL environment with INPUT variable
    2. The main execution loop (turn-based interaction)
    3. Sub-LM recursive calls via LM() function
    4. Code execution with safety checks
    5. FINAL() / FINAL_VAR() handling
    """

    def __init__(
        self,
        lm: dspy.LM,
        config: RLMConfig | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize the RLM Executor.

        Args:
            lm: The DSPy LM instance to use
            config: RLM configuration options
            system_prompt: Optional custom system prompt
        """
        self.lm = lm
        self.config = config or RLMConfig()

        # System prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Sub-LM handler for recursion
        self.sub_lm_handler = SubLMHandler(
            max_depth=self.config.max_depth,
            enable_cache=self.config.enable_cache,
            max_parallel=self.config.max_parallel,
        )

        # Create DSPy predictor
        self._predictor = dspy.Predict(self._create_signature())

        # Current execution state
        self._current_depth: int = 0
        self._execution_id: str | None = None

    def _get_default_system_prompt(self) -> str:
        """Get the default RLM system prompt."""
        from rlm_extractor.rlm.signatures import RLM_SYSTEM_PROMPT
        return RLM_SYSTEM_PROMPT

    def _create_signature(self) -> type:
        """Create the DSPy signature for RLM."""
        # Dynamic signature creation
        class DynamicRLMSignature(dspy.Signature):
            """Dynamic RLM signature."""

            task = dspy.InputField(desc="The task to solve")
            conversation_history = dspy.InputField(
                desc="Previous conversation turns",
                default="",
            )
            response = dspy.OutputField(
                desc="Your response: code in ```python blocks, or FINAL(answer) / FINAL_VAR(variable)"
            )

        return DynamicRLMSignature

    def run(
        self,
        task: str,
        input_context: str = "",
    ) -> RLMResult:
        """Run the RLM to completion.

        Args:
            task: The task description
            input_context: The INPUT variable value (document/context)

        Returns:
            RLMResult with final answer and execution metadata
        """
        start_time = time.time()

        # Create LM function for this execution
        def lm_func(prompt: str) -> str:
            return self._sub_lm_call(prompt, self._current_depth + 1)

        # Set up sub-LM handler
        self.sub_lm_handler.set_executor(self._execute_sub_lm)
        self.sub_lm_handler.reset()

        # Create REPL
        repl = RLMREPL(
            input_context=input_context,
            lm_function=lm_func,
            timeout=self.config.code_timeout,
            allow_imports=self.config.allow_imports,
        )

        # Initial conversation history
        conversation_history = ""

        # Main loop
        all_code = []
        all_thoughts = []

        try:
            for turn in range(1, self.config.max_turns + 1):
                # Get LM response
                with dspy.context(lm=self.lm):
                    prediction = self._predictor(
                        task=task,
                        conversation_history=conversation_history,
                    )

                response = getattr(prediction, "response", "")

                # Execute the turn
                try:
                    repl_turn = repl.execute_turn(response)
                except SafetyError as e:
                    # Safety violation - return error result
                    return RLMResult(
                        answer=f"Safety Error: {e}",
                        turns=turn,
                        success=False,
                        error=str(e),
                        all_code=all_code,
                        all_thoughts=all_thoughts,
                    )

                # Track execution
                all_code.extend(repl_turn.parsed.code_blocks)
                all_thoughts.append(repl_turn.parsed.thought)

                # Update conversation history
                conversation_history = repl.get_conversation_history()

                # Check if complete
                if repl.is_complete():
                    return RLMResult(
                        answer=repl.get_result(),
                        turns=turn,
                        final_variables=repl.get_user_variables(),
                        all_code=all_code,
                        all_thoughts=all_thoughts,
                        success=True,
                    )

            # Max turns exceeded
            return RLMResult(
                answer="",
                turns=self.config.max_turns,
                success=False,
                error=f"Maximum turns ({self.config.max_turns}) exceeded without final answer",
                all_code=all_code,
                all_thoughts=all_thoughts,
            )

        except Exception as e:
            return RLMResult(
                answer="",
                turns=len(all_code),
                success=False,
                error=str(e),
                all_code=all_code,
                all_thoughts=all_thoughts,
            )

    def _execute_sub_lm(self, prompt: str, depth: int) -> RLMResult:
        """Execute a sub-LM call.

        Args:
            prompt: The prompt for the sub-LM (becomes its INPUT)
            depth: Current recursion depth

        Returns:
            RLMResult from the sub-call
        """
        # Track depth
        old_depth = self._current_depth
        self._current_depth = depth

        try:
            # Create REPL for sub-call
            repl = RLMREPL(
                input_context=prompt,
                lm_function=lambda p: self._sub_lm_call(p, depth + 1),
                timeout=self.config.code_timeout,
                allow_imports=self.config.allow_imports,
            )

            # Execute sub-call
            all_code = []
            all_thoughts = []
            conversation = ""

            for turn in range(1, self.config.max_turns + 1):
                # Get LM response
                with dspy.context(lm=self.lm):
                    prediction = self._predictor(
                        task="Solve the task given in your INPUT variable.",
                        conversation_history=conversation,
                    )

                response = getattr(prediction, "response", "")

                # Execute turn
                repl_turn = repl.execute_turn(response)
                all_code.extend(repl_turn.parsed.code_blocks)
                all_thoughts.append(repl_turn.parsed.thought)

                # Update conversation
                conversation = repl.get_conversation_history()

                # Check if complete
                if repl.is_complete():
                    return RLMResult(
                        answer=repl.get_result(),
                        turns=turn,
                        all_code=all_code,
                        all_thoughts=all_thoughts,
                        success=True,
                    )

            # Max turns exceeded
            return RLMResult(
                answer="",
                turns=self.config.max_turns,
                success=False,
                error="Max turns exceeded in sub-call",
                all_code=all_code,
                all_thoughts=all_thoughts,
            )

        finally:
            self._current_depth = old_depth

    def _sub_lm_call(self, prompt: str, depth: int) -> str:
        """Direct LM() call interface.

        Args:
            prompt: The prompt
            depth: Recursion depth

        Returns:
            The answer string
        """
        result = self._execute_sub_lm(prompt, depth)
        return result.answer if result.success else f"Error: {result.error}"


class RLMExtractor:
    """RLM-based schema extraction.

    This uses the RLM paradigm for schema-based extraction from large documents.
    Unlike traditional extraction, this uses:
    1. Code execution for smart chunking and filtering
    2. LM() recursive calls for parallel processing
    3. Flexible strategy emergence
    """

    def __init__(
        self,
        root_lm: dspy.LM,
        worker_lm: dspy.LM,
        config: RLMConfig | None = None,
    ):
        """Initialize RLM Extractor.

        Args:
            root_lm: LM for orchestration (code execution, planning)
            worker_lm: LM for chunk processing (used in LM() calls)
            config: RLM configuration
        """
        self.root_lm = root_lm
        self.worker_lm = worker_lm
        self.config = config or RLMConfig()

        # Main RLM executor
        self.executor = RLMExecutor(
            lm=self.root_lm,
            config=self.config,
        )

    def extract(
        self,
        json_schema: dict,
        document: str,
        task: str | None = None,
    ) -> dict:
        """Extract structured data from a document using RLM paradigm.

        Args:
            json_schema: JSON Schema defining what to extract
            document: The document text
            task: Optional custom task description

        Returns:
            Extracted data matching the schema
        """
        import json

        # Default task
        if task is None:
            task = f"""Extract structured data from the INPUT document according to this JSON Schema:

{json.dumps(json_schema, indent=2)}

Return the extracted data as FINAL(json_string) where json_string is the extracted data as JSON."""

        # Run RLM
        result = self.executor.run(
            task=task,
            input_context=document,
        )

        if not result.success:
            return {
                "error": result.error,
                "partial_result": result.answer,
            }

        # Try to parse the answer as JSON
        try:
            # The answer might be in FINAL(json) format
            answer = result.answer.strip()

            # Remove FINAL() wrapper if present
            if answer.startswith("FINAL("):
                answer = answer[6:-1]  # Remove "FINAL(" and ")"

            # Try parsing as JSON
            return json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            # Return raw answer if not valid JSON
            return {
                "raw_answer": result.answer,
                "final_variables": result.final_variables,
            }

    def get_stats(self) -> dict:
        """Get execution statistics.

        Returns:
            Dict with stats from sub-LM handler
        """
        return self.sub_lm_handler.get_stats()
