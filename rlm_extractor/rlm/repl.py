"""RLM REPL Environment - The core execution environment for Recursive Language Models.

This module provides the REPL environment that:
1. Holds the INPUT variable (document/context)
2. Executes Python code safely
3. Provides the LM() function for recursive calls
4. Tracks state across turns
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from rlm_extractor.rlm.safety import SafeREPL, SafetyError
from rlm_extractor.rlm.output_parser import OutputParser, ParsedOutput


@dataclass
class REPLTurn:
    """Record of a single REPL turn."""

    # Turn number (1-indexed)
    turn_number: int

    # The LM's response (what it said)
    lm_response: str

    # Parsed components
    parsed: ParsedOutput

    # Code that was executed
    code_executed: str = ""

    # Output from code execution
    execution_output: str = ""

    # Whether this was a final turn
    is_final: bool = False


class RLMREPL:
    """The RLM REPL Environment.

    This is the core environment that enables the Recursive Language Model
    paradigm from the paper. It provides:

    1. INPUT variable - the document/context to operate on
    2. Code execution - safely run Python code
    3. LM() function - recursive self-calls
    4. State persistence - variables live across turns
    5. Turn tracking - history of execution
    """

    def __init__(
        self,
        input_context: str = "",
        lm_function: Callable[[str], str] | None = None,
        timeout: int = 30,
        allow_imports: set[str] | None = None,
    ):
        """Initialize the RLM REPL.

        Args:
            input_context: The initial INPUT value (document/context)
            lm_function: The LM() function for recursive calls
            timeout: Code execution timeout in seconds
            allow_imports: Set of allowed module names for imports
        """
        # Create the safe REPL
        self.repl = SafeREPL(
            allow_imports=allow_imports,
            timeout=timeout,
        )

        # Set INPUT
        if input_context:
            self.repl.set_input(input_context)

        # Set LM function if provided
        if lm_function:
            self.repl.set_lm_function(lm_function)

        # Turn history
        self.turns: list[REPLTurn] = []
        self.current_turn: int = 0

        # Final result
        self.final_answer: str | None = None
        self.final_variable: str | None = None

    def set_input(self, input_context: str) -> None:
        """Set or update the INPUT variable.

        Args:
            input_context: The document/context string
        """
        self.repl.set_input(input_context)

    def set_lm_function(self, lm_function: Callable[[str], str]) -> None:
        """Set or update the LM() function for recursive calls.

        Args:
            lm_function: Callable that takes prompt and returns answer
        """
        self.repl.set_lm_function(lm_function)

    def execute_turn(self, lm_response: str) -> REPLTurn:
        """Execute one turn of the RLM loop.

        Args:
            lm_response: The LM's response (code, FINAL, or thought)

        Returns:
            REPLTurn record of what happened

        Raises:
            SafetyError: If code violates safety rules
        """
        self.current_turn += 1

        # Parse the response
        parsed = OutputParser.parse(lm_response)

        # Create turn record
        turn = REPLTurn(
            turn_number=self.current_turn,
            lm_response=lm_response,
            parsed=parsed,
            is_final=parsed.is_final,
        )

        # Handle final answer
        if parsed.is_final:
            if parsed.final_answer is not None:
                self.final_answer = parsed.final_answer
            elif parsed.final_var is not None:
                var_value = self.repl.get_variable(parsed.final_var)
                self.final_answer = str(var_value) if var_value is not None else f"Variable {parsed.final_var} not found"
                self.final_variable = parsed.final_var
            self.turns.append(turn)
            return turn

        # Execute code blocks
        for code in parsed.code_blocks:
            turn.code_executed = code
            try:
                output = self.repl.execute(code)
                turn.execution_output = output
            except SafetyError as e:
                turn.execution_output = f"Safety Error: {e}"
                raise
            except Exception as e:
                turn.execution_output = f"Execution Error: {e}"

        self.turns.append(turn)
        return turn

    def get_conversation_history(self, include_code_output: bool = True) -> str:
        """Get formatted conversation history for the LM.

        Args:
            include_code_output: Whether to include code execution output

        Returns:
            Formatted string of previous turns
        """
        if not self.turns:
            return ""

        lines = []
        for turn in self.turns:
            lines.append(f"--- Turn {turn.turn_number} ---")

            if turn.parsed.thought:
                lines.append(f"Thought: {turn.parsed.thought[:500]}")

            if turn.code_executed:
                lines.append(f"Code executed:")
                lines.append(f"```python\n{turn.code_executed}\n```")

            if include_code_output and turn.execution_output:
                lines.append(f"Output: {turn.execution_output[:500]}")

            if turn.is_final:
                lines.append(f"FINAL: {turn.parsed.raw_match}")

        return "\n".join(lines)

    def is_complete(self) -> bool:
        """Check if the RLM has reached a final answer.

        Returns:
            True if FINAL() or FINAL_VAR() was called
        """
        return self.final_answer is not None

    def get_result(self) -> str:
        """Get the final answer.

        Returns:
            The final answer string

        Raises:
            RuntimeError: If not complete (no final answer yet)
        """
        if not self.is_complete():
            raise RuntimeError("RLM not complete - no final answer yet")
        return self.final_answer

    def get_variable(self, name: str) -> Any:
        """Get a variable from the REPL.

        Args:
            name: Variable name

        Returns:
            Variable value or None
        """
        return self.repl.get_variable(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the REPL.

        Args:
            name: Variable name
            value: Variable value
        """
        self.repl.set_variable(name, value)

    def get_user_variables(self) -> dict:
        """Get all user-defined variables.

        Returns:
            Dict of variable name to value
        """
        return self.repl.get_user_variables()

    def reset(self, keep_input: bool = True) -> None:
        """Reset the REPL state.

        Args:
            keep_input: Whether to keep the INPUT variable
        """
        input_val = self.repl.globals.get("INPUT") if keep_input else None
        lm_func = self.repl.globals.get("LM")

        self.repl.reset()

        if input_val is not None:
            self.repl.globals["INPUT"] = input_val
        if lm_func is not None:
            self.repl.globals["LM"] = lm_func

        self.turns.clear()
        self.current_turn = 0
        self.final_answer = None
        self.final_variable = None


class RLMEnvironment:
    """Complete RLM execution environment.

    Combines the REPL with the LM interface to provide the full
    Recursive Language Model experience as described in the paper.
    """

    def __init__(
        self,
        input_context: str = "",
        timeout: int = 30,
        max_turns: int = 20,
        allow_imports: set[str] | None = None,
    ):
        """Initialize the RLM Environment.

        Args:
            input_context: The document/context to operate on
            timeout: Code execution timeout per turn
            max_turns: Maximum turns before giving up
            allow_imports: Allowed modules for import
        """
        self.input_context = input_context
        self.timeout = timeout
        self.max_turns = max_turns

        # Will be created when executor is set
        self.repl: RLMREPL | None = None

        # The executor function - takes (prompt, conversation_history) -> response
        self._executor: Callable[[str, str], str] | None = None

    def set_executor(self, executor: Callable[[str, str], str]) -> None:
        """Set the LM executor function.

        Args:
            executor: Function that takes (task, conversation_history) and returns LM response
        """
        self._executor = executor

    def create_repl(self, lm_function: Callable[[str], str]) -> RLMREPL:
        """Create the REPL for a new RLM session.

        Args:
            lm_function: The LM() function for recursive calls

        Returns:
            Initialized RLMREPL
        """
        self.repl = RLMREPL(
            input_context=self.input_context,
            lm_function=lm_function,
            timeout=self.timeout,
        )
        return self.repl

    def run(self, task: str) -> tuple[str, list[REPLTurn]]:
        """Run the RLM loop until completion or max turns.

        Args:
            task: The task description

        Returns:
            Tuple of (final_answer, turn_history)

        Raises:
            RuntimeError: If no executor set or max turns exceeded
        """
        if self._executor is None:
            raise RuntimeError("No executor set. Call set_executor() first.")

        # Create LM function that uses our executor
        def lm_func(prompt: str) -> str:
            # This creates a new RLM sub-call
            # For now, just return the result of running the sub-task
            # In full implementation, this would create a new RLMEnvironment
            sub_repl = RLMREPL(input_context=prompt, timeout=self.timeout)
            # Execute the sub-task...
            # This is a simplified version
            return self._executor(prompt, "")

        # Create REPL
        self.repl = self.create_repl(lm_func)

        # Main loop
        conversation = ""

        for turn in range(1, self.max_turns + 1):
            # Get LM response
            response = self._executor(task, conversation)

            # Execute the turn
            repl_turn = self.repl.execute_turn(response)

            # Update conversation history
            conversation = self.repl.get_conversation_history()

            # Check if done
            if self.repl.is_complete():
                return self.repl.get_result(), self.repl.turns

        # Max turns exceeded
        raise RuntimeError(f"Maximum turns ({self.max_turns}) exceeded without final answer")
