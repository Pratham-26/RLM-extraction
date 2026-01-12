"""Output parser for RLM responses.

Extracts and parses:
- Code blocks from markdown
- FINAL() calls with direct answers
- FINAL_VAR() calls with variable references
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedOutput:
    """Result of parsing an RLM response."""

    # Whether this is a final answer (not intermediate code)
    is_final: bool = False

    # For FINAL(answer): the direct answer string
    final_answer: str | None = None

    # For FINAL_VAR(name): the variable name to return
    final_var: str | None = None

    # Python code blocks found in the response
    code_blocks: list[str] = None

    # The full thought/reasoning text (non-code parts)
    thought: str = ""

    # Raw parsed data for debugging
    raw_match: str = None

    def __post_init__(self):
        if self.code_blocks is None:
            self.code_blocks = []


class OutputParser:
    """Parse RLM LM responses for code execution and final answers."""

    # Pattern for FINAL(answer)
    FINAL_PATTERN = re.compile(r"FINAL\s*\(\s*(.*?)\s*\)", re.DOTALL)

    # Pattern for FINAL_VAR("variable_name") or FINAL_VAR(variable_name)
    FINAL_VAR_PATTERN = re.compile(r"FINAL_VAR\s*\(\s*['\"]?(\w+)['\"]?\s*\)", re.DOTALL)

    # Pattern for Python code blocks
    CODE_BLOCK_PATTERN = re.compile(r"```python\n(.*?)```", re.DOTALL)

    # Also match ```python with optional language
    CODE_BLOCK_PATTERN_ALT = re.compile(r"```python\n?\s*(.*?)```", re.DOTALL)

    # Match any code block as fallback
    CODE_BLOCK_FALLBACK = re.compile(r"```\n?(.*?)```", re.DOTALL)

    @classmethod
    def extract_code_blocks(cls, response: str) -> list[str]:
        """Extract all Python code blocks from a response.

        Args:
            response: LLM response text

        Returns:
            List of code strings (in order of appearance)
        """
        # Try specific python blocks first
        matches = cls.CODE_BLOCK_PATTERN.findall(response)
        if matches:
            return matches

        # Try alternative pattern
        matches = cls.CODE_BLOCK_PATTERN_ALT.findall(response)
        if matches:
            return matches

        # Fallback to any code block
        matches = cls.CODE_BLOCK_FALLBACK.findall(response)
        return matches

    @classmethod
    def check_final(cls, response: str) -> tuple[bool, str | None, str | None]:
        """Check if response contains a FINAL() or FINAL_VAR() call.

        Args:
            response: LLM response text

        Returns:
            Tuple of (is_final, final_answer, final_var)
            - is_final: True if FINAL or FINAL_VAR found
            - final_answer: The answer string if FINAL() found
            - final_var: The variable name if FINAL_VAR() found
        """
        # Check FINAL_VAR first (more specific)
        var_match = cls.FINAL_VAR_PATTERN.search(response)
        if var_match:
            return True, None, var_match.group(1)

        # Check FINAL(answer)
        final_match = cls.FINAL_PATTERN.search(response)
        if final_match:
            answer = final_match.group(1).strip()
            # Clean up quotes if the answer is quoted
            if (answer.startswith('"') and answer.endswith('"')) or (
                answer.startswith("'") and answer.endswith("'")
            ):
                answer = answer[1:-1]
            return True, answer, None

        return False, None, None

    @classmethod
    def parse(cls, response: str) -> ParsedOutput:
        """Parse an RLM response into structured components.

        Args:
            response: LLM response text

        Returns:
            ParsedOutput with code, final answer/variable, and thought
        """
        result = ParsedOutput()

        # Extract code blocks
        result.code_blocks = cls.extract_code_blocks(response)

        # Check for final answer
        is_final, final_answer, final_var = cls.check_final(response)
        result.is_final = is_final
        result.final_answer = final_answer
        result.final_var = final_var

        # Extract thought (everything except code blocks)
        thought = response
        for code in result.code_blocks:
            thought = thought.replace(f"```python\n{code}```", "")
            thought = thought.replace(f"```\n{code}```", "")
        result.thought = thought.strip()

        # Store raw match for debugging
        if final_answer:
            result.raw_match = f"FINAL({final_answer})"
        elif final_var:
            result.raw_match = f"FINAL_VAR({final_var})"

        return result

    @classmethod
    def extract_first_code(cls, response: str) -> str | None:
        """Extract only the first code block from a response.

        Args:
            response: LLM response text

        Returns:
            First code block as string, or None if no code found
        """
        codes = cls.extract_code_blocks(response)
        return codes[0] if codes else None

    @classmethod
    def has_code(cls, response: str) -> bool:
        """Check if response contains any code blocks.

        Args:
            response: LLM response text

        Returns:
            True if code blocks found
        """
        return bool(cls.extract_code_blocks(response))

    @classmethod
    def is_think_only(cls, response: str) -> bool:
        """Check if response is only thinking (no code, no final).

        Args:
            response: LLM response text

        Returns:
            True if response has no code and no FINAL markers
        """
        return not cls.has_code(response) and not cls.check_final(response)[0]


@dataclass
class RLMResult:
    """Result from a complete RLM execution."""

    # The final answer (from FINAL or FINAL_VAR)
    answer: str

    # Number of turns taken
    turns: int

    # Total tokens used (if tracked)
    tokens_used: int = 0

    # All code executed across all turns
    all_code: list[str] = None

    # All thoughts/reasoning across all turns
    all_thoughts: list[str] = None

    # Final REPL variables
    final_variables: dict = None

    # Whether execution completed successfully
    success: bool = True

    # Error message if not successful
    error: str | None = None

    def __post_init__(self):
        if self.all_code is None:
            self.all_code = []
        if self.all_thoughts is None:
            self.all_thoughts = []
        if self.final_variables is None:
            self.final_variables = {}
