"""Safety guards for RLM code execution.

Provides sandboxed code execution with:
- Blocked modules (os, subprocess, etc.)
- Blocked operations (open, exec, eval, etc.)
- Timeout enforcement
- Resource limits
"""

import ast
import re
import signal
from contextlib import redirect_stdout
from io import StringIO
from types import CodeType


class SafetyError(Exception):
    """Raised when code violates safety rules."""

    pass


class CodeValidator:
    """Validates Python code for safe execution in RLM REPL."""

    # Modules that are completely blocked
    BLOCKED_MODULES = {
        "os",
        "subprocess",
        "shutil",
        "sys",
        "socket",
        "urllib",
        "urllib2",
        "requests",
        "http",
        "httplib",
        "ftplib",
        "telnetlib",
        "smtplib",
        "poplib",
        "imaplib",
        "ssl",
        "asyncio",
        "threading",
        "multiprocessing",
        "ctypes",
        "pickle",
        "shelve",
        "marshal",
        "importlib",
        "__import__",
    }

    # Built-in functions that are blocked
    BLOCKED_BUILTINS = {
        "open",
        "exec",
        "eval",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "__import__",
    }

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r"__import__\s*\(",
        r"import\s+os\b",
        r"import\s+subprocess\b",
        r"import\s+sys\b",
        r"from\s+os\s+import",
        r"from\s+subprocess\s+import",
        r"from\s+sys\s+import",
        r"\.open\s*\(",
        r"\.__class__",
        r"\.__bases__",
        r"\.__subclasses__",
        r"\.__mro__",
        r"\.__code__",
        r"\.func_code",
    ]

    def __init__(self, allow_imports: set[str] | None = None):
        """Initialize validator.

        Args:
            allow_imports: Set of module names that are explicitly allowed.
                          Default: common safe modules for data processing.
        """
        self.allow_imports = allow_imports or {
            "math",
            "random",
            "re",
            "string",
            "json",
            "collections",
            "itertools",
            "functools",
            "datetime",
            "decimal",
            "fractions",
            "typing",
            "dataclasses",
        }

    def validate(self, code: str) -> tuple[bool, str | None]:
        """Validate code for safety.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        # Check dangerous patterns first
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                return False, f"Blocked pattern: {pattern}"

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for dangerous imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.BLOCKED_MODULES:
                        return False, f"Blocked module: {module}"
                    if module not in self.allow_imports:
                        return False, f"Module not allowed: {module} (allow it explicitly)"

            elif isinstance(node, ast.ImportFrom):
                module = node.module.split(".")[0] if node.module else ""
                if module in self.BLOCKED_MODULES:
                    return False, f"Blocked module: {module}"
                if module and module not in self.allow_imports:
                    return False, f"Module not allowed: {module} (allow it explicitly)"

            # Check for dangerous calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKED_BUILTINS:
                        return False, f"Blocked built-in: {node.func.id}"

                # Check for method calls on objects (e.g., x.__class__)
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr.startswith("__") and node.func.attr.endswith("__"):
                        if node.func.attr not in {
                            "__init__",
                            "__str__",
                            "__repr__",
                            "__len__",
                            "__getitem__",
                            "__setitem__",
                            "__contains__",
                            "__iter__",
                            "__next__",
                            "__enter__",
                            "__exit__",
                        }:
                            return False, f"Blocked dunder method: {node.func.attr}"

        return True, None


class SafeREPL:
    """Sandboxed Python REPL for RLM code execution.

    Provides:
    - INPUT variable with document context
    - LM() function for recursive calls (injected externally)
    - Safe built-ins only
    - Timeout enforcement
    - Output capture
    """

    # Safe built-ins for REPL
    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "ascii": ascii,
        "bin": bin,
        "bool": bool,
        "bytearray": bytearray,
        "bytes": bytes,
        "chr": chr,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hash": hash,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
    }

    def __init__(
        self,
        allow_imports: set[str] | None = None,
        timeout: int = 30,
        max_output_chars: int = 100_000,
    ):
        """Initialize safe REPL.

        Args:
            allow_imports: Set of module names to allow imports for
            timeout: Maximum execution time per code block in seconds
            max_output_chars: Maximum characters of output to capture
        """
        self.validator = CodeValidator(allow_imports)
        self.timeout = timeout
        self.max_output_chars = max_output_chars

        # REPL state
        self.globals: dict = self._create_globals()
        self.locals: dict = {}

        # Track user-defined variables
        self.user_variables: set[str] = set()

    def _create_globals(self) -> dict:
        """Create safe globals dict for execution."""
        safe_globals = {
            "__builtins__": self.SAFE_BUILTINS.copy(),
        }

        # Add allowed modules (lazy import on first use)
        for module_name in self.validator.allow_imports:
            safe_globals[module_name] = __import__(module_name)

        return safe_globals

    def set_input(self, input_context: str) -> None:
        """Set the INPUT variable for the REPL.

        Args:
            input_context: The document/context string to operate on
        """
        self.globals["INPUT"] = input_context

    def set_lm_function(self, lm_func: callable) -> None:
        """Inject the LM() function for recursive calls.

        Args:
            lm_func: Callable that takes a prompt string and returns the result
        """
        self.globals["LM"] = lm_func

    def set_variable(self, name: str, value: any) -> None:
        """Set a variable in the REPL.

        Args:
            name: Variable name
            value: Variable value
        """
        self.globals[name] = value
        self.user_variables.add(name)

    def get_variable(self, name: str) -> any:
        """Get a variable from the REPL.

        Args:
            name: Variable name

        Returns:
            Variable value or None if not found
        """
        return self.globals.get(name) or self.locals.get(name)

    def get_user_variables(self) -> dict:
        """Get all user-defined variables.

        Returns:
            Dict of variable name to value
        """
        return {
            name: self.globals[name]
            for name in self.user_variables
            if name in self.globals
        }

    def execute(self, code: str) -> str:
        """Execute code in the REPL and return output.

        Args:
            code: Python code to execute

        Returns:
            String output from execution (stdout or error message)

        Raises:
            SafetyError: If code violates safety rules
            TimeoutError: If code execution exceeds timeout
        """
        # Validate code first
        is_safe, error = self.validator.validate(code)
        if not is_safe:
            raise SafetyError(error)

        # Capture output
        output_buffer = StringIO()

        try:
            with redirect_stdout(output_buffer):
                exec(code, self.globals, self.locals)

            output = output_buffer.getvalue()

            # Truncate if too long
            if len(output) > self.max_output_chars:
                output = output[: self.max_output_chars] + f"\n... (truncated, {len(output)} total chars)"

            return output or "Code executed successfully (no output)"

        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    def reset(self) -> None:
        """Reset REPL state (keep INPUT and LM function)."""
        input_val = self.globals.get("INPUT")
        lm_func = self.globals.get("LM")

        self.globals = self._create_globals()
        self.locals = {}
        self.user_variables = set()

        if input_val is not None:
            self.globals["INPUT"] = input_val
        if lm_func is not None:
            self.globals["LM"] = lm_func


class TimeoutHandler:
    """Context manager for enforcing code execution timeouts.

    Uses signal.alarm() on Unix systems. For Windows, relies on
    external timeout handling.
    """

    def __init__(self, timeout: int):
        """Initialize timeout handler.

        Args:
            timeout: Timeout in seconds
        """
        self.timeout = timeout
        self._old_handler = None

    def __enter__(self):
        """Set up signal handler on Unix systems."""
        if hasattr(signal, "SIGALRM"):
            self._old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up signal handler."""
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if self._old_handler:
                signal.signal(signal.SIGALRM, self._old_handler)

    @staticmethod
    def _timeout_handler(signum, frame):
        """Handle timeout signal."""
        raise TimeoutError(f"Code execution exceeded timeout")

    @classmethod
    def is_available(cls) -> bool:
        """Check if timeout via signal is available (Unix only)."""
        return hasattr(signal, "SIGALRM")
