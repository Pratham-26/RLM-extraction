"""RLM Core - Recursive Language Model implementation.

This module provides the core RLM paradigm as described in the paper:
- Python REPL environment with INPUT variable
- LM() function for recursive self-calls
- FINAL() / FINAL_VAR() for returning answers
- Code execution with safety guards
"""

from rlm_extractor.rlm.executor import RLMExecutor, RLMExtractor, RLMConfig
from rlm_extractor.rlm.repl import RLMREPL, REPLTurn, RLMEnvironment
from rlm_extractor.rlm.sub_lm import SubLMHandler, SubLMCall
from rlm_extractor.rlm.output_parser import OutputParser, ParsedOutput, RLMResult
from rlm_extractor.rlm.safety import SafeREPL, CodeValidator, SafetyError
from rlm_extractor.rlm.signatures import (
    RLMSignature,
    RLMToolUseSignature,
    ExtractionTaskSignature,
    WorkerExtractionSignature,
    RLM_SYSTEM_PROMPT,
)

__all__ = [
    # Main executor
    "RLMExecutor",
    "RLMExtractor",
    "RLMConfig",
    # REPL
    "RLMREPL",
    "REPLTurn",
    "RLMEnvironment",
    # Sub-LM
    "SubLMHandler",
    "SubLMCall",
    # Output parsing
    "OutputParser",
    "ParsedOutput",
    "RLMResult",
    # Safety
    "SafeREPL",
    "CodeValidator",
    "SafetyError",
    # Signatures
    "RLMSignature",
    "RLMToolUseSignature",
    "ExtractionTaskSignature",
    "WorkerExtractionSignature",
    "RLM_SYSTEM_PROMPT",
]
