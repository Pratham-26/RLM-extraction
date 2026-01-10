"""DSPy Signatures for RLM.

Defines the input/output contracts for Root LM (orchestrator) and Worker LM
(extraction) interactions.
"""

import dspy


class RootExtractionSignature(dspy.Signature):
    """You are orchestrating schema extraction from a document.

Analyze the extraction progress and decide:
1. If key fields are missing, issue a 're_extract' action targeting the relevant chunk
2. If extraction is complete, issue 'finalize' action

Prioritize re-extraction for: required fields, high-value data, conflicting information."""

    task = dspy.InputField(
        desc="The extraction task - extract according to the provided schema"
    )
    trajectory = dspy.InputField(
        desc="History of previous actions and their results"
    )
    state_summary = dspy.InputField(
        desc="Current state: completed chunks, pending, failures"
    )
    chunk_summaries = dspy.InputField(
        desc="Summaries of processed chunks (gist, confidence, fields found)"
    )
    results_preview = dspy.InputField(
        desc="Preview of extraction results so far"
    )

    thought = dspy.OutputField(
        desc="Reasoning about what to do next - analyze what's missing and decide action"
    )
    action = dspy.OutputField(
        desc="Next action: 're_extract' a specific chunk or 'finalize' to complete"
    )
    target_chunk = dspy.OutputField(
        desc="For 're_extract': which chunk index to examine (integer or None)"
    )
    targeted_prompt = dspy.OutputField(
        desc="For 're_extract': specific instruction for the worker (e.g., 'Extract invoice_number from this chunk')"
    )


class WorkerExtractionSignature(dspy.Signature):
    """Extract information from a document chunk according to the schema.

Return ONLY fields actually present in this chunk. If a field is not found, list it in missing_fields.
Provide a brief gist summarizing what this chunk contains."""

    yaml_schema = dspy.InputField(
        desc="Full YAML schema defining what to extract"
    )
    chunk_content = dspy.InputField(
        desc="Document chunk text or base64 encoded image"
    )
    chunk_idx = dspy.InputField(
        desc="Index of this chunk for reference"
    )
    targeted_prompt = dspy.InputField(
        desc="Optional specific instruction - empty for first pass",
        default=""
    )

    gist = dspy.OutputField(
        desc="2-3 sentence summary of this chunk's content and relevant information"
    )
    extracted = dspy.OutputField(
        desc="Extracted fields in YAML format (only fields found in this chunk)"
    )
    confidence = dspy.OutputField(
        desc="Confidence level: high, medium, or low"
    )
    missing_fields = dspy.OutputField(
        desc="List of schema field names NOT found in this chunk"
    )
