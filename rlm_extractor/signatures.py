"""DSPy Signatures for RLM.

Defines the input/output contracts for Root LM (orchestrator) and Worker LM
(extraction) interactions.
"""

import dspy


class ContextCondensationSignature(dspy.Signature):
    """Condense user-provided context into extraction guidance.

    The Root LM uses this to transform verbose user context into concise,
    actionable guidance for Worker LMs. Focus on critical instructions and
    domain-specific information that workers need.

    Example input: "Please extract all the invoice details from this PDF. The invoice number should be captured exactly as it appears, including the prefix 'INV-'. The total amount should include any tax calculations. We're particularly interested in the line items for services provided."

    Example output: "Extract invoice_number (exact format with 'INV-' prefix), total_amount (including tax), and line_items for services.
    """

    user_context = dspy.InputField(desc="User-provided context and instructions for extraction")
    yaml_schema = dspy.InputField(desc="YAML schema defining extraction targets")

    condensed_guidance = dspy.OutputField(
        desc="Concise extraction guidance (2-4 sentences) "
        "highlighting critical instructions, domain-specific terms, "
        "and formatting requirements for workers"
    )


class RootExtractionSignature(dspy.Signature):
    """You are orchestrating schema extraction from a document.

    Analyze the extraction progress and decide:
    1. If key fields are missing, issue a 're_extract' action targeting the relevant chunk
    2. If extraction is complete, issue 'finalize' action

    Prioritize re-extraction for: required fields, high-value data, conflicting information.

    Finalize when all required fields are found with medium or higher confidence,
    or after max_retries re-extraction attempts without improvement.

    Re-extraction priority guide:
    - Required fields missing: always re-extract
    - Required fields with low confidence: re-extract if retry count < max_retries
    - High-value optional fields missing: re-extract if mentioned in > 1 chunk (infer value from field names/schema)
    - Conflicting values: re-extract chunks with conflicting info
    - Medium confidence required fields: acceptable, skip re-extraction

    Example: Re-extraction scenario
    - Schema requires: invoice_number, date, total_amount
    - State: invoice_number found (high confidence), date found (low confidence), total_amount missing
    - Decision: re_extract chunk 3 with prompt "Extract total_amount and verify date"

    Example: Finalize scenario
    - Schema requires: invoice_number, date, total_amount
    - State: all required fields found with high confidence
    - Decision: finalize"""

    task = dspy.InputField(desc="The extraction task - extract according to the provided schema")
    trajectory = dspy.InputField(desc="History of previous actions and their results")
    state_summary = dspy.InputField(desc="Current state: completed chunks, pending, failures")
    chunk_summaries = dspy.InputField(
        desc="Summaries of processed chunks (gist, confidence, fields found)"
    )
    results_preview = dspy.InputField(desc="Preview of extraction results so far")
    field_completion = dspy.InputField(
        desc="Field completion status: which required fields are found/missing. "
        "Use this to prioritize re-extraction for missing required fields.",
        default="",
    )
    retry_summary = dspy.InputField(
        desc="Summary of which chunks have been re-extracted and how many times"
    )
    max_retries = dspy.InputField(
        desc="Maximum number of re-extraction attempts allowed per chunk", default=3
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
    """Extract entity context descriptions from a document chunk.

    For each field in the schema that appears in this chunk, write a concise description
    that captures WHERE the information appears, WHAT the value is (in context), and
    any relevant details that would help extract the exact value later.

    DO NOT extract structured values - describe them in natural language.
    For list items, create separate entries with _0, _1, _2 suffixes.

    Return ONLY fields actually present in this chunk. If a field is not found, list it in missing_fields.

    Example good output:
    invoice_number: "Found at top of page as 'INV-2024-001'"
    total_amount: "Appears as '$1,234.56' in line 'Total Due' at bottom of page"
    line_items_0: "First item: 'Consulting Services - 10 hours @ $100/hr'"
    line_items_1: "Second item: 'Technical Support - 5 hours @ $75/hr'"

    Example bad output:
    invoice_number: INV-2024-001
    total_amount: 1234.56
    line_items_0: Consulting Services - 10 hours @ $100/hr"""

    yaml_schema = dspy.InputField(desc="Full YAML schema defining what to extract")
    chunk_content = dspy.InputField(
        desc="Document chunk content - text string for text documents or dspy.Image for vision processing"
    )
    chunk_idx = dspy.InputField(desc="Index of this chunk for reference")
    condensed_guidance = dspy.InputField(
        desc="Condensed user guidance for extraction - follow these instructions", default=""
    )
    targeted_prompt = dspy.InputField(
        desc="Optional specific instruction - empty for first pass", default=""
    )

    gist = dspy.OutputField(
        desc="2-3 sentence summary of this chunk's content and relevant information"
    )
    entity_contexts = dspy.OutputField(
        desc="Context descriptions for entities found in this chunk. "
        "Format: one line per entity as 'field_name: description'. "
        "Use field_name_0, field_name_1 for list items. "
        "Only include fields ACTUALLY PRESENT in this chunk."
    )
    confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")
    missing_fields = dspy.OutputField(desc="List of schema field names NOT found in this chunk")


class RootValueExtractionSignature(dspy.Signature):
    """Extract structured values from aggregated entity context descriptions.

    You receive all entity_contexts from all processed chunks. Your task is to:
    1. Parse each context description to extract the actual value
    2. Handle conflicting information across chunks (use most reliable/confident)
    3. Convert to the proper data type (string, number, boolean, etc.)
    4. Reconstruct list items from _0, _1, _2 entries
    5. Map to the JSON Schema structure

    The entity_contexts contain natural language descriptions of where values
    appear and what they contain. You must extract the actual values and structure
    them according to the schema.

    Example input contexts:
    invoice_number: "Found at top of page as 'INV-2024-001'" [chunk 0, high confidence]
    invoice_number: "Shows as 'INV-2024-001' in header section" [chunk 2, high confidence]
    total_amount: "Appears as '$1,234.56' in line 'Total Due'" [chunk 3, high confidence]
    total_amount: "Subtotal shows $1,100.00, tax $134.56" [chunk 2, medium confidence]

    Example output:
    invoice_number: "INV-2024-001"
    total_amount: 1234.56

    Note: Used $1,234.56 as total_amount because it's explicitly labeled as 'Total Due', more reliable than sum calculation from subtotal+tax.

    Example - Combine information across chunks:
    Input contexts:
    billing_address: "Found at top: '123 Main St'" [chunk 0, high confidence]
    billing_address: "Continues on page 2: 'Apt 4B, New York, NY'" [chunk 2, high confidence]

    Output:
    billing_address: "123 Main St, Apt 4B, New York, NY"

    Note: Combined information from multiple chunks (both high confidence).

    Example - Merge duplicate list items:
    Input contexts:
    line_items_0: "First item: 'Consulting - 10 hours'" [chunk 1, high confidence]
    line_items_1: "Second item: 'Support - 5 hours'" [chunk 1, high confidence]
    line_items_2: "Third item: 'Training - 8 hours'" [chunk 2, medium confidence]
    line_items_3: "Fourth item: 'Training - 3 hours'" [chunk 2, low confidence]

    Output:
    line_items:
      - description: "Consulting", hours: 10
      - description: "Support", hours: 5
      - description: "Training", hours: 11

    Note: Merged two 'Training' entries as they likely represent the same service line item."""

    yaml_schema = dspy.InputField(desc="YAML schema defining the target structure")
    entity_contexts_all = dspy.InputField(
        desc="All entity_contexts from all chunks, aggregated with chunk references"
    )
    chunk_summaries = dspy.InputField(desc="Summaries of each chunk for context (gist, confidence)")
    field_completion = dspy.InputField(
        desc="Field completion status: which required fields are found/missing", default=""
    )

    thought = dspy.OutputField(
        desc="Reasoning about value extraction - analyze conflicts, choose best values"
    )
    extracted_values = dspy.OutputField(
        desc="Extracted values in YAML format matching the schema structure. "
        "Use proper types (strings quoted, numbers unquoted, etc.)"
    )
    extraction_notes = dspy.OutputField(
        desc="Notes on extraction decisions: conflicts resolved, ambiguities handled", default=""
    )
