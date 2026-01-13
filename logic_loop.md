# RLM Logic Loop Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLM EXTRACTION LOOP                              │
└─────────────────────────────────────────────────────────────────────────────┘

1. INITIALIZE
   ├─ Chunk document (chunk_size chars)
   ├─ Convert JSON Schema → YAML (internal format)
   ├─ Set up REPL state with INPUT variable
   └─ Identify required vs optional fields from schema


2. FIRST PASS (if parallel_first_pass = TRUE)
   ├─ Process all chunks in parallel (max_parallel_workers)
   ├─ For each chunk:
   │   ├─ Worker LM receives: yaml_schema + chunk_content + chunk_idx
   │   └─ Worker LM returns: gist + entity_contexts + confidence + missing_fields
   ├─ Aggregate results
   └─ Early exit IF (success_rate >= 80%) OR (all_required_fields_found):
       → Go to step 6


3. ORCHESTRATION LOOP (repeat until max_turns)
   │
   ├─ 3.1 ASSESS STATE
   │   ├─ Check field completion: found vs missing (required only)
   │   ├─ Review chunk status: completed / pending / failed
   │   ├─ Calculate retry_count per chunk
   │   └─ Gather: field_completion + chunk_summaries + retry_summary
   │
   ├─ 3.2 DECIDE ACTION (Root LM)
   │   │
   │   ├─ IF (all_required_fields_found):
   │   │   └─→ FINALIZE (go to step 4)
   │   │
   │   ├─ IF (only_optional_fields_missing):
   │   │   └─→ FINALIZE (go to step 4)
   │   │
   │   ├─ IF (critical_required_field_missing)
   │   │       AND (specific_chunk_likely_contains_it)
   │   │       AND (chunk_retry_count < max_retries):
   │   │   └─→ RE_EXTRACT (go to step 3.3)
   │   │
   │   └─ ELSE:
   │       └─→ FINALIZE (go to step 4)
   │
   ├─ 3.3 EXECUTE RE-EXTRACT
   │   ├─ Parse target chunks: "5" OR "3,5,7" OR "3-7" OR "all"
   │   ├─ Create targeted_prompt (specific instruction from Root LM)
   │   ├─ IF (parallel_retry = TRUE) AND (multiple chunks):
   │   │   └─ Process chunks in parallel
   │   └─ ELSE:
   │       └─ Process chunks sequentially
   │   └─ Update state with new results
   │   └─ Repeat loop (go to step 3.1)
   │
   └─ Or if FINALIZE:
       → Continue to step 4


4. VALUE EXTRACTION
   ├─ Gather all entity_contexts from all chunks
   │
   ├─ APPLY CONTEXT COMPACTION (if entity_contexts > threshold)
   │   ├─ Strategy 1 (< threshold): Direct pass, no compaction
   │   ├─ Strategy 2 (>= threshold):
   │   │   ├─ Keep all high-confidence contexts
   │   │   ├─ Keep top medium-confidence contexts (up to limit)
   │   │   └─ Summarize low-confidence contexts
   │   └─ Strategy 3 (very large):
   │       └─ Create pattern-based aggregate summaries
   │
   ├─ Send to Root LM:
   │   ├─ Input: compacted_entity_contexts + chunk_summaries + yaml_schema
   │   └─ Output: extracted_values (YAML format)
   │
   └─ Parse YAML → JSON (validate against original schema)


5. HANDLE TRUNCATED RESPONSE (if needed)
   ├─ Detect unclosed brackets/braces
   ├─ Attempt to fix by closing delimiters
   └─ Re-parse if fix successful


6. RETURN RESULTS
   ├─ data: Final extracted JSON matching schema
   ├─ chunk_gists: Summary per chunk [{idx, gist, confidence, fields_found}]
   ├─ failures: Failed chunks [{chunk_idx, error, content_preview}]
   ├─ turns: Number of loop iterations
   └─ token_usage: {root, worker, total}


┌─────────────────────────────────────────────────────────────────────────────┐
│                          TERMINATION CONDITIONS                             │
└─────────────────────────────────────────────────────────────────────────────┘

SUCCESSFUL TERMINATION:
  • Root LM explicitly chooses "finalize"
  • All required fields found (any confidence)
  • First pass success rate >= 80%

FAILURE TERMINATION:
  • max_turns exceeded
  • All chunks failed
  • Fatal error (invalid schema, config)

GRACEFUL DEGRADATION:
  • Return partial results + failure list
  • No data loss


┌─────────────────────────────────────────────────────────────────────────────┐
│                            DECISION TREE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                          ┌─────────────────┐
                          │  Start Loop     │
                          └────────┬────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ All required fields found?   │
                    └──────────┬───────────────────┘
                               │
                  ┌────────────┴────────────┐
                  │ YES                     │ NO
                  ▼                         ▼
           ┌─────────────┐         ┌──────────────────────┐
           │  FINALIZE   │         │ Critical required    │
           └─────────────┘         │ field missing?       │
                                   └──────────┬───────────┘
                                              │
                                 ┌────────────┴────────────┐
                                 │ YES                     │ NO
                                 ▼                         ▼
                          ┌─────────────┐          ┌─────────────┐
                          │ Re-extract  │          │  FINALIZE   │
                          │ specific    │          └─────────────┘
                          │ chunks      │
                          └─────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTEXT COMPACTION                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Threshold: max_entity_contexts_per_field (default: 50)

IF contexts_per_field < threshold:
    → No compaction (direct pass)

ELSE IF enable_compaction = TRUE:
    → Keep all high-confidence contexts
    → Keep top medium-confidence contexts
    → Aggregate low-confidence into pattern summaries

ELSE:
    → Truncate to threshold, show count of omitted


┌─────────────────────────────────────────────────────────────────────────────┐
│                           CONFIGURATION IMPACT                              │
└─────────────────────────────────────────────────────────────────────────────┘

Parameter                    │ Impact
─────────────────────────────┼──────────────────────────────────────────────
max_turns                    │ Maximum loop iterations before forced finalize
parallel_first_pass          │ Skip Root LM on first iteration (if TRUE)
parallel_retry               │ Re-extract multiple chunks in parallel (if TRUE)
max_parallel_workers         │ Max chunks processed simultaneously
max_retries                  │ Retry attempts per chunk before marking failed
enable_context_compaction    │ Compact entity contexts when threshold exceeded
max_entity_contexts_per_field│ Threshold for context compaction
chunk_size                   │ Characters per chunk (affects chunk count)
summary_level                │ Detail level of chunk gists (minimal/standard/verbose)
```
