"""Simple RLM Extraction Demo.

A minimal example showing how to use the RLM schema extraction system.
"""

import os
from rlm import RLMExtractor, RLMConfig
from rlm.config import openai_config, anthropic_config, cost_optimized_config


# Sample document
DOCUMENT = """
Customer Information Request
==============================

Name: Jane Smith
Email: jane.smith@example.com
Phone: (555) 987-6543

Address: 789 Oak Avenue, Boston, MA 02101

Account Details:
    Account Number: ACC-2024-7890
    Account Type: Premium Business
    Status: Active
    Since: January 2020

Request Details:
    Subject: Request for account statement
    Priority: High
    Reference: TICKET-45678

Message:
    I would like to request a complete account statement for the past
    12 months, including all transactions and fees. This is needed
    for our annual audit.

    Please send the statement to the email address above.
    Thank you for your assistance.

    Sincerely,
    Jane Smith
    Chief Financial Officer
    Global Tech Solutions Inc.
"""


# Simple extraction schema
SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "address": {"type": "string"}
            }
        },
        "account": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string"},
                "account_type": {"type": "string"},
                "status": {"type": "string"}
            }
        },
        "request": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "priority": {"type": "string"},
                "reference": {"type": "string"}
            }
        }
    }
}


def main():
    """Run the demo."""
    print("=" * 50)
    print("RLM Schema Extraction Demo")
    print("=" * 50)

    # Use preset config (or customize)
    config = openai_config()
    config.chunk_size = 1500
    config.max_parallel_workers = 3

    print(f"\nConfig:")
    print(f"  Root LM: {config.root_model}")
    print(f"  Worker LM: {config.worker_text_model}")
    print(f"  Chunk size: {config.chunk_size}")

    # Initialize extractor
    print("\nInitializing RLM Extractor...")
    extractor = RLMExtractor(config)

    # Extract
    print("\nExtracting...")
    result = extractor.extract(
        json_schema=SCHEMA,
        document=DOCUMENT,
    )

    # Show results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    import json
    print("\nExtracted Data:")
    print(json.dumps(result.data, indent=2))

    print(f"\nStats: {result.turns} turns, {len(result.chunk_gists)} chunks")
    print(f"Complete: {result.is_complete()}")

    if result.failures:
        print(f"\nFailures: {len(result.failures)}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable to run this demo.")
        exit(1)

    main()
