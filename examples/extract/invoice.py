"""Invoice Extraction Example.

This example demonstrates extracting structured data from an invoice document
using the RLM schema extraction system.
"""

import os
from rlm_extractor import RLMExtractor, RLMConfig


# Sample invoice document (in practice, this would be a much larger file)
SAMPLE_INVOICE = """
INVOICE
=======
Invoice Number: INV-2024-12345
Date: 2024-01-15
Due Date: 2024-02-15

From:
    Acme Corporation Services
    123 Business Avenue, Suite 500
    New York, NY 10001
    Phone: (555) 123-4567

To:
    Tech Solutions Inc.
    456 Technology Lane
    San Francisco, CA 94105

Line Items:
-----------

Item #1: Cloud Infrastructure Services
    Description: Monthly cloud hosting and infrastructure services
    Quantity: 1
    Unit Price: $2,500.00
    Amount: $2,500.00

Item #2: Technical Support Package
    Description: 24/7 premium technical support (monthly)
    Quantity: 1
    Unit Price: $750.00
    Amount: $750.00

Item #3: Additional Storage (10 TB)
    Description: Extra cloud storage capacity
    Quantity: 10
    Unit Price: $50.00
    Amount: $500.00

Item #4: API Calls Overage
    Description: Additional API calls beyond standard package
    Quantity: 1,000,000
    Unit Price: $0.001
    Amount: $1,000.00

Subtotal: $4,750.00
Tax (8.875%): $421.56
Total: $5,171.56

Payment Terms:
    Net 30 days
    Bank Transfer to: Acme Corp - Chase Bank - Account: ****1234
    Payment Reference: INV-2024-12345

Notes:
    Thank you for your business! For questions regarding this invoice,
    please contact accounts@acmecorp.com
"""


def invoice_schema() -> dict:
    """JSON Schema for invoice extraction."""
    return {
        "type": "object",
        "properties": {
            "invoice_number": {
                "type": "string",
                "description": "Invoice number (e.g., INV-2024-12345)"
            },
            "invoice_date": {
                "type": "string",
                "description": "Invoice date in ISO format (YYYY-MM-DD)"
            },
            "due_date": {
                "type": "string",
                "description": "Due date in ISO format"
            },
            "vendor": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "phone": {"type": "string"},
                    "email": {"type": "string"}
                }
            },
            "customer": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"}
                }
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_number": {"type": "string"},
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "amount": {"type": "number"}
                    }
                }
            },
            "subtotal": {"type": "number"},
            "tax": {"type": "number"},
            "total": {"type": "number"},
            "payment_terms": {"type": "string"},
        },
        "required": ["invoice_number", "invoice_date", "total", "line_items"]
    }


def main():
    """Run the invoice extraction example."""
    print("=" * 60)
    print("Invoice Extraction Example")
    print("=" * 60)

    # Configure RLM (using preset config)
    config = RLMConfig(
        root_model="openai/gpt-4o",
        worker_text_model="openai/gpt-4o-mini",
        chunk_size=2000,
        max_parallel_workers=5,
    )

    # Initialize extractor
    print("\nInitializing RLM Extractor...")
    extractor = RLMExtractor(config)

    # Get the schema
    schema = invoice_schema()

    # Run extraction
    print("\nExtracting data from invoice...")
    print(f"Document size: {len(SAMPLE_INVOICE)} characters")

    result = extractor.extract(
        json_schema=schema,
        document=SAMPLE_INVOICE,
        task="Extract all invoice information including line items and totals."
    )

    # Display results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)

    print("\nExtracted Data:")
    print("-" * 40)
    import json
    print(json.dumps(result.data, indent=2))

    print("\n" + "=" * 60)
    print("Extraction Statistics:")
    print(f"  Turns: {result.turns}")
    print(f"  Chunks processed: {len(result.chunk_gists)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Complete: {result.is_complete()}")

    if result.chunk_gists:
        print("\nChunk Summaries:")
        for gist in result.chunk_gists[:5]:  # Show first 5
            print(f"  Chunk {gist['idx']}: {gist['gist'][:80]}...")

    if result.failures:
        print("\nFailures:")
        for failure in result.failures:
            print(f"  Chunk {failure['chunk_idx']}: {failure['error']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    main()
