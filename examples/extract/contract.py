"""Contract Extraction Example.

This example demonstrates extracting structured data from a legal contract
using the RLM schema extraction system.
"""

import os
from rlm import RLMExtractor, RLMConfig


# Sample contract document (in practice, this would be much larger)
SAMPLE_CONTRACT = """
SOFTWARE LICENSE AGREEMENT
==========================

This Software License Agreement ("Agreement") is entered into as of January 15, 2024
("Effective Date"), by and between:

LICENSOR:
    Name: Acme Software Technologies, Inc.
    Address: 100 Innovation Drive, Palo Alto, California 94301
    ("Licensor")

LICENSEE:
    Name: Global Enterprises Ltd.
    Address: 5 Broadgate, London, EC2M 7QS, United Kingdom
    ("Licensee")

1. GRANT OF LICENSE
-------------------

1.1 License Grant. Subject to the terms and conditions of this Agreement, Licensor
hereby grants to Licensee a perpetual, non-exclusive, non-transferable license to
use the Software specifically identified in Exhibit A attached hereto ("Software").

1.2 Permitted Uses. Licensee may use the Software solely for its internal business
operations and may not sublicense, distribute, or modify the Software.

2. LICENSE FEES
---------------

2.1 Initial License Fee. Licensee shall pay to Licensor a one-time, non-refundable
license fee of One Hundred Fifty Thousand Dollars ($150,000.00) upon execution of
this Agreement.

2.2 Annual Maintenance. Licensee shall pay an annual maintenance fee of Fifteen
Percent (15%) of the Initial License Fee, due on each anniversary of the Effective
Date.

3. TERM AND TERMINATION
------------------------

3.1 Term. This Agreement shall commence on the Effective Date and shall continue
in perpetuity unless terminated earlier as provided herein.

3.2 Termination for Cause. Either party may terminate this Agreement upon thirty
(30) days written notice if the other party materially breaches any provision of
this Agreement.

3.3 Effect of Termination. Upon termination, Licensee shall cease all use of the
Software and return or destroy all copies.

4. WARRANTIES
-------------

4.1 Licensor's Warranties. Licensor warrants that: (a) it owns all rights to the
Software; (b) the Software does not infringe any third-party intellectual
property rights; and (c) the Software will perform substantially as described
in the documentation.

4.2 Disclaimer. EXCEPT AS EXPRESSLY SET FORTH HEREIN, THE SOFTWARE IS PROVIDED
"AS IS" WITHOUT WARRANTIES OF ANY KIND.

5. LIMITATION OF LIABILITY
---------------------------

5.1 Liability Cap. Licensor's total liability under this Agreement shall not
exceed the amount of license fees actually paid by Licensee in the twelve (12)
months preceding the claim.

5.2 Consequential Damages. NEITHER PARTY SHALL BE LIABLE FOR ANY INDIRECT,
INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES.

6. CONFIDENTIALITY
------------------

6.1 Definition. "Confidential Information" means any non-public information
disclosed by either party.

6.2 Obligations. Each party shall maintain the confidentiality of the other's
Confidential Information for a period of five (5) years from disclosure.

7. MISCELLANEOUS
---------------

7.1 Governing Law. This Agreement shall be governed by the laws of the State
of California, USA.

7.2 Dispute Resolution. Any disputes shall be resolved through binding
arbitration in San Francisco, California.

7.3 Entire Agreement. This Agreement constitutes the entire understanding
between the parties.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the
Effective Date.

LICENSOR:                     LICENSEE:
Acme Software Technologies    Global Enterprises Ltd.
By: ___________________       By: ___________________
Name: John Smith              Name: Sarah Johnson
Title: CEO                    Title: Director
Date: January 15, 2024        Date: January 15, 2024

Exhibit A
=========

Software: "AcmeCloud Platform v3.0"
Version: 3.0.5
Description: Cloud infrastructure management software
"""


def contract_schema() -> dict:
    """JSON Schema for contract extraction."""
    return {
        "type": "object",
        "properties": {
            "agreement_type": {
                "type": "string",
                "description": "Type of agreement (e.g., Software License Agreement)"
            },
            "effective_date": {
                "type": "string",
                "description": "Date when agreement takes effect"
            },
            "parties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "description": "e.g., Licensor, Licensee"},
                        "name": {"type": "string"},
                        "address": {"type": "string"}
                    }
                }
            },
            "key_terms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term_name": {"type": "string"},
                        "description": {"type": "string"},
                        "section_reference": {"type": "string"}
                    }
                }
            },
            "financial_terms": {
                "type": "object",
                "properties": {
                    "initial_fee": {"type": "number"},
                    "initial_fee_currency": {"type": "string"},
                    "annual_maintenance_rate": {"type": "number"},
                    "payment_schedule": {"type": "string"}
                }
            },
            "term_details": {
                "type": "object",
                "properties": {
                    "duration": {"type": "string"},
                    "termination_notice_days": {"type": "number"},
                    "auto_renewal": {"type": "boolean"}
                }
            },
            "governing_law": {
                "type": "string",
                "description": "Governing law jurisdiction"
            },
            "signatories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "party": {"type": "string"},
                        "name": {"type": "string"},
                        "title": {"type": "string"}
                    }
                }
            }
        }
    }


def main():
    """Run the contract extraction example."""
    print("=" * 60)
    print("Contract Extraction Example")
    print("=" * 60)

    # Configure RLM (using Anthropic models for this example)
    config = RLMConfig(
        root_model="anthropic/claude-sonnet-4",
        worker_text_model="anthropic/claude-haiku-4",
        worker_vision_model="anthropic/claude-sonnet-4",
        chunk_size=2500,  # Larger chunks for contract
        summary_level="standard",
    )

    # Initialize extractor
    print("\nInitializing RLM Extractor...")
    extractor = RLMExtractor(config)

    # Get the schema
    schema = contract_schema()

    # Run extraction
    print("\nExtracting data from contract...")
    print(f"Document size: {len(SAMPLE_CONTRACT)} characters")

    result = extractor.extract(
        json_schema=schema,
        document=SAMPLE_CONTRACT,
        task="Extract all key terms, parties, financial details, and other important information from this contract."
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
        for gist in result.chunk_gists[:5]:
            print(f"  Chunk {gist['idx']}: {gist['gist'][:80]}...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    main()
