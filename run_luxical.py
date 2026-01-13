"""Product Ideas Extraction from Research Paper.

This script extracts product implementation ideas from research papers using:
- Minimax as root model (orchestrator)
- Gemini Flash as worker model (extractor)

Output:
- paper_summary: Brief context about the paper
- product_implementation_ideas: Actionable product/feature ideas grounded in paper findings

Input: sample_data/luxical.txt (text extracted from PDF)
Output: product_ideas_<timestamp>.json
"""

import json
import os
from rlm_extractor import RLMExtractor, RLMConfig


def load_schema(schema_path: str = "sample_data/paper_schema.json") -> dict:
    """Load JSON Schema from file.

    Args:
        schema_path: Path to JSON schema file

    Returns:
        Schema dictionary
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """Extract product ideas from research paper."""
    config = RLMConfig(
        root_model="openrouter/minimax/minimax-m2.1",
        worker_text_model="openrouter/google/gemini-2.5-flash-lite",
        chunk_size=4000,
        max_parallel_workers=2,
    )

    extractor = RLMExtractor(config)
    schema = load_schema()
    document_path = "sample_data/luxical.txt"

    result = extractor.extract(
        json_schema=schema,
        document=document_path,
        task="""Extract product implementation ideas from this research paper.

Your goal is to derive actionable product/feature ideas that are grounded in the paper's validated findings.

For the paper_summary section: Briefly capture what problem the paper solves, what approach they propose, and what they actually validated.

For product_implementation_ideas: Think like a product manager. What could be built based on these techniques?
- Each idea MUST be grounded in specific findings, mechanisms, or validated approaches from the paper
- Cite explicit evidence from the paper that supports why this idea would work
- Include realistic feasibility assessment based on the paper's technical depth and resource requirements
- Focus on ideas that leverage the paper's novel contributions as an unfair advantage

Quality over quantity - 2-3 well-grounded ideas are better than 5 vague ones.""",
    )

    print("\n=== EXTRACTION COMPLETE ===")
    print(f"Total turns: {result.turns}")
    print(f"Chunks processed: {len(result.chunk_gists)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Complete: {result.is_complete()}")
    if result.failures:
        print(f"\nFailures: {result.failures}")

    # Pretty print product ideas
    if result.data and "product_implementation_ideas" in result.data:
        ideas = result.data["product_implementation_ideas"]
        print(f"\n=== EXTRACTED {len(ideas)} PRODUCT IDEAS ===")
        for i, idea in enumerate(ideas, 1):
            print(f"\n{i}. {idea.get('idea_name', 'Unnamed')}")
            print(f"   Target: {idea.get('target_users', 'N/A')}")
            print(f"   Description: {idea.get('description', 'N/A')[:100]}...")

    # Save output to file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"product_ideas_{timestamp}.json"
    output_data = {
        "metadata": {
            "turns": result.turns,
            "chunks_processed": len(result.chunk_gists),
            "failures": result.failures,
            "complete": result.is_complete(),
        },
        "paper_summary": result.data.get("paper_summary", {}) if result.data else {},
        "product_implementation_ideas": result.data.get("product_implementation_ideas", []) if result.data else [],
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        exit(1)

    main()
