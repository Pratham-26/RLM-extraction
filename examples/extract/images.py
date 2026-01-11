"""Image Document Extraction Example.

This example demonstrates extracting structured data from document images
(e.g., scanned invoices, receipts, forms) using the RLM schema extraction
system with vision-capable models.
"""

import os
from pathlib import Path
from typing import List

from PIL import Image

from rlm_extractor import RLMExtractor, RLMConfig


def receipt_schema() -> dict:
    """JSON Schema for receipt/invoice extraction from images."""
    return {
        "type": "object",
        "properties": {
            "merchant": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "phone": {"type": "string"}
                }
            },
            "transaction": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "receipt_number": {"type": "string"}
                }
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"}
                    }
                }
            },
            "totals": {
                "type": "object",
                "properties": {
                    "subtotal": {"type": "number"},
                    "tax": {"type": "number"},
                    "total": {"type": "number"},
                    "payment_method": {"type": "string"}
                }
            }
        }
    }


def load_images_from_directory(image_dir: str) -> List[Image.Image]:
    """Load all images from a directory.

    Args:
        image_dir: Path to directory containing images

    Returns:
        List of PIL Image objects sorted by filename
    """
    path = Path(image_dir)
    if not path.exists():
        raise ValueError(f"Directory not found: {image_dir}")

    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    images = []

    for file_path in sorted(path.iterdir()):
        if file_path.suffix.lower() in image_extensions:
            try:
                img = Image.open(file_path)
                images.append(img)
                print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")

    if not images:
        raise ValueError(f"No images found in {image_dir}")

    return images


def main():
    """Run the image extraction example."""
    print("=" * 60)
    print("Image Document Extraction Example")
    print("=" * 60)

    # Configure RLM for vision processing
    config = RLMConfig(
        root_model="openai/gpt-4o",           # Text orchestrator
        worker_text_model="openai/gpt-4o-mini",
        worker_vision_model="openai/gpt-4o",   # Vision-capable worker
        max_parallel_workers=3,  # Lower for vision (more expensive)
        summary_level="standard",
    )

    # Initialize extractor
    print("\nInitializing RLM Extractor with vision support...")
    extractor = RLMExtractor(config)

    # Get the schema
    schema = receipt_schema()

    # Load images from directory
    image_dir = input("\nEnter path to directory containing receipt images: ").strip()

    if not image_dir or not os.path.exists(image_dir):
        print("\nUsing demo mode - would load images from:", image_dir)
        print("\nTo run this example:")
        print("1. Create a directory with document images (JPEG/PNG)")
        print("2. Run: python examples/extract/images.py")
        print("3. Enter the directory path when prompted")
        return

    print(f"\nLoading images from: {image_dir}")
    images = load_images_from_directory(image_dir)
    print(f"Loaded {len(images)} images")

    # Run extraction
    print("\nExtracting data from receipt images...")

    result = extractor.extract(
        json_schema=schema,
        document=images,
        task="Extract all receipt information including merchant details, line items, and totals."
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
    print(f"  Images processed: {len(result.chunk_gists)}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Complete: {result.is_complete()}")

    if result.chunk_gists:
        print("\nImage Summaries:")
        for gist in result.chunk_gists:
            print(f"  Image {gist['idx']}: {gist['gist']}")

    if result.failures:
        print("\nFailures:")
        for failure in result.failures:
            print(f"  Image {failure['chunk_idx']}: {failure['error']}")

    print("\n" + "=" * 60)


def create_demo_images():
    """Create simple demo receipt images for testing.

    This creates placeholder images with text for testing purposes.
    """
    from PIL import ImageDraw, ImageFont

    demo_dir = Path("demo_receipts")
    demo_dir.mkdir(exist_ok=True)

    # Create a simple receipt image
    width, height = 400, 600
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw receipt text (simplified)
    y = 20
    draw.text((200, y), "ACME STORE", anchor="mm", fill="black")
    y += 30
    draw.text((200, y), "123 Main Street", anchor="mm", fill="black")
    y += 20
    draw.text((200, y), "Tel: 555-1234", anchor="mm", fill="black")

    y += 40
    draw.text((20, y), "-" * 50, fill="black")
    y += 30

    draw.text((20, y), "Date: 2024-01-15  Time: 14:30", fill="black")
    y += 25
    draw.text((20, y), "Receipt #12345", fill="black")

    y += 30
    draw.text((20, y), "Items:", fill="black")
    y += 25

    items = [
        ("Coffee", 2, 3.50),
        ("Sandwich", 1, 8.99),
        ("Cookie", 3, 2.50),
    ]

    for name, qty, price in items:
        draw.text((20, y), f"{name}", fill="black")
        draw.text((300, y), f"{qty} x ${price:.2f}", fill="black")
        draw.text((350, y), f"${qty * price:.2f}", fill="black")
        y += 25

    y += 10
    draw.text((20, y), "-" * 40, fill="black")
    y += 25

    subtotal = sum(qty * price for _, qty, price in items)
    tax = subtotal * 0.08
    total = subtotal + tax

    draw.text((250, y), f"Subtotal: ${subtotal:.2f}", fill="black")
    y += 20
    draw.text((250, y), f"Tax: ${tax:.2f}", fill="black")
    y += 20
    draw.text((250, y), f"TOTAL: ${total:.2f}", fill="black")

    img.save(demo_dir / "receipt_1.png")
    print(f"Created demo image: {demo_dir / 'receipt_1.png'}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    # Check if user wants to create demo images
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-demo":
        create_demo_images()
        print("\nDemo images created in 'demo_receipts/' directory")
        print("Run again and enter 'demo_receipts' as the image directory")
    else:
        main()

    print("\nTip: Run with --create-demo to generate sample receipt images")
