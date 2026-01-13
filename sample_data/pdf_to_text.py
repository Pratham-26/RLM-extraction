#!/usr/bin/env python3
"""Convert PDF to text with page separators."""

from pathlib import Path
from pypdf import PdfReader


def pdf_to_text(pdf_path: Path | str, output_path: Path | str | None = None) -> str:
    """
    Extract text from PDF with page separators.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the text file

    Returns:
        The extracted text as a string
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(pdf_path)

    pages_text = []
    separator = "=" * 80

    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        page_header = f"\n\n{separator}\nPage {i} of {len(reader.pages)}\n{separator}\n\n"
        pages_text.append(page_header + text)

    full_text = "\n".join(pages_text) + "\n"

    if output_path:
        output_path = Path(output_path)
        output_path.write_text(full_text, encoding="utf-8")
        print(f"Saved text to: {output_path}")

    return full_text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_to_text.py <pdf_path> [output_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    text = pdf_to_text(pdf_path, output_path)
    print(f"\nExtracted {len(text)} characters from PDF")
