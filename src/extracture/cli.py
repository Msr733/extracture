"""CLI for extracture — quick extraction from the command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extracture",
        description="Extract structured data from documents with high accuracy.",
    )
    parser.add_argument("file", help="Path to the document file")
    parser.add_argument(
        "--schema",
        required=True,
        help="Python path to Pydantic model (e.g., 'myapp.models:Invoice')",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai:gpt-4o"],
        help="LLM providers (e.g., 'openai:gpt-4o' 'anthropic:claude-sonnet-4-6-20250514')",
    )
    parser.add_argument(
        "--ocr", default="pymupdf", help="OCR engine (pymupdf, surya, paddleocr, tesseract, doctr)"
    )
    parser.add_argument(
        "--consensus",
        default="confidence_weighted",
        choices=["confidence_weighted", "majority", "best_provider"],
        help="Consensus strategy",
    )
    parser.add_argument("--grounding", action="store_true", help="Enable grounding verification")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--format", choices=["json", "compact"], default="json", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Import schema
    schema_class = _import_schema(args.schema)
    if not schema_class:
        print(f"Error: Could not import schema '{args.schema}'", file=sys.stderr)
        sys.exit(1)

    # Import extracture
    from extracture import Extractor

    extractor = Extractor(
        schema=schema_class,
        providers=args.providers,
        ocr_engine=args.ocr,
        consensus=args.consensus,
        enable_grounding=args.grounding,
    )

    # Extract
    result = extractor.extract(args.file)

    # Output
    if args.format == "compact":
        output = json.dumps(
            {name: f.value for name, f in result.fields.items()},
            indent=2,
            default=str,
        )
    else:
        output = result.to_json()

    if args.output:
        Path(args.output).write_text(output)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Print summary to stderr
    print(
        f"\nConfidence: {result.overall_confidence:.3f} | "
        f"Status: {result.status.value} | "
        f"Review: {result.review_decision.value} | "
        f"Fields: {len(result.fields)}",
        file=sys.stderr,
    )


def _import_schema(path: str) -> type[Any] | None:
    """Import a Pydantic model from a dotted path like 'module:ClassName'."""
    try:
        if ":" in path:
            module_path, class_name = path.rsplit(":", 1)
        elif "." in path:
            parts = path.rsplit(".", 1)
            module_path, class_name = parts[0], parts[1]
        else:
            print(f"Schema must be 'module:ClassName' format, got '{path}'", file=sys.stderr)
            return None

        import importlib

        module = importlib.import_module(module_path)
        cls: type[Any] = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError) as e:
        print(f"Error importing schema: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    main()
