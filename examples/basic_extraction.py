"""Basic extraction example — extract structured data from an invoice PDF."""

from decimal import Decimal
from pydantic import BaseModel, Field

from extracture import Extractor, FieldAnchor


# Step 1: Define your schema
class Invoice(BaseModel):
    vendor_name: str = Field(description="Company name of the vendor/seller")
    invoice_number: str = Field(description="Invoice number or reference ID")
    invoice_date: str = Field(description="Invoice date in YYYY-MM-DD format")
    due_date: str | None = Field(default=None, description="Payment due date")
    subtotal: Decimal = Field(description="Subtotal before tax")
    tax: Decimal = Field(description="Tax amount")
    total: Decimal = Field(description="Total amount due")
    currency: str = Field(default="USD", description="Currency code (USD, EUR, etc.)")


# Step 2: Create extractor with your preferred providers
extractor = Extractor(
    schema=Invoice,

    # Use multiple providers for consensus (highest accuracy)
    providers=[
        "openai:gpt-4o",
        "anthropic:claude-sonnet-4-6-20250514",
    ],

    # OCR engine for scanned documents
    ocr_engine="pymupdf",  # or "surya", "paddleocr", "tesseract"

    # Consensus strategy
    consensus="confidence_weighted",

    # Cross-field validation rules
    validation_rules=[
        # Total must equal subtotal + tax
        (
            "total_check",
            ["subtotal", "tax", "total"],
            lambda f: (
                None
                if not hasattr(f, "total") or f.total is None
                else (
                    None
                    if abs(float(f.total) - float(f.subtotal or 0) - float(f.tax or 0)) < 0.01
                    else f"Total {f.total} != {f.subtotal} + {f.tax}"
                )
            ),
            "warning",
        ),
    ],

    # Enable grounding verification (verify values exist in source)
    enable_grounding=True,
)


# Step 3: Extract!
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python basic_extraction.py <invoice.pdf>")
        sys.exit(1)

    result = extractor.extract(sys.argv[1])

    # Access typed data
    print(f"Vendor: {result.data.vendor_name if result.data else 'N/A'}")
    print(f"Invoice #: {result.data.invoice_number if result.data else 'N/A'}")
    print(f"Total: {result.data.total if result.data else 'N/A'}")
    print()

    # Access per-field details
    print("Field Details:")
    for name, field in result.fields.items():
        grounded = "✓" if field.is_grounded else "✗" if field.is_grounded is False else "?"
        print(
            f"  {name}: {field.value} "
            f"(conf={field.effective_confidence:.2f}, grounded={grounded})"
        )

    print(f"\nOverall Confidence: {result.overall_confidence:.3f}")
    print(f"Status: {result.status.value}")
    print(f"Review Decision: {result.review_decision.value}")

    if result.validation_errors:
        print(f"\nValidation Errors:")
        for err in result.validation_errors:
            print(f"  [{err.severity}] {err.message}")

    # Audit trail
    print(f"\nAudit:")
    print(f"  Providers: {result.audit.providers_used}")
    print(f"  Duration: {result.audit.total_duration_ms:.0f}ms")
    print(f"  Cost: ${result.audit.cost_estimate_usd:.4f}")

    # Correct a field (HITL)
    # result.correct("vendor_name", "Corrected Name Inc.")
    # extractor.learn_from_corrections(result)
