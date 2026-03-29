# extracture

High-accuracy, schema-driven document extraction with multi-model consensus, grounding verification, and human-in-the-loop correction.

## Install

```bash
pip install extracture
```

**Optional extras:**

```bash
pip install extracture[surya]        # Surya OCR (best open-source accuracy)
pip install extracture[paddleocr]    # PaddleOCR (lightweight, multilingual)
pip install extracture[tesseract]    # Tesseract OCR
pip install extracture[doctr]        # DocTR OCR
pip install extracture[textract]     # AWS Textract (bounding boxes)
pip install extracture[grounding]    # NLI-based hallucination detection
pip install extracture[all]          # Everything
```

## Quick Start

```python
from pydantic import BaseModel, Field
from extracture import Extractor

class Invoice(BaseModel):
    vendor: str = Field(description="Vendor name")
    invoice_number: str = Field(description="Invoice number")
    total: float = Field(description="Total amount due")

extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
)

result = extractor.extract("invoice.pdf")
print(result.data.vendor)          # "Acme Corporation"
print(result.data.total)           # 6696.0
print(result.overall_confidence)   # 0.95
```

## Supported Providers

| Provider | Format | Env Variable |
|----------|--------|--------------|
| OpenAI GPT-4o | `openai:gpt-4o` | `OPENAI_API_KEY` |
| OpenAI GPT-4.1 | `openai:gpt-4.1` | `OPENAI_API_KEY` |
| OpenAI GPT-4.1 Mini | `openai:gpt-4.1-mini` | `OPENAI_API_KEY` |
| OpenAI GPT-4.1 Nano | `openai:gpt-4.1-nano` | `OPENAI_API_KEY` |
| Anthropic Claude Sonnet 4 | `anthropic:claude-sonnet-4-6-20250514` | `ANTHROPIC_API_KEY` |
| Anthropic Claude Haiku 3.5 | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Google Gemini 2.5 Flash | `gemini/gemini-2.5-flash-preview-05-20` | `GEMINI_API_KEY` |
| Google Gemini 2.5 Pro | `gemini/gemini-2.5-pro-preview-06-05` | `GEMINI_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| Ollama (local) | `ollama/llama3.2-vision` | None (local) |
| AWS Textract | `aws-textract` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| Any LiteLLM model | Pass the LiteLLM model string directly | Varies |

---

## Components Guide

### 1. Schema Definition (Pydantic)

Define what you want to extract as a Pydantic model. Every field's `description` tells the LLM what to look for.

```python
from decimal import Decimal
from pydantic import BaseModel, Field

class W2Form(BaseModel):
    employer_ein: str | None = Field(default=None, description="Employer EIN (XX-XXXXXXX)")
    employer_name: str | None = Field(default=None, description="Employer Name")
    employee_ssn: str | None = Field(default=None, description="Employee SSN (XXX-XX-XXXX)")
    box1_wages: Decimal | None = Field(default=None, description="Box 1 - Wages, tips, other comp.")
    box2_fed_tax: Decimal | None = Field(default=None, description="Box 2 - Federal tax withheld")
```

**Tips:**
- Use `Optional` / `| None` for fields that may not be present
- Use `Decimal` for monetary values (avoids floating point issues)
- Write clear descriptions — they become part of the LLM prompt

---

### 2. Single Provider Extraction

The simplest usage — one LLM, one document.

```python
from extracture import Extractor

extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
)

result = extractor.extract("invoice.pdf")

# Access typed data
print(result.data.vendor)    # "Acme Corp"
print(result.data.total)     # 6696.0

# Access per-field metadata
for name, field in result.fields.items():
    print(f"{name}: {field.value} (confidence={field.confidence:.2f})")
```

---

### 3. Multi-Model Consensus

Run extraction through multiple LLMs and merge results via confidence-weighted voting. This is the core accuracy differentiator.

```python
extractor = Extractor(
    schema=Invoice,
    providers=[
        "openai:gpt-4o",
        "anthropic:claude-sonnet-4-6-20250514",
        "gemini/gemini-2.5-flash-preview-05-20",
    ],
    consensus="confidence_weighted",  # "majority" or "best_provider" also available
)

result = extractor.extract("invoice.pdf")

# See consensus details per field
for name, field in result.fields.items():
    if field.value is not None:
        print(f"{name}: {field.value}")
        print(f"  Consensus: {field.consensus_type}")   # "unanimous", "majority", "disagreement"
        print(f"  Confidence: {field.confidence:.2f}")
        for source in field.sources:
            print(f"    {source.provider}: {source.value} ({source.confidence:.2f})")
```

**Consensus strategies:**
- `confidence_weighted` (default) — weight votes by provider confidence, boost for unanimous agreement
- `majority` — simple majority vote, highest-confidence value wins ties
- `best_provider` — just pick the highest-confidence source per field

---

### 4. AWS Textract (Bounding Boxes)

Add Textract as an OCR provider alongside LLMs for bounding box data.

```bash
pip install extracture[textract]
```

```python
extractor = Extractor(
    schema=Invoice,
    providers=[
        "openai:gpt-4o",      # LLM extraction
        "aws-textract",        # OCR key-value extraction with bboxes
    ],
    api_keys={
        "AWS_ACCESS_KEY_ID": "your-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret",
        "AWS_DEFAULT_REGION": "us-east-1",
    },
)

result = extractor.extract("invoice.pdf")

for name, field in result.fields.items():
    if field.bbox:
        print(f"{name}: page={field.bbox.page}, x={field.bbox.x:.2f}, y={field.bbox.y:.2f}")
```

---

### 5. OCR Engines (Scanned Documents)

For scanned PDFs and images, extracture auto-detects and applies OCR.

```python
# Default: PyMuPDF (digital PDFs only, no OCR)
extractor = Extractor(schema=Invoice, providers=["openai:gpt-4o"], ocr_engine="pymupdf")

# Surya OCR (best accuracy, GPU recommended)
extractor = Extractor(schema=Invoice, providers=["openai:gpt-4o"], ocr_engine="surya")

# PaddleOCR (good accuracy, lightweight)
extractor = Extractor(schema=Invoice, providers=["openai:gpt-4o"], ocr_engine="paddleocr")

# Tesseract (CPU-only, widely available)
extractor = Extractor(schema=Invoice, providers=["openai:gpt-4o"], ocr_engine="tesseract")

# DocTR (good accuracy, PyTorch/TF)
extractor = Extractor(schema=Invoice, providers=["openai:gpt-4o"], ocr_engine="doctr")
```

The library auto-detects digital vs scanned PDFs:
- **Digital PDF** (has text layer): Extracts text directly (100% accurate, free)
- **Scanned PDF / Image**: Renders to images, applies OCR, then sends to LLM

---

### 6. Grounding Verification

Verify that extracted values actually exist in the source document. Catches LLM hallucinations.

```python
extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
    enable_grounding=True,       # Fuzzy string matching
    # enable_nli_grounding=True,  # + NLI model (pip install extracture[grounding])
)

result = extractor.extract("invoice.pdf")

for name, field in result.fields.items():
    if field.value is not None:
        status = "grounded" if field.is_grounded else "UNGROUNDED"
        print(f"{name}: {field.value} [{status}] (score={field.grounding_score:.2f})")
```

**How it works:**
1. Exact match — does the value appear verbatim in the document?
2. Fuzzy match — sliding window with rapidfuzz similarity
3. Quote verification — if the LLM provided a source quote, verify it exists
4. NLI model (optional) — uses DeBERTa to check if context supports the claim

Ungrounded fields get their confidence penalized by 50%.

---

### 7. Cross-Field Validation

Add business rules that check relationships between fields.

```python
extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
    validation_rules=[
        # Total must equal subtotal + tax
        (
            "total_check",                          # rule name
            ["subtotal", "tax", "total"],            # affected fields
            lambda f: (                              # check function
                None if f.total is None
                else None if abs(f.total - (f.subtotal or 0) - (f.tax or 0)) < 0.01
                else f"Total {f.total} != {f.subtotal} + {f.tax}"
            ),
            "warning",                               # severity: "error" or "warning"
        ),
    ],
)

result = extractor.extract("invoice.pdf")

for err in result.validation_errors:
    print(f"[{err.severity}] {err.message}")
    print(f"  Affected fields: {err.affected_fields}")
```

**Built-in validation helpers:**

```python
from extracture.verification.validator import (
    CrossFieldValidator,
    sum_equals_rule,
    date_not_future_rule,
    required_fields_rule,
)

validator = CrossFieldValidator()

# Auto-detect format rules from field names (EIN, SSN, state, zip, email, phone)
validator.auto_detect_format_rules(["employer_ein", "employee_ssn", "state"])

# Add custom rules
validator.add_rule(*sum_equals_rule("total", "subtotal", "tax"))
validator.add_rule(*required_fields_rule("vendor_name", "total"))

errors = validator.validate({"employer_ein": "12-3456789", "total": None})
```

---

### 8. Confidence Calibration

Raw LLM confidence scores are typically overconfident. Calibration applies temperature scaling so "90% confidence" actually means 90% correct.

```python
from extracture.verification.calibration import ConfidenceCalibrator

calibrator = ConfidenceCalibrator()

# Calibrate a raw score (default T=1.5 reduces overconfidence)
raw = 0.95
calibrated = calibrator.calibrate("vendor_name", raw)
print(f"Raw: {raw:.2f} -> Calibrated: {calibrated:.2f}")  # 0.95 -> 0.90

# Fit on validation data (list of (field_name, predicted_conf, was_correct))
calibrator.fit([
    ("vendor_name", 0.95, True),
    ("vendor_name", 0.90, True),
    ("vendor_name", 0.85, False),
    ("total", 0.99, True),
    ("total", 0.80, False),
])

# Save/load calibration parameters
calibrator.save("calibration.json")
calibrator.load("calibration.json")

# Measure calibration quality (ECE < 0.05 is good)
ece = calibrator.compute_ece([(0.9, True), (0.9, False), (0.5, True)])
print(f"ECE: {ece:.4f}")
```

Use with the Extractor:

```python
extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
    calibration_path="calibration.json",  # Load pre-fitted calibration
)

result = extractor.extract("invoice.pdf")
print(result.fields["total"].confidence)             # Raw
print(result.fields["total"].calibrated_confidence)   # Calibrated
print(result.fields["total"].effective_confidence)    # Uses calibrated if available
```

---

### 9. Template Matching (No LLM Needed)

For known document layouts, define spatial anchors and regex patterns. 520x faster and 3700x cheaper than LLM extraction.

```python
from extracture import Extractor, FieldAnchor

extractor = Extractor(
    schema=W2Form,
    providers=["openai:gpt-4o"],  # Only used as fallback for low-confidence fields
    template_anchors={
        "employer_ein": FieldAnchor(
            label="Employer identification number",
            direction="below",             # Value is below the label
            value_type="str",
            aliases=["EIN", "Employer ID"],
        ),
        "box1_wages": FieldAnchor(
            label="Wages, tips, other compensation",
            direction="right_and_below",
            value_type="decimal",
            regex_pattern=r"Wages.*compensation[:\s]*\$?([\d,]+\.?\d*)",  # Regex fallback
        ),
    },
)

result = extractor.extract("w2.pdf")
print(result.extraction_method)  # "template" if all fields matched with high confidence
```

**FieldAnchor options:**
- `label`: Text to search for on the document
- `direction`: `"right"`, `"below"`, or `"right_and_below"`
- `value_type`: `"str"`, `"decimal"`, `"int"`, `"bool"`, `"date"`
- `aliases`: Alternative labels to match
- `regex_pattern`: Regex with capture group for the value
- `max_distance_ratio`: Max distance from label as ratio of page dimension (default 0.15)

---

### 10. Human-in-the-Loop (HITL) Corrections

The library tells you which fields need review and stores corrections for future improvement.

```python
result = extractor.extract("invoice.pdf")

# Check what needs review
print(result.review_decision)  # AUTO_ACCEPT, PARTIAL_REVIEW, or FULL_REVIEW

# Get detailed review queue
queue = extractor.review(result)
for item in queue.items:
    print(f"Review: {item.field_name} = {item.current_value}")
    print(f"  Reason: {item.reason}")
    print(f"  Confidence: {item.confidence:.2f}")

# Apply corrections
result.correct("vendor_name", "Acme Corporation Inc.", corrected_by="john")
result.correct("total", 6700.00)

# Confirm all fields are correct
result.confirm()
print(result.status)  # "confirmed"
```

---

### 11. RAG Few-Shot Learning from Corrections

Store corrections and use them as few-shot examples for future extractions.

```python
extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
    enable_rag=True,
    correction_store_path="./corrections",
)

# After making corrections, store them
result.correct("vendor_name", "Acme Corporation Inc.")
extractor.learn_from_corrections(result)

# Future extractions on similar documents will use these corrections
# as few-shot examples in the prompt automatically
```

**Use the correction store directly:**

```python
from extracture.correction.store import CorrectionStore

store = CorrectionStore("./corrections")

# Add corrections manually
store.add_correction("Invoice", "vendor_name", "Acme Corp", "Acme Corporation Inc.")

# Get few-shot examples for a document type
examples = store.get_few_shot_examples("Invoice", max_examples=3)

# Build a prompt section from corrections
prompt = store.build_few_shot_prompt("Invoice")

# Stats
stats = store.get_correction_stats("Invoice")
print(f"Total corrections: {stats['total']}")
print(f"Most corrected: {stats['most_corrected_fields']}")
```

---

### 12. Batch Processing

Extract from multiple documents concurrently.

```python
results = extractor.extract_batch(
    ["inv1.pdf", "inv2.pdf", "inv3.pdf", "inv4.pdf"],
    max_concurrent=5,
)

for i, result in enumerate(results):
    print(f"Doc {i}: {result.overall_confidence:.2f} - {result.status.value}")
```

---

### 13. Preprocessing Pipeline

For scanned/degraded documents, the library auto-detects quality issues and applies preprocessing.

```python
from extracture.ingest.preprocessor import Preprocessor

preprocessor = Preprocessor()

# Assess image quality
with open("scanned.jpg", "rb") as f:
    quality = preprocessor.assess_quality(f.read())

print(f"DPI: {quality.estimated_dpi}")
print(f"Skew: {quality.skew_angle:.1f} degrees")
print(f"Contrast: {quality.contrast_score:.2f}")
print(f"Needs preprocessing: {quality.needs_preprocessing}")

# Apply preprocessing
with open("scanned.jpg", "rb") as f:
    processed_bytes, steps = preprocessor.preprocess(f.read(), quality)
    print(f"Applied: {steps}")  # ["deskew(2.3deg)", "upscale(150->300dpi)", "clahe_contrast"]
```

Preprocessing is automatic when using the `Extractor` — no manual steps needed.

---

### 14. Audit Trail

Every extraction includes a full audit trail.

```python
result = extractor.extract("document.pdf")

print(result.audit.providers_used)        # ["gpt-4o", "claude-sonnet-4-6-20250514"]
print(result.audit.extraction_method)     # "digital" or "scanned"
print(result.audit.ocr_engine)            # "surya"
print(result.audit.preprocessing_steps)   # ["deskew(1.5deg)"]
print(result.audit.reexamined_fields)     # ["vendor_name"]
print(result.audit.grounding_stats)       # {"grounded": 8, "ungrounded": 1}
print(result.audit.total_duration_ms)     # 3200
print(result.audit.cost_estimate_usd)     # 0.0045
```

---

### 15. CLI

```bash
# Basic extraction
extracture invoice.pdf --schema myapp.models:Invoice --providers openai:gpt-4o

# Multi-model with grounding
extracture w2.pdf \
  --schema myapp.schemas:W2Form \
  --providers openai:gpt-4o anthropic:claude-sonnet-4-6-20250514 \
  --ocr surya \
  --grounding \
  --output result.json

# Options
extracture --help
```

---

### 16. Using Components Independently

Every component works standalone — you don't have to use the full `Extractor`.

**PDF Parsing only:**

```python
from extracture.ingest.pdf import PDFParser

parser = PDFParser()
text, word_positions, page_dims, page_count = parser.extract_text(pdf_bytes)
page_images = parser.render_pages(pdf_bytes, dpi=300)
```

**Consensus engine only:**

```python
from extracture.consensus.engine import ConsensusEngine
from extracture.models import FieldResult, RawExtraction

engine = ConsensusEngine(strategy="confidence_weighted")

extractions = [
    RawExtraction(provider="model_a", fields={
        "total": FieldResult(value="$1,500", confidence=0.9),
    }),
    RawExtraction(provider="model_b", fields={
        "total": FieldResult(value="1500.00", confidence=0.85),
    }),
]

merged = engine.merge(extractions, ["total"])
print(merged["total"].value)           # "$1,500"
print(merged["total"].consensus_type)  # "unanimous" (normalized match)
```

**Grounding only:**

```python
from extracture.verification.grounding import GroundingVerifier

verifier = GroundingVerifier()
result = verifier.verify_field(
    field_value="Acme Corporation",
    source_quote=None,
    document_text="Invoice from Acme Corporation for consulting.",
)
print(result.is_grounded)  # True
print(result.method)       # "exact"
print(result.score)        # 1.0
```

**Calibration only:**

```python
from extracture.verification.calibration import ConfidenceCalibrator

cal = ConfidenceCalibrator()
cal.fit([("field", 0.9, True), ("field", 0.9, False)] * 50)
print(cal.calibrate("field", 0.9))  # Calibrated score
```

**Validation only:**

```python
from extracture.verification.validator import CrossFieldValidator, sum_equals_rule

validator = CrossFieldValidator()
validator.auto_detect_format_rules(["employer_ein", "employee_ssn"])
validator.add_rule(*sum_equals_rule("total", "subtotal", "tax"))

errors = validator.validate({"employer_ein": "invalid", "total": 100, "subtotal": 80, "tax": 30})
for e in errors:
    print(f"[{e.severity}] {e.message}")
```

**Template extraction only:**

```python
from extracture.templates.engine import TemplateExtractor
from extracture.schema import ExtractionSchema, FieldAnchor

engine = TemplateExtractor()
schema = ExtractionSchema(
    model=MySchema,
    template_anchors={
        "total": FieldAnchor(
            label="Total",
            direction="right",
            value_type="decimal",
            regex_pattern=r"Total[:\s]*\$?([\d,]+\.?\d*)",
        ),
    },
)
fields = engine.extract(schema, text_content, word_positions)
```

---

## Architecture

```
Input Document
      |
      v
[INGEST LAYER]
  - Auto-detect digital vs scanned
  - PDF text extraction (PyMuPDF, subprocess-isolated)
  - OCR (Surya / PaddleOCR / Tesseract / DocTR)
  - Image preprocessing (deskew, upscale, contrast, denoise)
      |
      v
[EXTRACT LAYER]
  - Template matching (regex + spatial anchors) — fast path
  - Multi-provider LLM extraction (parallel)
  - AWS Textract key-value extraction (bounding boxes)
  - Consensus merging (confidence-weighted voting)
  - Re-examination of low-confidence fields
      |
      v
[VERIFY LAYER]
  - Grounding verification (fuzzy match + NLI)
  - Confidence calibration (per-field temperature scaling)
  - Cross-field validation (business rules)
  - Self-correction on validation failures
      |
      v
[OUTPUT]
  ExtractionResult[T]
  - .data (typed Pydantic object)
  - .fields (per-field confidence, bbox, grounding, sources)
  - .review_decision (auto_accept / partial_review / full_review)
  - .audit (providers, duration, cost, preprocessing steps)
```

---

## Environment Variables

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...

# AWS (for Textract)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# Extracture config (all optional)
EXTRACTURE_CONFIDENCE_FLOOR=0.70
EXTRACTURE_REEXAMINE_THRESHOLD=0.85
EXTRACTURE_AUTO_ACCEPT_THRESHOLD=0.95
EXTRACTURE_DEFAULT_OCR_ENGINE=pymupdf
EXTRACTURE_ENABLE_GROUNDING=false
EXTRACTURE_LOG_LEVEL=INFO
```

## License

MIT
