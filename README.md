# extracture

High-accuracy, schema-driven document extraction with multi-model consensus, grounding verification, and human-in-the-loop correction.

## Install

```bash
pip install extracture
```

## Quick Start

```python
from pydantic import BaseModel, Field
from extracture import Extractor

class Invoice(BaseModel):
    vendor: str = Field(description="Vendor name")
    total: float = Field(description="Total amount")

extractor = Extractor(
    schema=Invoice,
    providers=["openai:gpt-4o"],
)

result = extractor.extract("invoice.pdf")
print(result.data)        # Invoice(vendor="Acme", total=1500.0)
print(result.confidence)  # 0.96
```

## Features

- **Schema-driven**: Define your output as a Pydantic model
- **Multi-model consensus**: Run multiple LLMs, merge with confidence-weighted voting
- **Grounding verification**: Verify extracted values exist in the source document
- **Confidence calibration**: Per-field temperature scaling for calibrated scores
- **Template matching**: 520x faster extraction for known document layouts
- **HITL corrections**: Store human corrections, improve via RAG few-shot
- **Crash-isolated PDF**: Subprocess-wrapped parsing prevents worker crashes
- **Any LLM, any OCR**: OpenAI, Anthropic, Gemini, Ollama, Surya, PaddleOCR, Tesseract

## License

MIT
