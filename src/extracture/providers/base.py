"""Base protocols for extraction and OCR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from extracture.models import FieldResult, IngestResult, RawExtraction
from extracture.schema import ExtractionSchema


class ExtractionProvider(ABC):
    """Base class for LLM-based extraction providers."""

    provider_name: str = "base"

    @abstractmethod
    async def extract(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
    ) -> RawExtraction:
        """Extract structured data from an ingested document."""
        ...

    @abstractmethod
    async def reexamine(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
        low_confidence_fields: dict[str, dict[str, Any]],
    ) -> RawExtraction:
        """Re-examine specific low-confidence fields."""
        ...

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a given number of tokens."""
        return 0.0


class OCRProvider(ABC):
    """Base class for OCR providers (Textract, etc.)."""

    provider_name: str = "base_ocr"

    @abstractmethod
    async def extract_key_values(
        self,
        file_bytes: bytes,
        schema: ExtractionSchema,
    ) -> RawExtraction:
        """Extract key-value pairs with bounding boxes."""
        ...
