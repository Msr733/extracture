"""Core data models for extracture."""

from __future__ import annotations

import math
from decimal import Decimal
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class ExtractionMethod(str, Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    TEMPLATE = "template"
    HYBRID = "hybrid"


class ExtractionStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    LOW_CONFIDENCE = "low_confidence"
    CORRECTED = "corrected"
    CONFIRMED = "confirmed"


class ReviewDecision(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    PARTIAL_REVIEW = "partial_review"
    FULL_REVIEW = "full_review"


class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 range) for a field on a page."""

    page: int
    x: float = Field(ge=0, le=1, description="Left edge (normalized)")
    y: float = Field(ge=0, le=1, description="Top edge (normalized)")
    w: float = Field(ge=0, le=1, description="Width (normalized)")
    h: float = Field(ge=0, le=1, description="Height (normalized)")

    @property
    def x2(self) -> float:
        return min(self.x + self.w, 1.0)

    @property
    def y2(self) -> float:
        return min(self.y + self.h, 1.0)

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: BoundingBox) -> float:
        if self.page != other.page:
            return 0.0
        ix0 = max(self.x, other.x)
        iy0 = max(self.y, other.y)
        ix1 = min(self.x2, other.x2)
        iy1 = min(self.y2, other.y2)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class SourceDetail(BaseModel):
    """What a single provider extracted for a single field."""

    provider: str
    value: Any = None
    raw_value: Any = None
    confidence: float = 0.0
    bbox: BoundingBox | None = None
    source_quote: str | None = None


class FieldResult(BaseModel):
    """Extraction result for a single field."""

    value: Any = None
    confidence: float = 0.0
    calibrated_confidence: float | None = None
    bbox: BoundingBox | None = None
    label_bbox: BoundingBox | None = None
    source_quote: str | None = None
    is_grounded: bool | None = None
    grounding_score: float | None = None
    consensus_type: str | None = None  # "unanimous", "majority", "disagreement"
    sources: list[SourceDetail] = Field(default_factory=list)
    was_reexamined: bool = False
    was_corrected: bool = False
    flags: list[str] = Field(default_factory=list)

    @property
    def effective_confidence(self) -> float:
        return self.calibrated_confidence if self.calibrated_confidence is not None else self.confidence


class ValidationError(BaseModel):
    """A validation error from cross-field or domain validation."""

    rule_name: str
    message: str
    affected_fields: list[str] = Field(default_factory=list)
    severity: str = "error"  # "error" or "warning"


class CorrectionRecord(BaseModel):
    """Record of a human correction."""

    field_name: str
    original_value: Any
    corrected_value: Any
    corrected_by: str | None = None
    corrected_at: str | None = None


class ExtractionAudit(BaseModel):
    """Full audit trail for the extraction."""

    extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL
    providers_used: list[str] = Field(default_factory=list)
    ocr_engine: str | None = None
    preprocessing_steps: list[str] = Field(default_factory=list)
    template_used: str | None = None
    total_duration_ms: float | None = None
    cost_estimate_usd: float | None = None
    reexamined_fields: list[str] = Field(default_factory=list)
    grounding_stats: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel, Generic[T]):
    """Complete result of a document extraction."""

    data: T | None = None
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    overall_confidence: float = 0.0
    calibrated_overall_confidence: float | None = None
    extraction_method: ExtractionMethod = ExtractionMethod.DIGITAL
    status: ExtractionStatus = ExtractionStatus.EXTRACTED
    review_decision: ReviewDecision = ReviewDecision.FULL_REVIEW
    validation_errors: list[ValidationError] = Field(default_factory=list)
    corrections: list[CorrectionRecord] = Field(default_factory=list)
    audit: ExtractionAudit = Field(default_factory=ExtractionAudit)
    page_count: int = 1
    page_image_urls: list[str] = Field(default_factory=list)

    @property
    def min_field_confidence(self) -> float:
        confs = [f.effective_confidence for f in self.fields.values() if f.value is not None]
        return min(confs) if confs else 0.0

    @property
    def all_grounded(self) -> bool:
        return all(
            f.is_grounded is True
            for f in self.fields.values()
            if f.value is not None and f.is_grounded is not None
        )

    def correct(self, field_name: str, corrected_value: Any, corrected_by: str | None = None) -> None:
        if field_name not in self.fields:
            raise KeyError(f"Field '{field_name}' not found in extraction result")
        field = self.fields[field_name]
        self.corrections.append(
            CorrectionRecord(
                field_name=field_name,
                original_value=field.value,
                corrected_value=corrected_value,
                corrected_by=corrected_by,
            )
        )
        field.value = corrected_value
        field.confidence = 1.0
        field.calibrated_confidence = 1.0
        field.was_corrected = True
        self.status = ExtractionStatus.CORRECTED

    def confirm(self) -> None:
        self.status = ExtractionStatus.CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def get_low_confidence_fields(self, threshold: float = 0.85) -> list[str]:
        return [
            name
            for name, field in self.fields.items()
            if field.value is not None and field.effective_confidence < threshold
        ]

    def get_ungrounded_fields(self) -> list[str]:
        return [
            name
            for name, field in self.fields.items()
            if field.value is not None and field.is_grounded is False
        ]


class WordPosition(BaseModel):
    """A word with its position in the document."""

    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0
    font_name: str | None = None
    font_size: float | None = None


class PageDimensions(BaseModel):
    """Dimensions of a document page."""

    page: int
    width: float
    height: float


class IngestResult(BaseModel):
    """Result of document ingestion (pre-extraction)."""

    file_type: str
    extraction_method: ExtractionMethod
    text_content: str | None = None
    word_positions: list[WordPosition] = Field(default_factory=list)
    page_images: list[bytes] = Field(default_factory=list, exclude=True)
    page_dims: list[PageDimensions] = Field(default_factory=list)
    page_count: int = 1
    preprocessing_applied: list[str] = Field(default_factory=list)
    ocr_engine_used: str | None = None


class RawExtraction(BaseModel):
    """Raw extraction output from a single provider."""

    provider: str
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    raw_response: str | None = None
    duration_ms: float | None = None
    cost_estimate_usd: float | None = None
    error: str | None = None
