"""Extracture — High-accuracy, schema-driven document extraction."""

from extracture.models import (
    BoundingBox,
    ExtractionMethod,
    ExtractionStatus,
    FieldResult,
    ExtractionResult,
    ReviewDecision,
    ValidationError,
)
from extracture.schema import ExtractionSchema, FieldAnchor
from extracture.extractor import Extractor

__version__ = "0.1.0"

__all__ = [
    "Extractor",
    "ExtractionSchema",
    "FieldAnchor",
    "BoundingBox",
    "ExtractionMethod",
    "ExtractionStatus",
    "FieldResult",
    "ExtractionResult",
    "ReviewDecision",
    "ValidationError",
]
