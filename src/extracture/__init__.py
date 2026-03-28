"""Extracture — High-accuracy, schema-driven document extraction."""

from extracture.extractor import Extractor
from extracture.models import (
    BoundingBox,
    ExtractionMethod,
    ExtractionResult,
    ExtractionStatus,
    FieldResult,
    ReviewDecision,
    ValidationError,
)
from extracture.schema import ExtractionSchema, FieldAnchor

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
