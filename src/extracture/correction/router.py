"""HITL routing — determine which fields need human review."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from extracture.config import ExtractureConfig, get_config
from extracture.models import ExtractionResult, FieldResult, ReviewDecision

logger = logging.getLogger(__name__)


@dataclass
class ReviewItem:
    """A single field that needs human review."""

    field_name: str
    current_value: object = None
    confidence: float = 0.0
    reason: str = ""


@dataclass
class ReviewQueue:
    """Collection of fields needing human review."""

    decision: ReviewDecision
    items: list[ReviewItem] = field(default_factory=list)
    overall_confidence: float = 0.0

    @property
    def field_names(self) -> list[str]:
        return [item.field_name for item in self.items]

    @property
    def count(self) -> int:
        return len(self.items)


class HITLRouter:
    """Routes extraction results to human review based on confidence and validation."""

    def __init__(self, config: ExtractureConfig | None = None):
        self.config = config or get_config()

    def route(self, result: ExtractionResult[Any]) -> ReviewQueue:
        """Determine which fields need human review."""
        items: list[ReviewItem] = []

        # Check for validation errors
        error_fields: set[str] = set()
        for error in result.validation_errors:
            if error.severity == "error":
                error_fields.update(error.affected_fields)
                for field_name in error.affected_fields:
                    items.append(
                        ReviewItem(
                            field_name=field_name,
                            current_value=result.fields.get(field_name, FieldResult()).value,
                            confidence=result.fields.get(field_name, FieldResult()).confidence,
                            reason=f"validation_error: {error.message}",
                        )
                    )

        # Check field-level confidence
        for field_name, field_val in result.fields.items():
            if field_val.value is None:
                continue
            if field_name in error_fields:
                continue  # Already flagged

            if field_val.effective_confidence < self.config.auto_accept_threshold:
                items.append(
                    ReviewItem(
                        field_name=field_name,
                        current_value=field_val.value,
                        confidence=field_val.effective_confidence,
                        reason=f"low_confidence ({field_val.effective_confidence:.2f})",
                    )
                )

            elif field_val.is_grounded is False:
                items.append(
                    ReviewItem(
                        field_name=field_name,
                        current_value=field_val.value,
                        confidence=field_val.effective_confidence,
                        reason="ungrounded",
                    )
                )

        # Determine overall decision
        if not items:
            decision = ReviewDecision.AUTO_ACCEPT
        elif error_fields:
            decision = ReviewDecision.FULL_REVIEW
        else:
            decision = ReviewDecision.PARTIAL_REVIEW

        # Sort by confidence (lowest first — most important to review)
        items.sort(key=lambda x: x.confidence)

        return ReviewQueue(
            decision=decision,
            items=items,
            overall_confidence=result.overall_confidence,
        )
