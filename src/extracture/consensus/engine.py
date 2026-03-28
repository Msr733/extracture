"""Multi-source consensus engine with confidence-weighted voting.

Research-backed: confidence-weighted voting gives 7.4% accuracy improvement
over naive majority voting (CER framework, 2025).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

from extracture.models import BoundingBox, FieldResult, RawExtraction, SourceDetail

logger = logging.getLogger(__name__)

# Sources that provide reliable bounding boxes (never trust LLM bboxes)
TRUSTED_BBOX_SOURCES = {"textract", "surya", "paddleocr", "doctr", "tesseract"}


class ConsensusEngine:
    """Merges extractions from multiple providers using confidence-weighted voting."""

    def __init__(
        self,
        strategy: str = "confidence_weighted",
        agreement_boost: float = 0.05,
        disagreement_penalty: float = 0.8,
    ):
        self.strategy = strategy
        self.agreement_boost = agreement_boost
        self.disagreement_penalty = disagreement_penalty

    def merge(
        self,
        extractions: list[RawExtraction],
        field_names: list[str],
    ) -> dict[str, FieldResult]:
        """Merge multiple extraction results into a single consensus result."""
        if not extractions:
            return {}

        # Single source — no consensus needed
        if len(extractions) == 1:
            return dict(extractions[0].fields)

        consensus: dict[str, FieldResult] = {}

        for field_name in field_names:
            sources = self._gather_sources(field_name, extractions)

            if not sources:
                consensus[field_name] = FieldResult(value=None, confidence=0.0)
                continue

            if self.strategy == "confidence_weighted":
                consensus[field_name] = self._confidence_weighted_vote(field_name, sources)
            elif self.strategy == "majority":
                consensus[field_name] = self._majority_vote(field_name, sources)
            elif self.strategy == "best_provider":
                consensus[field_name] = self._best_provider(field_name, sources)
            else:
                consensus[field_name] = self._confidence_weighted_vote(field_name, sources)

        return consensus

    def _gather_sources(
        self, field_name: str, extractions: list[RawExtraction]
    ) -> list[SourceDetail]:
        """Gather all source values for a field across providers."""
        sources = []
        for ext in extractions:
            field = ext.fields.get(field_name)
            if field is None:
                continue
            sources.append(
                SourceDetail(
                    provider=ext.provider,
                    value=field.value,
                    raw_value=field.value,
                    confidence=field.confidence,
                    bbox=field.bbox,
                    source_quote=field.source_quote,
                )
            )
        return sources

    def _confidence_weighted_vote(
        self, field_name: str, sources: list[SourceDetail]
    ) -> FieldResult:
        """Confidence-weighted voting — groups by normalized value, picks highest-weight group."""
        # Filter out null/zero-confidence sources
        active_sources = [s for s in sources if s.value is not None and s.confidence > 0]

        if not active_sources:
            # All sources say null — that's a consensus
            return FieldResult(
                value=None,
                confidence=1.0 if len(sources) > 1 else 0.0,
                consensus_type="unanimous_null",
                sources=sources,
            )

        # Group by normalized value
        groups: dict[str, list[SourceDetail]] = defaultdict(list)
        for source in active_sources:
            normalized = self._normalize_value(source.value)
            groups[normalized].append(source)

        # Pick group with highest total confidence weight
        best_key = max(groups, key=lambda k: sum(s.confidence for s in groups[k]))
        best_group = groups[best_key]
        total_sources = len(active_sources)
        agreement_ratio = len(best_group) / total_sources

        # Pick the highest-confidence source from the winning group
        best_source = max(best_group, key=lambda s: s.confidence)

        # Calculate consensus confidence
        avg_conf = sum(s.confidence for s in best_group) / len(best_group)

        if agreement_ratio == 1.0:
            # Unanimous agreement — boost confidence
            consensus_type = "unanimous"
            final_conf = min(avg_conf + self.agreement_boost * (total_sources - 1), 1.0)
        elif agreement_ratio > 0.5:
            # Majority agreement
            consensus_type = "majority"
            final_conf = avg_conf
        else:
            # Disagreement — penalize
            consensus_type = "disagreement"
            final_conf = best_source.confidence * self.disagreement_penalty

        # Select bbox from trusted OCR sources only
        bbox = self._select_best_bbox(sources)

        return FieldResult(
            value=best_source.value,
            confidence=round(final_conf, 4),
            bbox=bbox,
            source_quote=best_source.source_quote,
            consensus_type=consensus_type,
            sources=sources,
        )

    def _majority_vote(self, field_name: str, sources: list[SourceDetail]) -> FieldResult:
        """Simple majority vote — most common normalized value wins."""
        active_sources = [s for s in sources if s.value is not None and s.confidence > 0]

        if not active_sources:
            return FieldResult(value=None, confidence=0.0, sources=sources)

        groups: dict[str, list[SourceDetail]] = defaultdict(list)
        for source in active_sources:
            normalized = self._normalize_value(source.value)
            groups[normalized].append(source)

        best_key = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_key]
        best_source = max(best_group, key=lambda s: s.confidence)

        return FieldResult(
            value=best_source.value,
            confidence=best_source.confidence,
            bbox=self._select_best_bbox(sources),
            source_quote=best_source.source_quote,
            consensus_type="majority" if len(best_group) > 1 else "single",
            sources=sources,
        )

    def _best_provider(self, field_name: str, sources: list[SourceDetail]) -> FieldResult:
        """Simply pick the highest-confidence source."""
        active = [s for s in sources if s.value is not None and s.confidence > 0]
        if not active:
            return FieldResult(value=None, confidence=0.0, sources=sources)

        best = max(active, key=lambda s: s.confidence)
        return FieldResult(
            value=best.value,
            confidence=best.confidence,
            bbox=best.bbox or self._select_best_bbox(sources),
            source_quote=best.source_quote,
            consensus_type="best_provider",
            sources=sources,
        )

    def _normalize_value(self, value: Any) -> str:
        """Normalize a value for comparison across sources."""
        if value is None:
            return "__null__"

        text = str(value).strip()

        # Normalize monetary: strip $, commas, parse as decimal
        cleaned = re.sub(r"[$,\s]", "", text)
        try:
            decimal_val = Decimal(cleaned)
            return str(decimal_val.normalize())
        except (InvalidOperation, ValueError):
            pass

        # Normalize text: lowercase, strip whitespace
        return text.lower().strip()

    def _select_best_bbox(self, sources: list[SourceDetail]) -> BoundingBox | None:
        """Select bbox from trusted OCR sources only (never LLM-generated bboxes)."""
        for source in sources:
            if source.bbox and any(
                trusted in source.provider.lower() for trusted in TRUSTED_BBOX_SOURCES
            ):
                return source.bbox

        # Fallback: any bbox is better than none, but flag it
        for source in sources:
            if source.bbox:
                return source.bbox

        return None
