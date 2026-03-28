"""Template-based extraction for known document layouts.

Research: Template matching is 520x faster and 3700x cheaper than VLM extraction
on structured documents (Berkeley 2025). For known forms (tax docs, standardized
invoices), regex + spatial rules achieve 95-99% accuracy with zero LLM cost.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rapidfuzz import fuzz

from extracture.models import BoundingBox, FieldResult, WordPosition
from extracture.schema import ExtractionSchema, FieldAnchor

logger = logging.getLogger(__name__)


class TemplateExtractor:
    """Extract fields using spatial anchors and regex patterns — no LLM needed."""

    def __init__(self, fuzzy_threshold: float = 75.0):
        self.fuzzy_threshold = fuzzy_threshold

    def extract(
        self,
        schema: ExtractionSchema,
        text_content: str | None,
        word_positions: list[WordPosition],
        page_dims: list[dict] | None = None,
    ) -> dict[str, FieldResult]:
        """Extract fields using template anchors."""
        if not schema.has_template:
            return {}

        fields: dict[str, FieldResult] = {}

        for field_name, anchor in schema.template_anchors.items():
            result = self._extract_field(
                field_name, anchor, text_content, word_positions
            )
            if result:
                fields[field_name] = result

        logger.info(
            f"Template extraction: {len(fields)}/{len(schema.template_anchors)} fields extracted"
        )
        return fields

    def _extract_field(
        self,
        field_name: str,
        anchor: FieldAnchor,
        text_content: str | None,
        word_positions: list[WordPosition],
    ) -> FieldResult | None:
        """Extract a single field using its anchor."""

        # Strategy 1: Regex pattern on full text
        if anchor.regex_pattern and text_content:
            match = re.search(anchor.regex_pattern, text_content, re.IGNORECASE)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                value = self._normalize_value(value.strip(), anchor.value_type)
                return FieldResult(
                    value=value,
                    confidence=0.95,
                    source_quote=match.group(0),
                )

        # Strategy 2: Spatial anchor matching on word positions
        if word_positions:
            return self._spatial_extract(field_name, anchor, word_positions)

        # Strategy 3: Simple text search
        if text_content:
            return self._text_search_extract(field_name, anchor, text_content)

        return None

    def _spatial_extract(
        self,
        field_name: str,
        anchor: FieldAnchor,
        word_positions: list[WordPosition],
    ) -> FieldResult | None:
        """Extract using spatial relationships between anchor label and value."""
        # Find the anchor label in word positions
        all_labels = [anchor.label.lower()] + [a.lower() for a in anchor.aliases]

        anchor_pos = None
        best_match_score = 0.0

        for wp in word_positions:
            for label in all_labels:
                # Build multi-word anchor from consecutive words
                label_words = label.split()
                if len(label_words) == 1:
                    score = fuzz.ratio(wp.text.lower(), label)
                    if score > best_match_score and score >= self.fuzzy_threshold:
                        best_match_score = score
                        anchor_pos = wp

        if not anchor_pos:
            return None

        # Find value based on direction
        value_words = self._find_value_words(
            anchor_pos, anchor.direction, word_positions, anchor.max_distance_ratio
        )

        if not value_words:
            return None

        # Combine words into value
        value_text = " ".join(w.text for w in value_words)
        value = self._normalize_value(value_text.strip(), anchor.value_type)

        # Build bounding box from value words
        bbox = self._words_to_bbox(value_words)
        anchor_bbox = BoundingBox(
            page=anchor_pos.page,
            x=anchor_pos.x0,
            y=anchor_pos.y0,
            w=anchor_pos.x1 - anchor_pos.x0,
            h=anchor_pos.y1 - anchor_pos.y0,
        )

        avg_conf = sum(w.confidence for w in value_words) / len(value_words)

        return FieldResult(
            value=value,
            confidence=round(min(avg_conf, best_match_score / 100.0), 4),
            bbox=bbox,
            label_bbox=anchor_bbox,
            source_quote=f"{anchor.label}: {value_text}",
        )

    def _find_value_words(
        self,
        anchor: WordPosition,
        direction: str,
        all_words: list[WordPosition],
        max_distance: float,
    ) -> list[WordPosition]:
        """Find value words in the specified direction from the anchor."""
        candidates: list[WordPosition] = []

        for w in all_words:
            if w.page != anchor.page:
                continue
            if w.text == anchor.text and abs(w.x0 - anchor.x0) < 0.01:
                continue  # Skip the anchor itself

            if direction == "right":
                # Value is to the right, same vertical band
                if (
                    w.x0 > anchor.x1 - 0.02
                    and abs(w.y0 - anchor.y0) < max_distance
                    and w.x0 - anchor.x1 < max_distance * 3
                ):
                    candidates.append(w)

            elif direction == "below":
                # Value is below, same horizontal band
                if (
                    w.y0 > anchor.y1 - 0.02
                    and abs(w.x0 - anchor.x0) < max_distance * 2
                    and w.y0 - anchor.y1 < max_distance * 2
                ):
                    candidates.append(w)

            elif direction == "right_and_below":
                # Value is in a box to the right and below
                if (
                    (w.x0 > anchor.x0 - 0.02 or w.y0 > anchor.y1 - 0.02)
                    and w.x0 - anchor.x0 < max_distance * 4
                    and w.y0 - anchor.y0 < max_distance * 4
                ):
                    candidates.append(w)

        # Sort by position (top-left to bottom-right)
        candidates.sort(key=lambda w: (w.y0, w.x0))

        # Take the first few words (heuristic: stop at large gap)
        result: list[WordPosition] = []
        for i, w in enumerate(candidates):
            if i > 0:
                prev = candidates[i - 1]
                # Stop at large horizontal or vertical gap
                if w.y0 - prev.y1 > max_distance or (
                    w.x0 - prev.x1 > max_distance * 2 and abs(w.y0 - prev.y0) < 0.01
                ):
                    break
            result.append(w)
            if len(result) >= 10:
                break

        return result

    def _text_search_extract(
        self,
        field_name: str,
        anchor: FieldAnchor,
        text_content: str,
    ) -> FieldResult | None:
        """Simple text-based extraction using label search."""
        labels = [anchor.label] + anchor.aliases
        text_lower = text_content.lower()

        for label in labels:
            label_lower = label.lower()
            pos = text_lower.find(label_lower)
            if pos < 0:
                continue

            # Extract text after the label
            after = text_content[pos + len(label) :].strip()
            # Remove common separators
            after = re.sub(r"^[:\s]+", "", after)
            # Take first line/value
            value_match = re.match(r"^([^\n\r]{1,100})", after)
            if value_match:
                value = self._normalize_value(value_match.group(1).strip(), anchor.value_type)
                return FieldResult(
                    value=value,
                    confidence=0.80,
                    source_quote=f"{label}: {value_match.group(1).strip()}",
                )

        return None

    def _normalize_value(self, text: str, value_type: str) -> Any:
        """Normalize extracted text to the expected type."""
        if not text:
            return None

        if value_type == "decimal":
            # Remove $ and commas, parse as float
            cleaned = re.sub(r"[$,\s]", "", text)
            try:
                return float(cleaned)
            except ValueError:
                return text

        elif value_type == "int":
            cleaned = re.sub(r"[,\s]", "", text)
            try:
                return int(cleaned)
            except ValueError:
                return text

        elif value_type == "bool":
            lower = text.lower().strip()
            if lower in ("true", "yes", "x", "✓", "✗", "checked"):
                return True
            elif lower in ("false", "no", "", "unchecked"):
                return False
            return text

        elif value_type == "date":
            # Try to normalize to YYYY-MM-DD
            # Handle MM/DD/YYYY
            m = re.match(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", text)
            if m:
                return f"{m.group(3)}-{m.group(1):0>2}-{m.group(2):0>2}"
            return text

        return text

    def _words_to_bbox(self, words: list[WordPosition]) -> BoundingBox | None:
        """Create a bounding box encompassing all words."""
        if not words:
            return None

        x0 = min(w.x0 for w in words)
        y0 = min(w.y0 for w in words)
        x1 = max(w.x1 for w in words)
        y1 = max(w.y1 for w in words)

        return BoundingBox(
            page=words[0].page,
            x=round(x0, 4),
            y=round(y0, 4),
            w=round(x1 - x0, 4),
            h=round(y1 - y0, 4),
        )
