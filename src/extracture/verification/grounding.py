"""Grounding verification — verify extracted values exist in source document.

Inspired by SafePassage (2025): detects 85% of hallucinations using
fuzzy string alignment + NLI model at 1/2000th the cost of an LLM call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from extracture.config import ExtractureConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of grounding verification for a single field."""

    is_grounded: bool
    score: float = 0.0
    matched_text: str | None = None
    match_position: int | None = None
    method: str = "fuzzy"  # "exact", "fuzzy", "nli"


class GroundingVerifier:
    """Verifies that extracted values actually exist in the source document."""

    def __init__(
        self,
        config: ExtractureConfig | None = None,
        use_nli: bool = False,
    ):
        self.config = config or get_config()
        self.use_nli = use_nli
        self._nli_model = None

    def verify_field(
        self,
        field_value: str | None,
        source_quote: str | None,
        document_text: str,
        field_name: str = "",
    ) -> GroundingResult:
        """Verify a single field value is grounded in the document."""
        if field_value is None:
            return GroundingResult(is_grounded=True, score=1.0, method="null_field")

        value_str = str(field_value).strip()
        if not value_str:
            return GroundingResult(is_grounded=True, score=1.0, method="empty_field")

        if not document_text:
            return GroundingResult(is_grounded=False, score=0.0, method="no_document_text")

        # Step 1: Try exact match of the value in document text
        exact_result = self._exact_match(value_str, document_text)
        if exact_result.is_grounded:
            return exact_result

        # Step 2: Try fuzzy match of the value
        fuzzy_result = self._fuzzy_match(value_str, document_text)
        if fuzzy_result.is_grounded:
            return fuzzy_result

        # Step 3: If source_quote provided, verify the quote exists
        if source_quote:
            quote_result = self._verify_quote(source_quote, document_text)
            if quote_result.is_grounded:
                return quote_result

        # Step 4: NLI verification (optional, requires transformers)
        if self.use_nli and source_quote:
            nli_result = self._nli_verify(value_str, source_quote, document_text, field_name)
            return nli_result

        return GroundingResult(is_grounded=False, score=fuzzy_result.score, method="ungrounded")

    def verify_all_fields(
        self,
        fields: dict[str, Any],
        document_text: str,
    ) -> dict[str, GroundingResult]:
        """Verify all fields against the document text."""
        results: dict[str, GroundingResult] = {}
        for field_name, field_result in fields.items():
            value = field_result.value if hasattr(field_result, "value") else field_result
            source_quote = (
                field_result.source_quote if hasattr(field_result, "source_quote") else None
            )
            results[field_name] = self.verify_field(
                str(value) if value is not None else None,
                source_quote,
                document_text,
                field_name,
            )
        return results

    def _exact_match(self, value: str, document_text: str) -> GroundingResult:
        """Check if the value appears exactly in the document."""
        # Normalize for comparison
        value_normalized = value.lower().strip()
        doc_normalized = document_text.lower()

        pos = doc_normalized.find(value_normalized)
        if pos >= 0:
            return GroundingResult(
                is_grounded=True,
                score=1.0,
                matched_text=document_text[pos : pos + len(value)],
                match_position=pos,
                method="exact",
            )

        # Try without common separators
        value_clean = value_normalized.replace("-", "").replace(" ", "").replace(",", "")
        if len(value_clean) >= 3:
            doc_clean = doc_normalized.replace("-", "").replace(" ", "").replace(",", "")
            pos = doc_clean.find(value_clean)
            if pos >= 0:
                return GroundingResult(
                    is_grounded=True,
                    score=0.95,
                    matched_text=value_clean,
                    match_position=pos,
                    method="exact_normalized",
                )

        return GroundingResult(is_grounded=False, score=0.0, method="exact")

    def _fuzzy_match(self, value: str, document_text: str) -> GroundingResult:
        """Fuzzy match the value against sliding windows in the document."""
        threshold = self.config.grounding_similarity_threshold * 100  # rapidfuzz uses 0-100

        value_lower = value.lower().strip()
        doc_lower = document_text.lower()

        # For short values (< 10 chars), use partial ratio against the whole doc
        if len(value_lower) < 10:
            score = fuzz.partial_ratio(value_lower, doc_lower)
            if score >= threshold:
                return GroundingResult(
                    is_grounded=True,
                    score=score / 100.0,
                    method="fuzzy_partial",
                )

        # For longer values, use sliding window
        window_size = len(value_lower)
        best_score = 0.0
        best_pos = 0
        best_text = ""

        # Sample windows for efficiency on large documents
        step = max(1, window_size // 3)
        for i in range(0, len(doc_lower) - window_size + 1, step):
            window = doc_lower[i : i + window_size]
            score = fuzz.ratio(value_lower, window)
            if score > best_score:
                best_score = score
                best_pos = i
                best_text = document_text[i : i + window_size]

        if best_score >= threshold:
            return GroundingResult(
                is_grounded=True,
                score=best_score / 100.0,
                matched_text=best_text,
                match_position=best_pos,
                method="fuzzy_window",
            )

        return GroundingResult(
            is_grounded=False,
            score=best_score / 100.0,
            method="fuzzy_window",
        )

    def _verify_quote(self, source_quote: str, document_text: str) -> GroundingResult:
        """Verify that the source quote exists in the document."""
        quote_lower = source_quote.lower().strip()
        doc_lower = document_text.lower()

        # Exact quote match
        if quote_lower in doc_lower:
            pos = doc_lower.index(quote_lower)
            return GroundingResult(
                is_grounded=True,
                score=1.0,
                matched_text=document_text[pos : pos + len(source_quote)],
                match_position=pos,
                method="quote_exact",
            )

        # Fuzzy quote match
        score = fuzz.partial_ratio(quote_lower, doc_lower)
        threshold = self.config.grounding_similarity_threshold * 100

        if score >= threshold:
            return GroundingResult(
                is_grounded=True,
                score=score / 100.0,
                method="quote_fuzzy",
            )

        return GroundingResult(is_grounded=False, score=score / 100.0, method="quote_unmatched")

    def _nli_verify(
        self, value: str, source_quote: str, document_text: str, field_name: str
    ) -> GroundingResult:
        """Use NLI model to verify value against document context."""
        try:
            if self._nli_model is None:
                from transformers import pipeline

                self._nli_model = pipeline(
                    "text-classification",
                    model="cross-encoder/nli-deberta-v3-base",
                    device=-1,  # CPU
                )

            # Find the best matching context in the document
            context = self._find_context_window(source_quote or value, document_text, window=200)

            hypothesis = f"The {field_name.replace('_', ' ')} is {value}"
            nli_fn = self._nli_model
            assert nli_fn is not None
            result = nli_fn(
                f"{context} [SEP] {hypothesis}",
                top_k=3,
            )

            # Find entailment score
            entailment_score = 0.0
            for r in result:
                if r["label"].lower() == "entailment":
                    entailment_score = r["score"]
                    break

            is_grounded = entailment_score > self.config.grounding_nli_threshold

            return GroundingResult(
                is_grounded=is_grounded,
                score=entailment_score,
                matched_text=context[:100],
                method="nli",
            )

        except Exception as e:
            logger.warning(f"NLI verification failed: {e}")
            return GroundingResult(is_grounded=False, score=0.0, method="nli_error")

    def _find_context_window(self, query: str, document_text: str, window: int = 200) -> str:
        """Find the best context window in the document for the given query."""
        query_lower = query.lower()
        doc_lower = document_text.lower()

        # Try to find exact match position
        pos = doc_lower.find(query_lower[:20])
        if pos >= 0:
            start = max(0, pos - window // 4)
            end = min(len(document_text), pos + len(query) + window)
            return document_text[start:end]

        # Fallback: use partial ratio to find best window
        best_score = 0.0
        best_start = 0
        step = window // 2

        for i in range(0, len(doc_lower) - window, step):
            chunk = doc_lower[i : i + window]
            score = fuzz.partial_ratio(query_lower[:50], chunk)
            if score > best_score:
                best_score = score
                best_start = i

        return document_text[best_start : best_start + window]
