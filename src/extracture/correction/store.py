"""Correction storage and RAG-based few-shot learning.

Stores human corrections and uses them to improve future extractions
via retrieval-augmented few-shot examples (11% F1 improvement per research).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from extracture.models import CorrectionRecord

logger = logging.getLogger(__name__)


class CorrectionStore:
    """Stores corrections and provides few-shot examples for RAG."""

    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".extracture" / "corrections"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._corrections: list[dict[str, Any]] = []
        self._load()

    def add_correction(
        self,
        document_type: str,
        field_name: str,
        original_value: Any,
        corrected_value: Any,
        document_text_snippet: str | None = None,
        corrected_by: str | None = None,
    ) -> None:
        """Store a correction for future learning."""
        record = {
            "document_type": document_type,
            "field_name": field_name,
            "original_value": str(original_value) if original_value is not None else None,
            "corrected_value": str(corrected_value) if corrected_value is not None else None,
            "document_text_snippet": document_text_snippet[:500] if document_text_snippet else None,
            "corrected_by": corrected_by,
            "timestamp": time.time(),
        }
        self._corrections.append(record)
        self._save()
        logger.info(
            f"Correction stored: {document_type}.{field_name}: "
            f"'{original_value}' → '{corrected_value}'"
        )

    def add_corrections_from_result(
        self,
        document_type: str,
        corrections: list[CorrectionRecord],
        document_text: str | None = None,
    ) -> None:
        """Store multiple corrections from an ExtractionResult."""
        for c in corrections:
            self.add_correction(
                document_type=document_type,
                field_name=c.field_name,
                original_value=c.original_value,
                corrected_value=c.corrected_value,
                document_text_snippet=document_text,
                corrected_by=c.corrected_by,
            )

    def get_few_shot_examples(
        self,
        document_type: str,
        document_text: str | None = None,
        max_examples: int = 3,
    ) -> list[dict[str, Any]]:
        """Get relevant few-shot examples for RAG-augmented extraction.

        Returns the most relevant past corrections as examples.
        """
        # Filter by document type
        type_corrections = [
            c for c in self._corrections if c["document_type"] == document_type
        ]

        if not type_corrections:
            return []

        if document_text:
            # Score by text similarity (simple overlap)
            scored = []
            doc_words = set(document_text.lower().split()[:100])

            for c in type_corrections:
                snippet = c.get("document_text_snippet", "")
                if snippet:
                    snippet_words = set(snippet.lower().split())
                    overlap = len(doc_words & snippet_words) / max(len(doc_words), 1)
                    scored.append((overlap, c))
                else:
                    scored.append((0, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored[:max_examples]]
        else:
            # Return most recent corrections
            sorted_corrections = sorted(
                type_corrections, key=lambda c: c.get("timestamp", 0), reverse=True
            )
            return sorted_corrections[:max_examples]

    def build_few_shot_prompt(
        self,
        document_type: str,
        document_text: str | None = None,
        max_examples: int = 3,
    ) -> str | None:
        """Build a few-shot prompt section from past corrections."""
        examples = self.get_few_shot_examples(document_type, document_text, max_examples)

        if not examples:
            return None

        lines = [
            "Based on previous corrections for similar documents, please note:",
            "",
        ]

        for ex in examples:
            lines.append(
                f"  - Field '{ex['field_name']}': "
                f"'{ex['original_value']}' was corrected to '{ex['corrected_value']}'"
            )

        lines.append("")
        lines.append("Apply similar corrections proactively where applicable.")

        return "\n".join(lines)

    def get_correction_stats(self, document_type: str | None = None) -> dict[str, Any]:
        """Get statistics about stored corrections."""
        corrections = self._corrections
        if document_type:
            corrections = [c for c in corrections if c["document_type"] == document_type]

        if not corrections:
            return {"total": 0}

        # Most corrected fields
        field_counts: dict[str, int] = {}
        for c in corrections:
            field = c["field_name"]
            field_counts[field] = field_counts.get(field, 0) + 1

        top_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total": len(corrections),
            "document_types": list(set(c["document_type"] for c in corrections)),
            "most_corrected_fields": top_fields,
        }

    def clear(self, document_type: str | None = None) -> int:
        """Clear corrections. Returns count of removed corrections."""
        if document_type:
            original = len(self._corrections)
            self._corrections = [
                c for c in self._corrections if c["document_type"] != document_type
            ]
            removed = original - len(self._corrections)
        else:
            removed = len(self._corrections)
            self._corrections = []

        self._save()
        return removed

    def _save(self) -> None:
        """Persist corrections to disk."""
        path = self.storage_path / "corrections.jsonl"
        with open(path, "w") as f:
            for record in self._corrections:
                f.write(json.dumps(record, default=str) + "\n")

    def _load(self) -> None:
        """Load corrections from disk."""
        path = self.storage_path / "corrections.jsonl"
        if not path.exists():
            return

        self._corrections = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self._corrections.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        logger.debug(f"Loaded {len(self._corrections)} corrections from {path}")
