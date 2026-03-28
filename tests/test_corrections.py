"""Tests for correction store and HITL router."""

import pytest

from extracture.correction.router import HITLRouter, ReviewDecision
from extracture.correction.store import CorrectionStore
from extracture.models import (
    ExtractionResult,
    ExtractionStatus,
    FieldResult,
    ValidationError,
)


class TestCorrectionStore:
    def test_add_and_retrieve(self, tmp_path):
        store = CorrectionStore(tmp_path / "corrections")
        store.add_correction(
            document_type="Invoice",
            field_name="vendor_name",
            original_value="Acme Corp",
            corrected_value="Acme Corporation",
            document_text_snippet="Invoice from Acme Corp...",
        )

        examples = store.get_few_shot_examples("Invoice")
        assert len(examples) == 1
        assert examples[0]["original_value"] == "Acme Corp"
        assert examples[0]["corrected_value"] == "Acme Corporation"

    def test_few_shot_prompt(self, tmp_path):
        store = CorrectionStore(tmp_path / "corrections")
        store.add_correction("Invoice", "total", "1500", "1500.00")

        prompt = store.build_few_shot_prompt("Invoice")
        assert prompt is not None
        assert "total" in prompt
        assert "1500.00" in prompt

    def test_no_examples_returns_none(self, tmp_path):
        store = CorrectionStore(tmp_path / "corrections")
        assert store.build_few_shot_prompt("Unknown") is None

    def test_persistence(self, tmp_path):
        store1 = CorrectionStore(tmp_path / "corrections")
        store1.add_correction("Invoice", "name", "Old", "New")

        store2 = CorrectionStore(tmp_path / "corrections")
        examples = store2.get_few_shot_examples("Invoice")
        assert len(examples) == 1

    def test_stats(self, tmp_path):
        store = CorrectionStore(tmp_path / "corrections")
        store.add_correction("Invoice", "name", "a", "b")
        store.add_correction("Invoice", "name", "c", "d")
        store.add_correction("Invoice", "total", "1", "2")
        store.add_correction("W2", "wages", "x", "y")

        stats = store.get_correction_stats("Invoice")
        assert stats["total"] == 3
        assert stats["most_corrected_fields"][0][0] == "name"

    def test_clear(self, tmp_path):
        store = CorrectionStore(tmp_path / "corrections")
        store.add_correction("Invoice", "name", "a", "b")
        store.add_correction("W2", "wages", "x", "y")

        removed = store.clear("Invoice")
        assert removed == 1
        assert len(store.get_few_shot_examples("Invoice")) == 0
        assert len(store.get_few_shot_examples("W2")) == 1


class TestHITLRouter:
    def setup_method(self):
        self.router = HITLRouter()

    def test_auto_accept_high_confidence(self):
        result = ExtractionResult(
            fields={
                "name": FieldResult(value="Acme", confidence=0.99, is_grounded=True),
                "total": FieldResult(value=100, confidence=0.98, is_grounded=True),
            },
            overall_confidence=0.985,
        )
        queue = self.router.route(result)
        assert queue.decision == ReviewDecision.AUTO_ACCEPT
        assert queue.count == 0

    def test_partial_review_low_confidence(self):
        result = ExtractionResult(
            fields={
                "name": FieldResult(value="Acme", confidence=0.99),
                "total": FieldResult(value=100, confidence=0.5),
            },
            overall_confidence=0.75,
        )
        queue = self.router.route(result)
        assert queue.decision == ReviewDecision.PARTIAL_REVIEW
        assert "total" in queue.field_names

    def test_full_review_validation_errors(self):
        result = ExtractionResult(
            fields={
                "total": FieldResult(value=100, confidence=0.99),
            },
            validation_errors=[
                ValidationError(
                    rule_name="sum_check",
                    message="Total mismatch",
                    affected_fields=["total"],
                    severity="error",
                )
            ],
        )
        queue = self.router.route(result)
        assert queue.decision == ReviewDecision.FULL_REVIEW

    def test_partial_review_ungrounded(self):
        result = ExtractionResult(
            fields={
                "name": FieldResult(value="Acme", confidence=0.99, is_grounded=True),
                "total": FieldResult(value=100, confidence=0.99, is_grounded=False),
            },
        )
        queue = self.router.route(result)
        assert queue.decision == ReviewDecision.PARTIAL_REVIEW
        assert "total" in queue.field_names
