"""Tests for the consensus engine."""

import pytest

from extracture.consensus.engine import ConsensusEngine
from extracture.models import BoundingBox, FieldResult, RawExtraction


class TestConsensusEngine:
    def setup_method(self):
        self.engine = ConsensusEngine()

    def test_single_source(self):
        ext = RawExtraction(
            provider="openai",
            fields={"name": FieldResult(value="Acme", confidence=0.9)},
        )
        result = self.engine.merge([ext], ["name"])
        assert result["name"].value == "Acme"
        assert result["name"].confidence == 0.9

    def test_unanimous_agreement_boosts_confidence(self):
        exts = [
            RawExtraction(
                provider="openai",
                fields={"total": FieldResult(value="1500.00", confidence=0.85)},
            ),
            RawExtraction(
                provider="anthropic",
                fields={"total": FieldResult(value="$1,500.00", confidence=0.90)},
            ),
            RawExtraction(
                provider="gemini",
                fields={"total": FieldResult(value="1500", confidence=0.88)},
            ),
        ]
        result = self.engine.merge(exts, ["total"])
        assert result["total"].consensus_type == "unanimous"
        assert result["total"].confidence > 0.88  # boosted

    def test_majority_vote(self):
        exts = [
            RawExtraction(
                provider="a",
                fields={"name": FieldResult(value="Acme Corp", confidence=0.9)},
            ),
            RawExtraction(
                provider="b",
                fields={"name": FieldResult(value="Acme Corp", confidence=0.85)},
            ),
            RawExtraction(
                provider="c",
                fields={"name": FieldResult(value="ACME LLC", confidence=0.7)},
            ),
        ]
        result = self.engine.merge(exts, ["name"])
        assert result["name"].value == "Acme Corp"
        assert result["name"].consensus_type == "majority"

    def test_disagreement_penalizes(self):
        exts = [
            RawExtraction(
                provider="a",
                fields={"total": FieldResult(value="100", confidence=0.9)},
            ),
            RawExtraction(
                provider="b",
                fields={"total": FieldResult(value="200", confidence=0.8)},
            ),
        ]
        result = self.engine.merge(exts, ["total"])
        assert result["total"].consensus_type == "disagreement"
        assert result["total"].confidence < 0.9  # penalized

    def test_all_null_consensus(self):
        exts = [
            RawExtraction(
                provider="a",
                fields={"field": FieldResult(value=None, confidence=0.0)},
            ),
            RawExtraction(
                provider="b",
                fields={"field": FieldResult(value=None, confidence=0.0)},
            ),
        ]
        result = self.engine.merge(exts, ["field"])
        assert result["field"].value is None
        assert result["field"].consensus_type == "unanimous_null"

    def test_bbox_from_trusted_source(self):
        textract_bbox = BoundingBox(page=0, x=0.1, y=0.2, w=0.3, h=0.04)
        exts = [
            RawExtraction(
                provider="openai",
                fields={"total": FieldResult(value="100", confidence=0.9)},
            ),
            RawExtraction(
                provider="textract",
                fields={"total": FieldResult(value="100", confidence=0.95, bbox=textract_bbox)},
            ),
        ]
        result = self.engine.merge(exts, ["total"])
        assert result["total"].bbox is not None
        assert result["total"].bbox.x == 0.1

    def test_monetary_normalization(self):
        """$1,500.00, 1500, and 1500.00 should all be treated as equal."""
        exts = [
            RawExtraction(
                provider="a",
                fields={"total": FieldResult(value="$1,500.00", confidence=0.85)},
            ),
            RawExtraction(
                provider="b",
                fields={"total": FieldResult(value="1500", confidence=0.90)},
            ),
        ]
        result = self.engine.merge(exts, ["total"])
        assert result["total"].consensus_type == "unanimous"

    def test_empty_extractions(self):
        result = self.engine.merge([], ["field"])
        assert result == {}

    def test_missing_field_in_some_providers(self):
        exts = [
            RawExtraction(
                provider="a",
                fields={"name": FieldResult(value="Acme", confidence=0.9)},
            ),
            RawExtraction(
                provider="b",
                fields={},  # doesn't have "name"
            ),
        ]
        result = self.engine.merge(exts, ["name"])
        assert result["name"].value == "Acme"

    def test_best_provider_strategy(self):
        engine = ConsensusEngine(strategy="best_provider")
        exts = [
            RawExtraction(
                provider="a",
                fields={"name": FieldResult(value="Low", confidence=0.5)},
            ),
            RawExtraction(
                provider="b",
                fields={"name": FieldResult(value="High", confidence=0.99)},
            ),
        ]
        result = engine.merge(exts, ["name"])
        assert result["name"].value == "High"
        assert result["name"].confidence == 0.99
