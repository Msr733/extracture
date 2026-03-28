"""Tests for core data models."""

import pytest
from pydantic import BaseModel, Field

from extracture.models import (
    BoundingBox,
    ExtractionResult,
    ExtractionStatus,
    FieldResult,
    RawExtraction,
    WordPosition,
)


class SampleSchema(BaseModel):
    name: str = Field(description="Name")
    amount: float = Field(description="Amount")


class TestBoundingBox:
    def test_basic_creation(self):
        bbox = BoundingBox(page=0, x=0.1, y=0.2, w=0.3, h=0.4)
        assert bbox.page == 0
        assert bbox.x == 0.1
        assert bbox.w == 0.3

    def test_computed_properties(self):
        bbox = BoundingBox(page=0, x=0.1, y=0.2, w=0.3, h=0.4)
        assert abs(bbox.x2 - 0.4) < 0.001
        assert abs(bbox.y2 - 0.6) < 0.001
        assert abs(bbox.area - 0.12) < 0.001

    def test_iou_identical(self):
        bbox = BoundingBox(page=0, x=0.1, y=0.1, w=0.5, h=0.5)
        assert abs(bbox.iou(bbox) - 1.0) < 0.001

    def test_iou_no_overlap(self):
        a = BoundingBox(page=0, x=0.0, y=0.0, w=0.1, h=0.1)
        b = BoundingBox(page=0, x=0.5, y=0.5, w=0.1, h=0.1)
        assert bbox_iou_zero(a, b)

    def test_iou_different_pages(self):
        a = BoundingBox(page=0, x=0.1, y=0.1, w=0.5, h=0.5)
        b = BoundingBox(page=1, x=0.1, y=0.1, w=0.5, h=0.5)
        assert a.iou(b) == 0.0


def bbox_iou_zero(a, b):
    return a.iou(b) == 0.0


class TestFieldResult:
    def test_default_values(self):
        fr = FieldResult()
        assert fr.value is None
        assert fr.confidence == 0.0
        assert fr.is_grounded is None
        assert fr.sources == []
        assert fr.flags == []

    def test_effective_confidence_uses_calibrated(self):
        fr = FieldResult(confidence=0.8, calibrated_confidence=0.7)
        assert fr.effective_confidence == 0.7

    def test_effective_confidence_falls_back(self):
        fr = FieldResult(confidence=0.8)
        assert fr.effective_confidence == 0.8


class TestExtractionResult:
    def test_correct_field(self):
        result = ExtractionResult(
            fields={
                "name": FieldResult(value="Old", confidence=0.8),
                "amount": FieldResult(value=100.0, confidence=0.9),
            }
        )
        result.correct("name", "New", corrected_by="test_user")

        assert result.fields["name"].value == "New"
        assert result.fields["name"].confidence == 1.0
        assert result.fields["name"].was_corrected is True
        assert result.status == ExtractionStatus.CORRECTED
        assert len(result.corrections) == 1
        assert result.corrections[0].original_value == "Old"
        assert result.corrections[0].corrected_value == "New"

    def test_correct_nonexistent_field_raises(self):
        result = ExtractionResult(fields={"name": FieldResult(value="X")})
        with pytest.raises(KeyError):
            result.correct("nonexistent", "value")

    def test_confirm(self):
        result = ExtractionResult()
        result.confirm()
        assert result.status == ExtractionStatus.CONFIRMED

    def test_low_confidence_fields(self):
        result = ExtractionResult(
            fields={
                "good": FieldResult(value="x", confidence=0.95),
                "bad": FieldResult(value="y", confidence=0.5),
                "null": FieldResult(value=None, confidence=0.1),
            }
        )
        low = result.get_low_confidence_fields(threshold=0.85)
        assert "bad" in low
        assert "good" not in low
        assert "null" not in low  # null values excluded

    def test_ungrounded_fields(self):
        result = ExtractionResult(
            fields={
                "grounded": FieldResult(value="x", is_grounded=True),
                "ungrounded": FieldResult(value="y", is_grounded=False),
                "unknown": FieldResult(value="z", is_grounded=None),
            }
        )
        ungrounded = result.get_ungrounded_fields()
        assert ungrounded == ["ungrounded"]

    def test_min_field_confidence(self):
        result = ExtractionResult(
            fields={
                "a": FieldResult(value="x", confidence=0.9),
                "b": FieldResult(value="y", confidence=0.5),
                "c": FieldResult(value=None, confidence=0.1),  # excluded (null)
            }
        )
        assert abs(result.min_field_confidence - 0.5) < 0.001

    def test_serialization(self):
        result = ExtractionResult(
            fields={"name": FieldResult(value="Test", confidence=0.95)},
            overall_confidence=0.95,
        )
        data = result.to_dict()
        assert data["overall_confidence"] == 0.95
        assert data["fields"]["name"]["value"] == "Test"

        json_str = result.to_json()
        assert '"Test"' in json_str


class TestWordPosition:
    def test_creation(self):
        wp = WordPosition(text="hello", page=0, x0=0.1, y0=0.2, x1=0.3, y1=0.25)
        assert wp.text == "hello"
        assert wp.page == 0
        assert wp.confidence == 1.0  # default


class TestRawExtraction:
    def test_with_error(self):
        raw = RawExtraction(provider="test", error="API timeout")
        assert raw.error == "API timeout"
        assert raw.fields == {}

    def test_with_fields(self):
        raw = RawExtraction(
            provider="openai:gpt-4o",
            fields={"name": FieldResult(value="Acme", confidence=0.95)},
            duration_ms=1500,
            cost_estimate_usd=0.005,
        )
        assert raw.fields["name"].value == "Acme"
        assert raw.cost_estimate_usd == 0.005
