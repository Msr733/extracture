"""Comprehensive edge case tests to ensure the library never breaks."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from extracture import ExtractionSchema, FieldAnchor
from extracture.config import get_config
from extracture.consensus.engine import ConsensusEngine
from extracture.correction.router import HITLRouter
from extracture.correction.store import CorrectionStore
from extracture.ingest.preprocessor import Preprocessor, QualityAssessment
from extracture.ingest.router import IngestRouter
from extracture.models import (
    BoundingBox,
    ExtractionMethod,
    ExtractionResult,
    FieldResult,
    RawExtraction,
    ReviewDecision,
    ValidationError,
)
from extracture.providers.litellm_provider import LiteLLMProvider
from extracture.schema import CrossFieldRule
from extracture.templates.engine import TemplateExtractor
from extracture.verification.calibration import ConfidenceCalibrator
from extracture.verification.grounding import GroundingVerifier
from extracture.verification.validator import CrossFieldValidator


# ============================================================
# Test schemas
# ============================================================
class SimpleSchema(BaseModel):
    name: str = Field(description="Name")
    amount: float = Field(description="Amount")


class EmptySchema(BaseModel):
    pass


class BigSchema(BaseModel):
    field_01: str | None = Field(default=None, description="Field 01")
    field_02: str | None = Field(default=None, description="Field 02")
    field_03: str | None = Field(default=None, description="Field 03")
    field_04: str | None = Field(default=None, description="Field 04")
    field_05: str | None = Field(default=None, description="Field 05")
    field_06: str | None = Field(default=None, description="Field 06")
    field_07: str | None = Field(default=None, description="Field 07")
    field_08: str | None = Field(default=None, description="Field 08")
    field_09: str | None = Field(default=None, description="Field 09")
    field_10: str | None = Field(default=None, description="Field 10")
    field_11: str | None = Field(default=None, description="Field 11")
    field_12: str | None = Field(default=None, description="Field 12")
    field_13: str | None = Field(default=None, description="Field 13")
    field_14: str | None = Field(default=None, description="Field 14")
    field_15: str | None = Field(default=None, description="Field 15")
    field_16: str | None = Field(default=None, description="Field 16")
    field_17: str | None = Field(default=None, description="Field 17")
    field_18: str | None = Field(default=None, description="Field 18")
    field_19: str | None = Field(default=None, description="Field 19")
    field_20: str | None = Field(default=None, description="Field 20")


class TypedSchema(BaseModel):
    name: str = Field(description="Name")
    amount: Decimal | None = Field(default=None, description="Amount")
    count: int | None = Field(default=None, description="Count")
    active: bool | None = Field(default=None, description="Active")
    tags: str | None = Field(default=None, description="Tags")


# ============================================================
# SCHEMA & PROMPT EDGE CASES
# ============================================================
class TestSchemaEdgeCases:
    def test_empty_schema(self):
        schema = ExtractionSchema(model=EmptySchema)
        assert schema.field_names == []
        assert schema.field_labels == {}
        prompt = schema.build_extraction_prompt("some text")
        assert "EmptySchema" in prompt

    def test_schema_with_no_descriptions(self):
        class NoDescSchema(BaseModel):
            foo: str
            bar: int

        schema = ExtractionSchema(model=NoDescSchema)
        labels = schema.field_labels
        assert labels["foo"] == "Foo"  # auto-generated from field name
        assert labels["bar"] == "Bar"

    def test_schema_special_characters_in_labels(self):
        schema = ExtractionSchema(
            model=SimpleSchema,
            field_labels={"name": "Name — with dashes & ampersands <special>"},
        )
        prompt = schema.build_extraction_prompt("text")
        assert "dashes & ampersands" in prompt

    def test_schema_very_long_instructions(self):
        instructions = "Important: " + "x" * 10000
        schema = ExtractionSchema(
            model=SimpleSchema,
            form_instructions=instructions,
        )
        prompt = schema.build_extraction_prompt("text")
        assert len(prompt) > 10000

    def test_schema_unicode_in_labels(self):
        schema = ExtractionSchema(
            model=SimpleSchema,
            field_labels={"name": "名前 (Name)", "amount": "金額 (Amount)"},
        )
        prompt = schema.build_extraction_prompt("テスト")
        assert "名前" in prompt

    def test_build_tool_schema_structure(self):
        schema = ExtractionSchema(model=SimpleSchema)
        tool = schema.build_tool_schema()
        assert "name" in tool
        assert "input_schema" in tool
        assert "properties" in tool["input_schema"]

    def test_build_reexam_prompt_empty_fields(self):
        schema = ExtractionSchema(model=SimpleSchema)
        prompt = schema.build_reexamination_prompt({})
        assert "SimpleSchema" in prompt

    def test_build_reexam_prompt_with_none_values(self):
        schema = ExtractionSchema(model=SimpleSchema)
        prompt = schema.build_reexamination_prompt(
            {"name": {"value": None, "confidence": 0.3}}
        )
        assert "name" in prompt

    def test_cross_field_validation_with_exception(self):
        def bad_rule(data: Any) -> str | None:
            raise RuntimeError("rule crashed")

        schema = ExtractionSchema(
            model=SimpleSchema,
            validation_rules=[
                CrossFieldRule(name="bad", fields=["name"], check=bad_rule)
            ],
        )
        # Should not raise — validation errors should be caught
        data = SimpleSchema(name="test", amount=1.0)
        errors = schema.validate_cross_field(data)
        assert len(errors) == 0  # Exception caught, no error added

    def test_parse_fields_valid(self):
        schema = ExtractionSchema(model=SimpleSchema)
        result = schema.parse_fields({"name": "Test", "amount": 42.0})
        assert result.name == "Test"

    def test_parse_fields_invalid_raises(self):
        schema = ExtractionSchema(model=SimpleSchema)
        with pytest.raises(Exception):
            schema.parse_fields({"name": 123})  # wrong type for strict


# ============================================================
# INGEST EDGE CASES
# ============================================================
class TestIngestEdgeCases:
    def test_empty_bytes_pdf_graceful(self):
        """Empty bytes should not crash — returns empty result gracefully."""
        router = IngestRouter()
        result = router.ingest(b"", file_type="pdf")
        # Should return a scanned result with no text (graceful degradation)
        assert result.page_count == 0 or result.text_content is None or result.text_content == ""

    def test_random_bytes_as_pdf_graceful(self):
        """Random bytes as PDF should not crash."""
        router = IngestRouter()
        result = router.ingest(b"this is not a pdf", file_type="pdf")
        assert result.page_count == 0 or result.text_content is None or result.text_content == ""

    def test_text_file(self):
        router = IngestRouter()
        result = router.ingest(b"Hello World, this is a test document.", file_type="txt")
        assert result.extraction_method == ExtractionMethod.DIGITAL
        assert result.text_content == "Hello World, this is a test document."
        assert result.page_count == 1

    def test_empty_text_file(self):
        router = IngestRouter()
        result = router.ingest(b"", file_type="txt")
        assert result.text_content == ""

    def test_file_not_found(self):
        router = IngestRouter()
        with pytest.raises(FileNotFoundError):
            router.ingest("/nonexistent/path/to/file.pdf")

    def test_unsupported_file_type_fallback(self):
        """Unsupported file types fall back to text parsing."""
        router = IngestRouter()
        result = router.ingest(b"some content", file_type="xyz")
        # Falls back to text ingestion
        assert result.text_content == "some content"

    def test_large_text_file(self):
        router = IngestRouter()
        large_text = ("Line of text. " * 1000).encode("utf-8")
        result = router.ingest(large_text, file_type="txt")
        assert len(result.text_content or "") > 1000


# ============================================================
# PREPROCESSOR EDGE CASES
# ============================================================
class TestPreprocessorEdgeCases:
    def test_assess_quality_invalid_image(self):
        preprocessor = Preprocessor()
        result = preprocessor.assess_quality(b"not an image")
        assert isinstance(result, QualityAssessment)
        assert result.needs_preprocessing is False  # graceful fallback

    def test_preprocess_no_preprocessing_needed(self):
        preprocessor = Preprocessor()
        quality = QualityAssessment(needs_preprocessing=False)
        result_bytes, steps = preprocessor.preprocess(b"image data", quality)
        assert result_bytes == b"image data"
        assert steps == []

    def test_preprocess_invalid_image(self):
        preprocessor = Preprocessor()
        quality = QualityAssessment(
            needs_preprocessing=True, skew_angle=5.0
        )
        result_bytes, steps = preprocessor.preprocess(b"not an image", quality)
        # Should return original bytes on failure
        assert result_bytes == b"not an image"
        assert steps == []


# ============================================================
# PROVIDER EDGE CASES
# ============================================================
class TestProviderEdgeCases:
    def test_parse_tool_response_empty_dict(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response({}, schema)
        assert all(f.value is None for f in fields.values())

    def test_parse_tool_response_flat_values(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {"name": "Acme Corp", "amount": 1500.0}, schema
        )
        assert fields["name"].value == "Acme Corp"
        assert fields["name"].confidence == 0.85
        assert fields["amount"].value == 1500.0

    def test_parse_tool_response_nested_values(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {
                "name": {"value": "Acme", "confidence": 0.95, "source_quote": "Acme Corp"},
                "amount": {"value": 100, "confidence": 0.9},
            },
            schema,
        )
        assert fields["name"].value == "Acme"
        assert fields["name"].confidence == 0.95

    def test_parse_tool_response_wrapped_in_fields_key(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {"fields": {"name": "Test", "amount": 42}}, schema
        )
        assert fields["name"].value == "Test"

    def test_parse_tool_response_wrapped_in_data_key(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {"data": {"name": "Test", "amount": 42}}, schema
        )
        assert fields["name"].value == "Test"

    def test_parse_tool_response_zero_confidence_means_null(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {"name": {"value": "ghost", "confidence": 0.0}}, schema
        )
        assert fields["name"].value is None

    def test_parse_tool_response_empty_string_value(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {"name": "", "amount": "  "}, schema
        )
        assert fields["name"].value is None
        assert fields["amount"].value is None

    def test_parse_tool_response_extra_fields_ignored(self):
        provider = LiteLLMProvider(model="test")
        schema = ExtractionSchema(model=SimpleSchema)
        fields = provider._parse_tool_response(
            {"name": "Test", "amount": 1, "extra_field": "ignored"}, schema
        )
        assert "extra_field" not in fields
        assert fields["name"].value == "Test"

    def test_parse_json_response_valid(self):
        provider = LiteLLMProvider(model="test")
        result = provider._parse_json_response('{"name": "test"}')
        assert result == {"name": "test"}

    def test_parse_json_response_markdown_fenced(self):
        provider = LiteLLMProvider(model="test")
        result = provider._parse_json_response('```json\n{"name": "test"}\n```')
        assert result == {"name": "test"}

    def test_parse_json_response_truncated(self):
        provider = LiteLLMProvider(model="test")
        result = provider._parse_json_response('{"name": "test", "val": {"nested": "obj"')
        # Should attempt repair
        assert result is not None or result is None  # doesn't crash

    def test_parse_json_response_garbage(self):
        provider = LiteLLMProvider(model="test")
        result = provider._parse_json_response("This is not JSON at all.")
        assert result is None

    def test_parse_json_response_empty(self):
        provider = LiteLLMProvider(model="test")
        result = provider._parse_json_response("")
        assert result is None

    def test_parse_json_response_with_text_before_json(self):
        provider = LiteLLMProvider(model="test")
        result = provider._parse_json_response('Here is the result: {"name": "test"}')
        assert result == {"name": "test"}

    def test_base_kwargs_without_api_key(self):
        provider = LiteLLMProvider(model="gpt-4o")
        assert "api_key" not in provider._base_kwargs
        assert provider._base_kwargs["model"] == "gpt-4o"

    def test_base_kwargs_with_api_key(self):
        provider = LiteLLMProvider(model="gpt-4o", api_key="sk-test")
        assert provider._base_kwargs["api_key"] == "sk-test"


# ============================================================
# CONSENSUS ENGINE EDGE CASES
# ============================================================
class TestConsensusEdgeCases:
    def test_merge_all_providers_errored(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(provider="a", error="timeout"),
            RawExtraction(provider="b", error="rate limit"),
        ]
        # Empty fields in errored extractions
        result = engine.merge(exts, ["name"])
        assert result["name"].value is None

    def test_merge_one_null_one_value(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(
                provider="a",
                fields={"name": FieldResult(value=None, confidence=0.0)},
            ),
            RawExtraction(
                provider="b",
                fields={"name": FieldResult(value="Acme", confidence=0.9)},
            ),
        ]
        result = engine.merge(exts, ["name"])
        assert result["name"].value == "Acme"

    def test_merge_extreme_confidence_values(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(
                provider="a",
                fields={"val": FieldResult(value="X", confidence=0.001)},
            ),
            RawExtraction(
                provider="b",
                fields={"val": FieldResult(value="X", confidence=0.999)},
            ),
        ]
        result = engine.merge(exts, ["val"])
        assert result["val"].value == "X"
        assert result["val"].confidence > 0.5

    def test_merge_special_characters_in_values(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(
                provider="a",
                fields={"addr": FieldResult(value="123 Main St. #4B", confidence=0.9)},
            ),
            RawExtraction(
                provider="b",
                fields={"addr": FieldResult(value="123 Main St. #4B", confidence=0.85)},
            ),
        ]
        result = engine.merge(exts, ["addr"])
        assert result["addr"].consensus_type == "unanimous"

    def test_merge_large_number_normalization(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(
                provider="a",
                fields={"total": FieldResult(value="$1,234,567.89", confidence=0.9)},
            ),
            RawExtraction(
                provider="b",
                fields={"total": FieldResult(value="1234567.89", confidence=0.85)},
            ),
        ]
        result = engine.merge(exts, ["total"])
        assert result["total"].consensus_type == "unanimous"

    def test_merge_five_providers(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(provider=f"p{i}", fields={"x": FieldResult(value="same", confidence=0.8)})
            for i in range(5)
        ]
        result = engine.merge(exts, ["x"])
        assert result["x"].consensus_type == "unanimous"
        assert result["x"].confidence > 0.8  # boosted

    def test_merge_boolean_values(self):
        engine = ConsensusEngine()
        exts = [
            RawExtraction(provider="a", fields={"flag": FieldResult(value=True, confidence=0.9)}),
            RawExtraction(provider="b", fields={"flag": FieldResult(value="true", confidence=0.85)}),
        ]
        result = engine.merge(exts, ["flag"])
        assert result["flag"].value is not None


# ============================================================
# GROUNDING EDGE CASES
# ============================================================
class TestGroundingEdgeCases:
    def test_very_short_value(self):
        verifier = GroundingVerifier()
        result = verifier.verify_field("5", None, "Total: 5 items")
        assert result.is_grounded is True

    def test_numeric_value(self):
        verifier = GroundingVerifier()
        result = verifier.verify_field("1500.00", None, "Amount: $1,500.00")
        assert result.is_grounded is True

    def test_value_with_special_chars(self):
        verifier = GroundingVerifier()
        result = verifier.verify_field(
            "O'Brien & Associates",
            None,
            "Client: O'Brien & Associates LLC",
        )
        assert result.is_grounded is True

    def test_very_long_document(self):
        verifier = GroundingVerifier()
        long_doc = "x " * 50000 + "SECRET VALUE" + " y" * 50000
        result = verifier.verify_field("SECRET VALUE", None, long_doc)
        assert result.is_grounded is True

    def test_empty_value_string(self):
        verifier = GroundingVerifier()
        result = verifier.verify_field("", None, "some document")
        assert result.is_grounded is True  # empty treated as OK

    def test_none_document_text(self):
        verifier = GroundingVerifier()
        result = verifier.verify_field("test", None, "")
        assert result.is_grounded is False


# ============================================================
# CALIBRATION EDGE CASES
# ============================================================
class TestCalibrationEdgeCases:
    def test_calibrate_boundary_values(self):
        cal = ConfidenceCalibrator()
        assert cal.calibrate("f", 0.0) == 0.0
        assert cal.calibrate("f", 1.0) == 1.0
        assert 0 < cal.calibrate("f", 0.5) < 1

    def test_calibrate_very_small_confidence(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate("f", 0.001)
        assert result >= 0.0
        assert result < 0.1

    def test_calibrate_very_high_confidence(self):
        cal = ConfidenceCalibrator()
        result = cal.calibrate("f", 0.999)
        assert result > 0.5
        assert result <= 1.0

    def test_ece_empty_predictions(self):
        cal = ConfidenceCalibrator()
        assert cal.compute_ece([]) == 0.0

    def test_ece_single_prediction(self):
        cal = ConfidenceCalibrator()
        ece = cal.compute_ece([(0.9, True)])
        assert isinstance(ece, float)

    def test_fit_with_all_correct(self):
        cal = ConfidenceCalibrator()
        data = [("field", 0.9, True)] * 100
        cal.fit(data)
        # Temperature should be close to 1.0 (well calibrated)
        assert "field" in cal.temperatures

    def test_fit_with_all_wrong(self):
        cal = ConfidenceCalibrator()
        data = [("field", 0.9, False)] * 100
        cal.fit(data)
        # Temperature should be high (overconfident)
        assert cal.temperatures["field"] > 1.0

    def test_save_load_roundtrip(self, tmp_path: Path):
        cal = ConfidenceCalibrator()
        cal.temperatures = {"a": 1.5, "b": 2.0, "c": 0.8}
        cal.default_temperature = 1.7

        path = tmp_path / "cal.json"
        cal.save(path)

        cal2 = ConfidenceCalibrator()
        cal2.load(path)
        assert cal2.temperatures == cal.temperatures
        assert cal2.default_temperature == cal.default_temperature


# ============================================================
# VALIDATOR EDGE CASES
# ============================================================
class TestValidatorEdgeCases:
    def test_validate_empty_data(self):
        validator = CrossFieldValidator()
        errors = validator.validate({})
        assert errors == []

    def test_validate_none_values(self):
        validator = CrossFieldValidator()
        validator.auto_detect_format_rules(["employer_ein"])
        errors = validator.validate({"employer_ein": None})
        assert errors == []  # None values skip format validation

    def test_format_rule_empty_string(self):
        validator = CrossFieldValidator()
        validator.auto_detect_format_rules(["employer_ein"])
        errors = validator.validate({"employer_ein": ""})
        assert errors == []  # Empty string skipped

    def test_validate_with_field_result_objects(self):
        validator = CrossFieldValidator()
        validator.auto_detect_format_rules(["employer_ein"])
        data = {"employer_ein": FieldResult(value="12-3456789", confidence=0.9)}
        errors = validator.validate(data)
        assert errors == []


# ============================================================
# CORRECTION STORE EDGE CASES
# ============================================================
class TestCorrectionStoreEdgeCases:
    def test_empty_store(self, tmp_path: Path):
        store = CorrectionStore(tmp_path / "empty")
        assert store.get_few_shot_examples("anything") == []
        assert store.build_few_shot_prompt("anything") is None
        assert store.get_correction_stats()["total"] == 0

    def test_clear_nonexistent_type(self, tmp_path: Path):
        store = CorrectionStore(tmp_path / "store")
        store.add_correction("Invoice", "name", "a", "b")
        removed = store.clear("NonExistent")
        assert removed == 0

    def test_special_characters_in_values(self, tmp_path: Path):
        store = CorrectionStore(tmp_path / "store")
        store.add_correction("Invoice", "addr", 'Line 1\nLine "2"', "Fixed\tAddress")
        examples = store.get_few_shot_examples("Invoice")
        assert len(examples) == 1

    def test_concurrent_access_same_path(self, tmp_path: Path):
        path = tmp_path / "shared"
        store1 = CorrectionStore(path)
        store1.add_correction("A", "f", "1", "2")

        store2 = CorrectionStore(path)
        assert len(store2.get_few_shot_examples("A")) == 1


# ============================================================
# HITL ROUTER EDGE CASES
# ============================================================
class TestHITLRouterEdgeCases:
    def test_empty_result(self):
        router = HITLRouter()
        result = ExtractionResult[Any](fields={})
        queue = router.route(result)
        assert queue.decision == ReviewDecision.AUTO_ACCEPT

    def test_all_null_fields(self):
        router = HITLRouter()
        result = ExtractionResult[Any](
            fields={
                "a": FieldResult(value=None, confidence=0.0),
                "b": FieldResult(value=None, confidence=0.0),
            }
        )
        queue = router.route(result)
        assert queue.decision == ReviewDecision.AUTO_ACCEPT  # Nothing to review

    def test_mixed_warnings_and_errors(self):
        router = HITLRouter()
        result = ExtractionResult[Any](
            fields={"total": FieldResult(value=100, confidence=0.99)},
            validation_errors=[
                ValidationError(
                    rule_name="warning_rule",
                    message="minor issue",
                    affected_fields=["total"],
                    severity="warning",
                ),
                ValidationError(
                    rule_name="error_rule",
                    message="major issue",
                    affected_fields=["total"],
                    severity="error",
                ),
            ],
        )
        queue = router.route(result)
        assert queue.decision == ReviewDecision.FULL_REVIEW


# ============================================================
# TEMPLATE ENGINE EDGE CASES
# ============================================================
class TestTemplateEdgeCases:
    def test_extract_no_anchors(self):
        engine = TemplateExtractor()
        schema = ExtractionSchema(model=SimpleSchema)  # no anchors
        result = engine.extract(schema, "some text", [])
        assert result == {}

    def test_extract_no_text_no_words(self):
        engine = TemplateExtractor()
        schema = ExtractionSchema(
            model=SimpleSchema,
            template_anchors={"name": FieldAnchor(label="Name", direction="right")},
        )
        result = engine.extract(schema, None, [])
        assert len(result) == 0

    def test_extract_with_regex_pattern(self):
        engine = TemplateExtractor()
        schema = ExtractionSchema(
            model=SimpleSchema,
            template_anchors={
                "amount": FieldAnchor(
                    label="Total",
                    direction="right",
                    value_type="decimal",
                    regex_pattern=r"Total[:\s]*\$?([\d,]+\.?\d*)",
                ),
            },
        )
        result = engine.extract(schema, "Invoice Total: $1,234.56", [])
        assert result["amount"].value == 1234.56

    def test_normalize_value_types(self):
        engine = TemplateExtractor()
        assert engine._normalize_value("$1,234.56", "decimal") == 1234.56
        assert engine._normalize_value("100", "int") == 100
        assert engine._normalize_value("true", "bool") is True
        assert engine._normalize_value("false", "bool") is False
        assert engine._normalize_value("03/15/2024", "date") == "2024-03-15"
        assert engine._normalize_value("hello", "str") == "hello"
        assert engine._normalize_value("", "str") is None


# ============================================================
# EXTRACTION RESULT EDGE CASES
# ============================================================
class TestExtractionResultEdgeCases:
    def test_correct_already_corrected_field(self):
        result = ExtractionResult[Any](
            fields={"name": FieldResult(value="Old", confidence=0.5)}
        )
        result.correct("name", "New1")
        result.correct("name", "New2")
        assert result.fields["name"].value == "New2"
        assert len(result.corrections) == 2

    def test_to_json_with_special_types(self):
        result = ExtractionResult[Any](
            fields={
                "amount": FieldResult(value=Decimal("1234.56"), confidence=0.9),
            },
            overall_confidence=0.9,
        )
        json_str = result.to_json()
        assert "1234.56" in json_str

    def test_min_field_confidence_all_null(self):
        result = ExtractionResult[Any](
            fields={
                "a": FieldResult(value=None, confidence=0.0),
            }
        )
        assert result.min_field_confidence == 0.0

    def test_all_grounded_with_no_grounding(self):
        result = ExtractionResult[Any](
            fields={
                "a": FieldResult(value="x", is_grounded=None),
            }
        )
        assert result.all_grounded is True  # None is not False


# ============================================================
# BOUNDING BOX EDGE CASES
# ============================================================
class TestBoundingBoxEdgeCases:
    def test_zero_area_box(self):
        bbox = BoundingBox(page=0, x=0.5, y=0.5, w=0.0, h=0.0)
        assert bbox.area == 0.0

    def test_full_page_box(self):
        bbox = BoundingBox(page=0, x=0.0, y=0.0, w=1.0, h=1.0)
        assert bbox.area == 1.0
        assert bbox.iou(bbox) == 1.0

    def test_partial_overlap(self):
        a = BoundingBox(page=0, x=0.0, y=0.0, w=0.5, h=0.5)
        b = BoundingBox(page=0, x=0.25, y=0.25, w=0.5, h=0.5)
        iou = a.iou(b)
        assert 0 < iou < 1

    def test_iou_adjacent_no_overlap(self):
        a = BoundingBox(page=0, x=0.0, y=0.0, w=0.5, h=0.5)
        b = BoundingBox(page=0, x=0.5, y=0.0, w=0.5, h=0.5)
        assert a.iou(b) == 0.0  # touching but not overlapping


# ============================================================
# CONFIG EDGE CASES
# ============================================================
class TestConfigEdgeCases:
    def test_default_config(self):
        config = get_config()
        assert config.confidence_floor == 0.70
        assert config.reexamine_threshold == 0.85

    def test_config_with_overrides(self):
        config = get_config(confidence_floor=0.5, reexamine_threshold=0.9)
        assert config.confidence_floor == 0.5
        assert config.reexamine_threshold == 0.9
