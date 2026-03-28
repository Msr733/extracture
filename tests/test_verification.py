"""Tests for verification layer — grounding, calibration, validation."""

import math

import pytest

from extracture.verification.calibration import ConfidenceCalibrator
from extracture.verification.grounding import GroundingVerifier
from extracture.verification.validator import (
    CrossFieldValidator,
    date_not_future_rule,
    required_fields_rule,
    sum_equals_rule,
)


class TestGroundingVerifier:
    def setup_method(self):
        self.verifier = GroundingVerifier()

    def test_exact_match(self):
        result = self.verifier.verify_field(
            "Acme Corporation",
            None,
            "Invoice from Acme Corporation for services rendered.",
        )
        assert result.is_grounded is True
        assert result.method == "exact"

    def test_exact_match_case_insensitive(self):
        result = self.verifier.verify_field(
            "ACME CORP",
            None,
            "Invoice from acme corp for services.",
        )
        assert result.is_grounded is True

    def test_normalized_match(self):
        result = self.verifier.verify_field(
            "123-45-6789",
            None,
            "SSN: 123 45 6789",
        )
        assert result.is_grounded is True

    def test_fuzzy_match(self):
        result = self.verifier.verify_field(
            "Acme Corporaton",  # typo
            None,
            "Invoice from Acme Corporation for services rendered.",
        )
        assert result.is_grounded is True
        assert result.method.startswith("fuzzy")

    def test_ungrounded(self):
        result = self.verifier.verify_field(
            "Completely Fabricated Name",
            None,
            "This document mentions nothing about that company.",
        )
        assert result.is_grounded is False

    def test_null_value_is_grounded(self):
        result = self.verifier.verify_field(None, None, "Any document text.")
        assert result.is_grounded is True

    def test_empty_document(self):
        result = self.verifier.verify_field("test", None, "")
        assert result.is_grounded is False

    def test_quote_verification(self):
        result = self.verifier.verify_field(
            "$1,500.00",
            "Total Amount: $1,500.00",
            "Items totaling... Total Amount: $1,500.00 is due.",
        )
        assert result.is_grounded is True


class TestConfidenceCalibrator:
    def setup_method(self):
        self.cal = ConfidenceCalibrator()

    def test_default_temperature_reduces_overconfidence(self):
        # T=1.5 should reduce confidence
        raw = 0.95
        calibrated = self.cal.calibrate("test", raw)
        assert calibrated < raw

    def test_zero_confidence(self):
        assert self.cal.calibrate("test", 0.0) == 0.0

    def test_one_confidence(self):
        assert self.cal.calibrate("test", 1.0) == 1.0

    def test_per_field_temperature(self):
        self.cal.temperatures["field_a"] = 2.0
        self.cal.temperatures["field_b"] = 1.0

        cal_a = self.cal.calibrate("field_a", 0.9)
        cal_b = self.cal.calibrate("field_b", 0.9)

        assert cal_a < cal_b  # Higher T = more conservative

    def test_ece_perfect_calibration(self):
        # If confidence perfectly matches accuracy, ECE should be ~0
        predictions = [(0.9, True)] * 9 + [(0.9, False)] * 1
        ece = self.cal.compute_ece(predictions)
        assert ece < 0.1

    def test_ece_overconfident(self):
        # High confidence but only 50% correct = high ECE
        predictions = [(0.95, True)] * 5 + [(0.95, False)] * 5
        ece = self.cal.compute_ece(predictions)
        assert ece > 0.3

    def test_save_load(self, tmp_path):
        self.cal.temperatures = {"field_a": 1.5, "field_b": 2.0}
        path = tmp_path / "cal.json"

        self.cal.save(path)
        new_cal = ConfidenceCalibrator()
        new_cal.load(path)

        assert new_cal.temperatures == {"field_a": 1.5, "field_b": 2.0}


class TestCrossFieldValidator:
    def test_sum_rule(self):
        validator = CrossFieldValidator()
        name, fields, check, severity = sum_equals_rule("total", "subtotal", "tax")
        validator.add_rule(name, fields, check, severity)

        data = {"total": 100, "subtotal": 80, "tax": 20}
        errors = validator.validate(data)
        assert len(errors) == 0

    def test_sum_rule_fails(self):
        validator = CrossFieldValidator()
        name, fields, check, severity = sum_equals_rule("total", "subtotal", "tax")
        validator.add_rule(name, fields, check, severity)

        data = {"total": 100, "subtotal": 80, "tax": 30}
        errors = validator.validate(data)
        assert len(errors) == 1
        assert "100" in errors[0].message

    def test_format_auto_detect(self):
        validator = CrossFieldValidator()
        validator.auto_detect_format_rules(["employer_ein", "employee_ssn", "email"])

        assert "employer_ein" in validator.format_rules
        assert "employee_ssn" in validator.format_rules
        assert "email" in validator.format_rules

    def test_format_validation(self):
        validator = CrossFieldValidator()
        validator.auto_detect_format_rules(["employer_ein"])

        # Valid EIN
        errors = validator.validate({"employer_ein": "12-3456789"})
        assert len(errors) == 0

        # Invalid EIN
        errors = validator.validate({"employer_ein": "invalid"})
        assert len(errors) == 1

    def test_required_fields(self):
        validator = CrossFieldValidator()
        name, fields, check, severity = required_fields_rule("name", "total")
        validator.add_rule(name, fields, check, severity)

        errors = validator.validate({"name": "Acme", "total": None})
        assert len(errors) == 1
        assert "total" in errors[0].message
