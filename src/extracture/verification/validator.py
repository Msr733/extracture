"""Cross-field validation and domain-specific rules."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from pydantic import BaseModel

from extracture.models import ValidationError as ExtractureValidationError

logger = logging.getLogger(__name__)

# Common format validators
EIN_PATTERN = re.compile(r"^\d{2}-\d{7}$")
SSN_PATTERN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
US_STATE_PATTERN = re.compile(r"^[A-Z]{2}$")
US_ZIP_PATTERN = re.compile(r"^\d{5}(-\d{4})?$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_PATTERN = re.compile(r"^[\d\-\(\)\s\+\.]{7,20}$")


class CrossFieldValidator:
    """Validates extracted data using cross-field rules and format checks."""

    def __init__(self):
        self.rules: list[tuple[str, list[str], Callable, str]] = []
        self.format_rules: dict[str, re.Pattern] = {}

    def add_rule(
        self,
        name: str,
        fields: list[str],
        check: Callable[..., str | None],
        severity: str = "error",
    ) -> None:
        """Add a cross-field validation rule."""
        self.rules.append((name, fields, check, severity))

    def add_format_rule(self, field_name: str, pattern: re.Pattern) -> None:
        """Add a format validation rule for a specific field."""
        self.format_rules[field_name] = pattern

    def validate(
        self,
        data: dict[str, Any] | BaseModel,
        schema_fields: list[str] | None = None,
    ) -> list[ExtractureValidationError]:
        """Run all validation rules against extracted data."""
        errors: list[ExtractureValidationError] = []

        # Get values dict
        if isinstance(data, BaseModel):
            values = data.model_dump()
        elif isinstance(data, dict):
            values = {}
            for k, v in data.items():
                if hasattr(v, "value"):
                    values[k] = v.value
                else:
                    values[k] = v
        else:
            return errors

        # Run cross-field rules
        for name, fields, check, severity in self.rules:
            try:
                # Create a simple namespace for the check function
                result = check(type("Fields", (), values)())
                if result:
                    errors.append(
                        ExtractureValidationError(
                            rule_name=name,
                            message=result,
                            affected_fields=fields,
                            severity=severity,
                        )
                    )
            except Exception as e:
                logger.debug(f"Validation rule '{name}' failed with error: {e}")

        # Run format rules
        for field_name, pattern in self.format_rules.items():
            value = values.get(field_name)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str and not pattern.match(value_str):
                errors.append(
                    ExtractureValidationError(
                        rule_name=f"format_{field_name}",
                        message=f"'{field_name}' value '{value_str}' doesn't match expected format",
                        affected_fields=[field_name],
                        severity="warning",
                    )
                )

        return errors

    def auto_detect_format_rules(self, field_names: list[str]) -> None:
        """Auto-detect format rules based on field name patterns."""
        for name in field_names:
            name_lower = name.lower()

            if "ein" in name_lower and "employer" in name_lower:
                self.format_rules[name] = EIN_PATTERN
            elif "ssn" in name_lower or "social_security" in name_lower:
                self.format_rules[name] = SSN_PATTERN
            elif name_lower.endswith("_state") or name_lower == "state":
                self.format_rules[name] = US_STATE_PATTERN
            elif "zip" in name_lower or "postal" in name_lower:
                self.format_rules[name] = US_ZIP_PATTERN
            elif "email" in name_lower:
                self.format_rules[name] = EMAIL_PATTERN
            elif "phone" in name_lower or "tel" in name_lower:
                self.format_rules[name] = PHONE_PATTERN


# Pre-built validation rule sets
def sum_equals_rule(total_field: str, *addend_fields: str):
    """Create a rule that checks if fields sum to a total."""

    def check(fields):
        total = getattr(fields, total_field, None)
        if total is None:
            return None

        addend_sum = 0
        all_none = True
        for f in addend_fields:
            val = getattr(fields, f, None)
            if val is not None:
                try:
                    addend_sum += float(val)
                    all_none = False
                except (ValueError, TypeError):
                    pass

        if all_none:
            return None

        try:
            total_float = float(total)
        except (ValueError, TypeError):
            return None

        if abs(total_float - addend_sum) > 0.01:
            return (
                f"{total_field} ({total_float}) doesn't equal sum of "
                f"{', '.join(addend_fields)} ({addend_sum})"
            )
        return None

    return (
        f"sum_check_{total_field}",
        [total_field, *addend_fields],
        check,
        "warning",
    )


def date_not_future_rule(date_field: str):
    """Create a rule that checks if a date is not in the future."""
    from datetime import date

    def check(fields):
        value = getattr(fields, date_field, None)
        if value is None:
            return None
        try:
            if isinstance(value, str) and ISO_DATE_PATTERN.match(value):
                parts = value.split("-")
                d = date(int(parts[0]), int(parts[1]), int(parts[2]))
                if d > date.today():
                    return f"{date_field} ({value}) is in the future"
        except (ValueError, TypeError):
            pass
        return None

    return (f"future_date_{date_field}", [date_field], check, "warning")


def required_fields_rule(*field_names: str):
    """Create a rule that checks required fields are present."""

    def check(fields):
        missing = []
        for name in field_names:
            val = getattr(fields, name, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(name)
        if missing:
            return f"Required fields missing: {', '.join(missing)}"
        return None

    return ("required_fields", list(field_names), check, "error")
