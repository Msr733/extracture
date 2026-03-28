"""Schema introspection and management for extracture."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel
from pydantic.fields import FieldInfo


@dataclass
class FieldAnchor:
    """Spatial anchor for template-based extraction."""

    label: str
    direction: str = "right"  # "right", "below", "right_and_below"
    value_type: str = "str"  # "str", "decimal", "int", "date", "bool"
    aliases: list[str] = field(default_factory=list)
    regex_pattern: str | None = None
    max_distance_ratio: float = 0.15  # max distance from anchor as ratio of page dimension


@dataclass
class CrossFieldRule:
    """A cross-field validation rule."""

    name: str
    fields: list[str]
    check: Callable[..., str | None]
    severity: str = "error"

    def validate(self, data: BaseModel) -> str | None:
        return self.check(data)


class ExtractionSchema:
    """Wraps a Pydantic model to provide extraction metadata."""

    def __init__(
        self,
        model: type[BaseModel],
        *,
        form_title: str | None = None,
        form_instructions: str = "",
        field_labels: dict[str, str] | None = None,
        field_sections: dict[str, list[str]] | None = None,
        validation_rules: list[CrossFieldRule] | None = None,
        template_anchors: dict[str, FieldAnchor] | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.form_title = form_title or model.__name__
        self.form_instructions = form_instructions
        self._field_labels = field_labels or {}
        self.field_sections = field_sections or {}
        self.validation_rules = validation_rules or []
        self.template_anchors = template_anchors or {}
        self.max_tokens = max_tokens

    @property
    def field_names(self) -> list[str]:
        return list(self.model.model_fields.keys())

    @property
    def field_types(self) -> dict[str, Any]:
        hints = get_type_hints(self.model)
        return {name: hints.get(name, Any) for name in self.model.model_fields}

    @property
    def field_labels(self) -> dict[str, str]:
        labels = {}
        for name, field_info in self.model.model_fields.items():
            if name in self._field_labels:
                labels[name] = self._field_labels[name]
            elif field_info.description:
                labels[name] = field_info.description
            else:
                labels[name] = name.replace("_", " ").title()
        return labels

    @property
    def has_template(self) -> bool:
        return len(self.template_anchors) > 0

    def get_json_schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()

    def get_field_info(self, field_name: str) -> FieldInfo | None:
        return self.model.model_fields.get(field_name)

    def validate_cross_field(self, data: BaseModel) -> list[str]:
        errors = []
        for rule in self.validation_rules:
            result = rule.validate(data)
            if result:
                errors.append(result)
        return errors

    def parse_fields(self, raw: dict[str, Any]) -> BaseModel:
        return self.model.model_validate(raw)

    def build_tool_schema(self) -> dict[str, Any]:
        """Build a tool/function calling schema for LLM providers."""
        json_schema = self.get_json_schema()

        extraction_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for field_name in self.field_names:
            label = self.field_labels.get(field_name, field_name)
            field_schema = self._get_field_schema(field_name, json_schema)
            field_schema["description"] = label
            extraction_schema["properties"][field_name] = {
                "type": "object",
                "properties": {
                    "value": field_schema,
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Confidence score 0.0-1.0. Use 0.0 if the field is not visible.",
                    },
                    "source_quote": {
                        "type": ["string", "null"],
                        "description": "Exact text from the document that contains this value.",
                    },
                },
                "required": ["value", "confidence"],
            }

        return {
            "name": f"extract_{self.model.__name__.lower()}",
            "description": f"Extract structured {self.form_title} data from the document.",
            "input_schema": extraction_schema,
        }

    def _get_field_schema(self, field_name: str, full_schema: dict[str, Any]) -> dict[str, Any]:
        props = full_schema.get("properties", {})
        if field_name in props:
            schema = dict(props[field_name])
            schema.pop("title", None)
            return schema
        return {"type": "string"}

    def build_extraction_prompt(self, text_content: str | None = None, mode: str = "text") -> str:
        """Build a prompt for LLM extraction."""
        lines = []

        if mode == "vision":
            lines.append(f"Look at this document image. This is a {self.form_title}.")
        else:
            lines.append(f"Extract structured data from this {self.form_title} document.")

        if self.form_instructions:
            lines.append(f"\nImportant: {self.form_instructions}")

        lines.append("\nExtract the following fields:")
        for name in self.field_names:
            label = self.field_labels.get(name, name)
            lines.append(f"  - {name}: {label}")

        lines.append("\nFor each field, provide:")
        lines.append("  - value: the extracted value (null if not found)")
        lines.append("  - confidence: 0.0 to 1.0 (0.0 = not visible in document)")
        lines.append("  - source_quote: the exact text from the document containing this value")

        lines.append("\nRules:")
        lines.append("  - For monetary values: return as a number without $ or commas (e.g., 1234.56)")
        lines.append("  - For dates: return in YYYY-MM-DD format when possible")
        lines.append("  - For SSN/EIN: preserve the format with dashes (e.g., 123-45-6789)")
        lines.append("  - For checkboxes: return true/false")
        lines.append("  - If a field is not visible or not applicable, set value to null and confidence to 0.0")
        lines.append("  - Do NOT guess or fabricate values. Only extract what you can see.")

        if text_content:
            lines.append(f"\n--- DOCUMENT TEXT ---\n{text_content}\n--- END DOCUMENT ---")

        lines.append("\nRespond with a JSON object containing the extracted fields.")

        return "\n".join(lines)

    def build_reexamination_prompt(
        self,
        low_confidence_fields: dict[str, Any],
        text_content: str | None = None,
    ) -> str:
        """Build a focused re-examination prompt for low-confidence fields."""
        lines = [
            f"Look very carefully at this {self.form_title}.",
            "I need you to verify these specific fields that had low confidence in initial extraction:",
            "",
        ]

        for field_name, current in low_confidence_fields.items():
            label = self.field_labels.get(field_name, field_name)
            val = current.get("value", "null")
            conf = current.get("confidence", 0)
            lines.append(f"  - {field_name} ({label}): current_value={val}, confidence={conf:.2f}")

        lines.append("")
        lines.append("For each field, look again carefully and provide:")
        lines.append("  - value: your best reading of the field")
        lines.append("  - confidence: your confidence 0.0-1.0")
        lines.append("  - source_quote: exact text from document containing this value")
        lines.append("")
        lines.append("Only update fields you can see more clearly now. Do NOT guess.")

        if text_content:
            lines.append(f"\n--- DOCUMENT TEXT ---\n{text_content}\n--- END DOCUMENT ---")

        lines.append("\nRespond with JSON containing only the re-examined fields.")

        return "\n".join(lines)
