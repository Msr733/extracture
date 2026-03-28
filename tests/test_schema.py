"""Tests for schema introspection and prompt building."""

from decimal import Decimal

from pydantic import BaseModel, Field

from extracture.schema import CrossFieldRule, ExtractionSchema, FieldAnchor


class Invoice(BaseModel):
    vendor_name: str = Field(description="Company name of the vendor")
    invoice_number: str = Field(description="Invoice number")
    total: Decimal = Field(description="Total amount due")
    tax: Decimal = Field(description="Tax amount")


class TestExtractionSchema:
    def setup_method(self):
        self.schema = ExtractionSchema(
            model=Invoice,
            form_title="Invoice",
            form_instructions="Read only the first page.",
        )

    def test_field_names(self):
        assert set(self.schema.field_names) == {
            "vendor_name", "invoice_number", "total", "tax"
        }

    def test_field_labels_from_description(self):
        labels = self.schema.field_labels
        assert labels["vendor_name"] == "Company name of the vendor"
        assert labels["total"] == "Total amount due"

    def test_field_labels_override(self):
        schema = ExtractionSchema(
            model=Invoice,
            field_labels={"vendor_name": "Custom Label"},
        )
        assert schema.field_labels["vendor_name"] == "Custom Label"
        assert schema.field_labels["total"] == "Total amount due"  # fallback

    def test_has_template_false(self):
        assert self.schema.has_template is False

    def test_has_template_true(self):
        schema = ExtractionSchema(
            model=Invoice,
            template_anchors={"vendor_name": FieldAnchor(label="Vendor")},
        )
        assert schema.has_template is True

    def test_json_schema(self):
        js = self.schema.get_json_schema()
        assert "properties" in js
        assert "vendor_name" in js["properties"]

    def test_build_tool_schema(self):
        tool = self.schema.build_tool_schema()
        assert tool["name"] == "extract_invoice"
        assert "input_schema" in tool
        props = tool["input_schema"]["properties"]
        assert "vendor_name" in props
        assert "confidence" in props["vendor_name"]["properties"]

    def test_build_extraction_prompt_text(self):
        prompt = self.schema.build_extraction_prompt(text_content="Hello world", mode="text")
        assert "Invoice" in prompt
        assert "vendor_name" in prompt
        assert "Hello world" in prompt
        assert "Read only the first page" in prompt

    def test_build_extraction_prompt_vision(self):
        prompt = self.schema.build_extraction_prompt(mode="vision")
        assert "Look at this document image" in prompt
        assert "vendor_name" in prompt

    def test_build_reexamination_prompt(self):
        prompt = self.schema.build_reexamination_prompt(
            low_confidence_fields={
                "vendor_name": {"value": "Acme", "confidence": 0.5},
            }
        )
        assert "vendor_name" in prompt
        assert "0.50" in prompt
        assert "Acme" in prompt

    def test_validate_cross_field(self):
        schema = ExtractionSchema(
            model=Invoice,
            validation_rules=[
                CrossFieldRule(
                    name="total_check",
                    fields=["total", "tax"],
                    check=lambda data: "bad" if data.total < data.tax else None,
                )
            ],
        )
        data = Invoice(vendor_name="X", invoice_number="1", total=Decimal("5"), tax=Decimal("10"))
        errors = schema.validate_cross_field(data)
        assert len(errors) == 1
        assert errors[0] == "bad"

    def test_parse_fields(self):
        result = self.schema.parse_fields({
            "vendor_name": "Acme",
            "invoice_number": "INV-001",
            "total": "150.00",
            "tax": "15.00",
        })
        assert isinstance(result, Invoice)
        assert result.vendor_name == "Acme"
        assert result.total == Decimal("150.00")
