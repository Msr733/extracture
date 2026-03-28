"""Template-based extraction — 520x faster than LLM for known document types."""

from decimal import Decimal
from pydantic import BaseModel, Field

from extracture import Extractor, FieldAnchor


# W-2 Tax Form schema with template anchors
class W2Form(BaseModel):
    employer_ein: str = Field(description="Employer Identification Number (XX-XXXXXXX)")
    employer_name: str = Field(description="Employer's name")
    employee_ssn: str = Field(description="Employee's Social Security Number")
    employee_name: str = Field(description="Employee's name")
    box1_wages: Decimal = Field(description="Wages, tips, other compensation")
    box2_federal_tax: Decimal = Field(description="Federal income tax withheld")
    box3_ss_wages: Decimal = Field(description="Social security wages")
    box4_ss_tax: Decimal = Field(description="Social security tax withheld")
    box5_medicare_wages: Decimal = Field(description="Medicare wages and tips")
    box6_medicare_tax: Decimal = Field(description="Medicare tax withheld")


# Create extractor with template anchors — extraction uses spatial rules first,
# falls back to LLM only for low-confidence fields
extractor = Extractor(
    schema=W2Form,
    providers=["openai:gpt-4o"],  # Fallback only
    ocr_engine="pymupdf",

    # Template anchors define WHERE to look on the form
    template_anchors={
        "employer_ein": FieldAnchor(
            label="Employer identification number",
            direction="below",
            value_type="str",
            aliases=["EIN", "employer's identification number"],
        ),
        "employer_name": FieldAnchor(
            label="Employer's name, address, and ZIP",
            direction="below",
            value_type="str",
            aliases=["employer name"],
        ),
        "employee_ssn": FieldAnchor(
            label="Employee's social security number",
            direction="below",
            value_type="str",
            aliases=["SSN", "social security"],
        ),
        "employee_name": FieldAnchor(
            label="Employee's first name and initial",
            direction="below",
            value_type="str",
        ),
        "box1_wages": FieldAnchor(
            label="Wages, tips, other compensation",
            direction="right_and_below",
            value_type="decimal",
            aliases=["wages tips other compensation", "box 1"],
            regex_pattern=r"(?:Wages.*?compensation|Box\s*1)[:\s]*\$?([\d,]+\.?\d*)",
        ),
        "box2_federal_tax": FieldAnchor(
            label="Federal income tax withheld",
            direction="right_and_below",
            value_type="decimal",
            aliases=["federal income tax", "box 2"],
            regex_pattern=r"(?:Federal income tax withheld|Box\s*2)[:\s]*\$?([\d,]+\.?\d*)",
        ),
        "box3_ss_wages": FieldAnchor(
            label="Social security wages",
            direction="right_and_below",
            value_type="decimal",
            aliases=["social security wages", "box 3"],
        ),
        "box4_ss_tax": FieldAnchor(
            label="Social security tax withheld",
            direction="right_and_below",
            value_type="decimal",
            aliases=["social security tax", "box 4"],
        ),
        "box5_medicare_wages": FieldAnchor(
            label="Medicare wages and tips",
            direction="right_and_below",
            value_type="decimal",
            aliases=["medicare wages", "box 5"],
        ),
        "box6_medicare_tax": FieldAnchor(
            label="Medicare tax withheld",
            direction="right_and_below",
            value_type="decimal",
            aliases=["medicare tax", "box 6"],
        ),
    },

    # Validation rules for W-2
    validation_rules=[
        (
            "ss_tax_rate",
            ["box3_ss_wages", "box4_ss_tax"],
            lambda f: (
                None
                if not hasattr(f, "box3_ss_wages") or f.box3_ss_wages is None
                else (
                    None
                    if f.box4_ss_tax is None
                    else (
                        None
                        if abs(float(f.box4_ss_tax) / float(f.box3_ss_wages) - 0.062) < 0.005
                        else f"SS tax rate {float(f.box4_ss_tax)/float(f.box3_ss_wages):.3f} != 6.2%"
                    )
                )
            ),
            "warning",
        ),
    ],
)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python template_extraction.py <w2.pdf>")
        sys.exit(1)

    result = extractor.extract(sys.argv[1])

    print(f"Extraction Method: {result.extraction_method.value}")
    print(f"Overall Confidence: {result.overall_confidence:.3f}")
    print()

    for name, field in result.fields.items():
        print(f"  {name}: {field.value} (conf={field.confidence:.2f})")
