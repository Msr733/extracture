"""AWS Textract provider for key-value extraction with bounding boxes."""

from __future__ import annotations

import logging
import time
from typing import Any

from rapidfuzz import fuzz

from extracture.models import BoundingBox, FieldResult, IngestResult, RawExtraction
from extracture.providers.base import OCRProvider
from extracture.schema import ExtractionSchema

logger = logging.getLogger(__name__)


class TextractProvider(OCRProvider):
    """AWS Textract for structured key-value extraction with bounding boxes."""

    provider_name = "textract"

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_region: str = "us-east-1",
    ):
        self._access_key = aws_access_key_id
        self._secret_key = aws_secret_access_key
        self._region = aws_region
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3

            kwargs: dict[str, Any] = {"region_name": self._region}
            if self._access_key:
                kwargs["aws_access_key_id"] = self._access_key
                kwargs["aws_secret_access_key"] = self._secret_key
            self._client = boto3.client("textract", **kwargs)
        return self._client

    async def extract_key_values(
        self,
        file_bytes: bytes,
        schema: ExtractionSchema,
    ) -> RawExtraction:
        """Extract key-value pairs using Textract AnalyzeDocument."""
        start = time.time()

        try:
            client = self._get_client()
            response = client.analyze_document(
                Document={"Bytes": file_bytes},
                FeatureTypes=["FORMS"],
            )
        except Exception as e:
            logger.error(f"Textract API call failed: {e}")
            return RawExtraction(
                provider=self.provider_name,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        # Parse Textract blocks
        blocks = {b["Id"]: b for b in response.get("Blocks", [])}
        key_value_pairs = self._extract_key_value_pairs(blocks)

        # Build label map from schema
        label_map = self._build_label_map(schema)

        # Match Textract keys to schema fields
        fields: dict[str, FieldResult] = {}

        for tv_key, tv_value, tv_key_conf, tv_val_conf, key_bbox, val_bbox in key_value_pairs:
            matched_field = self._match_to_field(tv_key, label_map)
            if not matched_field:
                continue

            confidence = min(tv_key_conf, tv_val_conf) / 100.0

            # Skip if we already have a higher-confidence match
            if matched_field in fields and fields[matched_field].confidence >= confidence:
                continue

            fields[matched_field] = FieldResult(
                value=tv_value.strip() if tv_value else None,
                confidence=confidence,
                bbox=val_bbox,
                label_bbox=key_bbox,
                source_quote=f"{tv_key}: {tv_value}",
            )

        duration_ms = (time.time() - start) * 1000
        logger.info(f"Textract: matched {len(fields)} fields in {duration_ms:.0f}ms")

        return RawExtraction(
            provider=self.provider_name,
            fields=fields,
            duration_ms=duration_ms,
        )

    def _extract_key_value_pairs(
        self, blocks: dict[str, dict]
    ) -> list[tuple[str, str, float, float, BoundingBox | None, BoundingBox | None]]:
        """Extract key-value pairs from Textract blocks.
        Returns: [(key_text, value_text, key_confidence, value_confidence, key_bbox, value_bbox)]
        """
        pairs = []

        for block_id, block in blocks.items():
            if block.get("BlockType") != "KEY_VALUE_SET":
                continue
            if "KEY" not in block.get("EntityTypes", []):
                continue

            # Get key text
            key_text = self._get_text_from_block(block, blocks)
            key_confidence = block.get("Confidence", 0.0)
            key_bbox = self._block_to_bbox(block)

            # Find associated VALUE block
            value_text = ""
            value_confidence = 0.0
            value_bbox = None

            for rel in block.get("Relationships", []):
                if rel["Type"] == "VALUE":
                    for val_id in rel["Ids"]:
                        val_block = blocks.get(val_id, {})
                        value_text = self._get_text_from_block(val_block, blocks)
                        value_confidence = val_block.get("Confidence", 0.0)
                        value_bbox = self._block_to_bbox(val_block)
                        break

            if key_text.strip():
                pairs.append((
                    key_text.strip(),
                    value_text.strip(),
                    key_confidence,
                    value_confidence,
                    key_bbox,
                    value_bbox,
                ))

        return pairs

    def _get_text_from_block(self, block: dict, blocks: dict[str, dict]) -> str:
        """Get text from a block by traversing CHILD relationships."""
        words = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    child = blocks.get(child_id, {})
                    if child.get("BlockType") == "WORD":
                        words.append(child.get("Text", ""))
        return " ".join(words)

    def _block_to_bbox(self, block: dict) -> BoundingBox | None:
        """Convert Textract block geometry to BoundingBox."""
        geo = block.get("Geometry", {}).get("BoundingBox")
        if not geo:
            return None
        return BoundingBox(
            page=block.get("Page", 1) - 1,  # 0-indexed
            x=round(geo["Left"], 4),
            y=round(geo["Top"], 4),
            w=round(geo["Width"], 4),
            h=round(geo["Height"], 4),
        )

    def _build_label_map(self, schema: ExtractionSchema) -> dict[str, list[str]]:
        """Build a map of field_name -> list of possible labels for matching."""
        label_map: dict[str, list[str]] = {}

        for field_name in schema.field_names:
            labels: list[str] = []
            # Primary label from schema
            label = schema.field_labels.get(field_name, "")
            if label:
                labels.append(label.lower())
                # Split on separators
                for sep in [" — ", " - ", ": ", " – "]:
                    if sep in label:
                        parts = label.split(sep)
                        labels.extend(p.strip().lower() for p in parts if p.strip())

            # Field name as label
            labels.append(field_name.replace("_", " ").lower())

            # Template anchors if available
            if field_name in schema.template_anchors:
                anchor = schema.template_anchors[field_name]
                labels.append(anchor.label.lower())
                labels.extend(a.lower() for a in anchor.aliases)

            label_map[field_name] = list(set(labels))

        return label_map

    def _match_to_field(
        self, textract_key: str, label_map: dict[str, list[str]]
    ) -> str | None:
        """Match a Textract key to a schema field using fuzzy matching."""
        key_lower = textract_key.lower().strip()

        best_match: str | None = None
        best_score = 0.0
        threshold = 75.0  # Minimum fuzzy match score

        for field_name, labels in label_map.items():
            for label in labels:
                # Exact substring match first
                if label in key_lower or key_lower in label:
                    return field_name

                # Fuzzy match
                score = fuzz.ratio(key_lower, label)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = field_name

                # Token sort ratio for reordered words
                token_score = fuzz.token_sort_ratio(key_lower, label)
                if token_score > best_score and token_score >= threshold:
                    best_score = token_score
                    best_match = field_name

        return best_match
