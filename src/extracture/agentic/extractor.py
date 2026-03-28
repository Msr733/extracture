"""Agentic multi-pass extraction with self-verification and re-examination.

Research: Agentic RL-driven extraction improves F1 from 0.30 to 0.962
on financial documents (2025). Self-correction after cross-field validation
catches an additional 30-50% of remaining errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from extracture.config import ExtractureConfig, get_config
from extracture.consensus.engine import ConsensusEngine
from extracture.models import (
    ExtractionAudit,
    ExtractionResult,
    ExtractionStatus,
    FieldResult,
    IngestResult,
    RawExtraction,
    ReviewDecision,
    ValidationError,
)
from extracture.providers.base import ExtractionProvider, OCRProvider
from extracture.schema import ExtractionSchema
from extracture.verification.calibration import ConfidenceCalibrator
from extracture.verification.grounding import GroundingVerifier
from extracture.verification.validator import CrossFieldValidator

logger = logging.getLogger(__name__)


class AgenticExtractor:
    """Multi-pass, self-verifying extraction engine."""

    def __init__(
        self,
        extraction_providers: list[ExtractionProvider],
        ocr_providers: list[OCRProvider] | None = None,
        consensus_engine: ConsensusEngine | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        calibrator: ConfidenceCalibrator | None = None,
        validator: CrossFieldValidator | None = None,
        config: ExtractureConfig | None = None,
    ):
        self.extraction_providers = extraction_providers
        self.ocr_providers = ocr_providers or []
        self.consensus = consensus_engine or ConsensusEngine()
        self.grounding = grounding_verifier
        self.calibrator = calibrator or ConfidenceCalibrator()
        self.validator: CrossFieldValidator = validator or CrossFieldValidator()
        self.config = config or get_config()

    async def extract(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
        file_bytes: bytes | None = None,
    ) -> ExtractionResult[Any]:
        """Run the full agentic extraction pipeline."""
        start = time.time()
        audit = ExtractionAudit(
            extraction_method=ingest_result.extraction_method,
            ocr_engine=ingest_result.ocr_engine_used,
            preprocessing_steps=ingest_result.preprocessing_applied,
        )

        # ===== PASS 1: Multi-provider parallel extraction =====
        logger.info(f"Pass 1: Extracting with {len(self.extraction_providers)} providers")
        raw_extractions = await self._parallel_extract(schema, ingest_result)

        # Add OCR provider extractions (Textract etc.) if available
        if file_bytes and self.ocr_providers:
            ocr_extractions = await self._ocr_extract(schema, file_bytes)
            raw_extractions.extend(ocr_extractions)

        # Filter out failed extractions
        successful = [e for e in raw_extractions if not e.error]
        audit.providers_used = [e.provider for e in successful]

        if not successful:
            logger.error("All extraction providers failed")
            errors = [e.error for e in raw_extractions if e.error]
            return ExtractionResult(
                status=ExtractionStatus.LOW_CONFIDENCE,
                extraction_method=ingest_result.extraction_method,
                audit=audit,
                page_count=ingest_result.page_count,
            )

        # ===== PASS 2: Consensus merge =====
        logger.info(f"Pass 2: Merging {len(successful)} extractions via consensus")
        consensus_fields = self.consensus.merge(successful, schema.field_names)

        # ===== PASS 3: Grounding verification =====
        if self.grounding and ingest_result.text_content:
            logger.info("Pass 3: Verifying grounding")
            grounding_results = self.grounding.verify_all_fields(
                consensus_fields, ingest_result.text_content
            )

            grounded_count = 0
            ungrounded_count = 0
            for field_name, gr in grounding_results.items():
                field = consensus_fields.get(field_name)
                if field:
                    field.is_grounded = gr.is_grounded
                    field.grounding_score = gr.score
                    if gr.is_grounded:
                        grounded_count += 1
                    else:
                        ungrounded_count += 1
                        field.confidence *= 0.5  # Penalize ungrounded fields
                        field.flags.append("ungrounded")

            audit.grounding_stats = {
                "grounded": grounded_count,
                "ungrounded": ungrounded_count,
            }
            logger.info(f"Grounding: {grounded_count} grounded, {ungrounded_count} ungrounded")

        # ===== PASS 4: Re-examine low-confidence fields =====
        low_conf_fields = {
            name: {"value": f.value, "confidence": f.confidence}
            for name, f in consensus_fields.items()
            if f.value is not None
            and 0 < f.confidence < self.config.reexamine_threshold
        }

        if low_conf_fields:
            # Cap at max_reexamine_fields, sorted by confidence (lowest first)
            sorted_fields = sorted(low_conf_fields.items(), key=lambda x: x[1]["confidence"])
            fields_to_reexam = dict(sorted_fields[: self.config.max_reexamine_fields])

            logger.info(f"Pass 4: Re-examining {len(fields_to_reexam)} low-confidence fields")
            reexam_results = await self._reexamine(schema, ingest_result, fields_to_reexam)

            for field_name, reexam_field in reexam_results.items():
                original = consensus_fields.get(field_name)
                if original and reexam_field.confidence > original.confidence:
                    original.value = reexam_field.value
                    original.confidence = reexam_field.confidence
                    original.source_quote = reexam_field.source_quote
                    original.was_reexamined = True

            audit.reexamined_fields = list(fields_to_reexam.keys())

        # ===== PASS 5: Confidence calibration =====
        for field_name, field in consensus_fields.items():
            field.calibrated_confidence = self.calibrator.calibrate(
                field_name, field.confidence
            )

        # ===== PASS 6: Cross-field validation =====
        validation_errors: list[ValidationError] = []
        try:
            # Validate using schema rules
            parsed_data = self._build_data_object(schema, consensus_fields)
            if parsed_data:
                errors = schema.validate_cross_field(parsed_data)
                for err_msg in errors:
                    validation_errors.append(
                        ValidationError(rule_name="cross_field", message=err_msg)
                    )

            # Validate using validator rules
            validator_errors = self.validator.validate(consensus_fields, schema.field_names)
            validation_errors.extend(validator_errors)

        except Exception as e:
            logger.warning(f"Cross-field validation failed: {e}")

        # ===== PASS 7: Self-correction on validation failures =====
        if validation_errors and self.extraction_providers:
            logger.info(f"Pass 7: Self-correcting {len(validation_errors)} validation errors")
            corrected = await self._self_correct(
                schema, ingest_result, consensus_fields, validation_errors
            )
            if corrected:
                for field_name, corrected_field in corrected.items():
                    if corrected_field.confidence > consensus_fields.get(field_name, FieldResult()).confidence:
                        consensus_fields[field_name] = corrected_field
                        consensus_fields[field_name].flags.append("self_corrected")

                # Re-validate
                try:
                    parsed_data = self._build_data_object(schema, consensus_fields)
                    if parsed_data:
                        remaining_errors = schema.validate_cross_field(parsed_data)
                        validation_errors = [
                            ValidationError(rule_name="cross_field", message=e)
                            for e in remaining_errors
                        ]
                except Exception:
                    pass

        # ===== Calculate overall confidence =====
        non_null_confs = [
            f.effective_confidence
            for f in consensus_fields.values()
            if f.value is not None and f.effective_confidence > 0
        ]
        overall_confidence = (
            round(sum(non_null_confs) / len(non_null_confs), 4) if non_null_confs else 0.0
        )

        # ===== Determine status and review decision =====
        if overall_confidence < self.config.confidence_floor:
            status = ExtractionStatus.LOW_CONFIDENCE
        else:
            status = ExtractionStatus.EXTRACTED

        review_decision = self._determine_review(consensus_fields, validation_errors)

        # ===== Build result =====
        duration_ms = (time.time() - start) * 1000
        audit.total_duration_ms = duration_ms
        audit.cost_estimate_usd = sum(
            e.cost_estimate_usd or 0 for e in raw_extractions
        )

        # Try to parse into schema model
        parsed = None
        try:
            parsed = self._build_data_object(schema, consensus_fields)
        except Exception:
            pass

        result = ExtractionResult(
            data=parsed,
            fields=consensus_fields,
            overall_confidence=overall_confidence,
            extraction_method=ingest_result.extraction_method,
            status=status,
            review_decision=review_decision,
            validation_errors=validation_errors,
            audit=audit,
            page_count=ingest_result.page_count,
        )

        # Calculate calibrated overall confidence
        cal_confs = [
            f.calibrated_confidence
            for f in consensus_fields.values()
            if f.value is not None and f.calibrated_confidence is not None and f.calibrated_confidence > 0
        ]
        if cal_confs:
            result.calibrated_overall_confidence = round(sum(cal_confs) / len(cal_confs), 4)

        logger.info(
            f"Extraction complete: {len(consensus_fields)} fields, "
            f"confidence={overall_confidence:.3f}, status={status.value}, "
            f"duration={duration_ms:.0f}ms, cost=${audit.cost_estimate_usd:.4f}"
        )

        return result

    async def _parallel_extract(
        self, schema: ExtractionSchema, ingest_result: IngestResult
    ) -> list[RawExtraction]:
        """Run extraction across all providers in parallel."""
        tasks = [
            provider.extract(schema, ingest_result)
            for provider in self.extraction_providers
        ]

        gather_results: list[RawExtraction | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

        extractions: list[RawExtraction] = []
        for i, r in enumerate(gather_results):
            if isinstance(r, BaseException):
                provider_name = self.extraction_providers[i].provider_name
                logger.error(f"Provider {provider_name} failed: {r}")
                extractions.append(
                    RawExtraction(provider=provider_name, error=str(r))
                )
            else:
                extractions.append(r)

        return extractions

    async def _ocr_extract(
        self, schema: ExtractionSchema, file_bytes: bytes
    ) -> list[RawExtraction]:
        """Run OCR provider extractions."""
        tasks = [
            provider.extract_key_values(file_bytes, schema)
            for provider in self.ocr_providers
        ]
        ocr_gather_results: list[RawExtraction | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

        extractions: list[RawExtraction] = []
        for i, r in enumerate(ocr_gather_results):
            if isinstance(r, BaseException):
                logger.error(f"OCR provider failed: {r}")
            else:
                extractions.append(r)

        return extractions

    async def _reexamine(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
        low_confidence_fields: dict[str, dict[str, Any]],
    ) -> dict[str, FieldResult]:
        """Re-examine low-confidence fields with a focused prompt."""
        # Use the first provider for re-examination
        provider = self.extraction_providers[0]
        result = await provider.reexamine(schema, ingest_result, low_confidence_fields)
        return result.fields if not result.error else {}

    async def _self_correct(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
        current_fields: dict[str, FieldResult],
        validation_errors: list[ValidationError],
    ) -> dict[str, FieldResult] | None:
        """Self-correct extraction based on validation errors."""
        try:

            error_descriptions = "\n".join(
                f"- {e.message} (affects: {', '.join(e.affected_fields)})"
                for e in validation_errors
            )

            current_values = {
                name: {"value": f.value, "confidence": f.confidence}
                for name, f in current_fields.items()
                if f.value is not None
            }

            affected_fields = set()
            for e in validation_errors:
                affected_fields.update(e.affected_fields)

            (
                f"I extracted data from a {schema.form_title} but found validation errors:\n"
                f"{error_descriptions}\n\n"
                f"Current extracted values:\n"
                f"{json.dumps(current_values, indent=2, default=str)}\n\n"
                f"Please re-examine the document and provide corrected values for ONLY "
                f"these fields: {', '.join(affected_fields)}\n"
                f"Respond with JSON containing the corrected fields."
            )

            # Use the first provider for self-correction
            provider = self.extraction_providers[0]
            reexam_fields = {
                name: current_values.get(name, {"value": None, "confidence": 0})
                for name in affected_fields
            }
            result = await provider.reexamine(schema, ingest_result, reexam_fields)
            return result.fields if not result.error else None

        except Exception as e:
            logger.warning(f"Self-correction failed: {e}")
            return None

    def _build_data_object(
        self, schema: ExtractionSchema, fields: dict[str, FieldResult]
    ) -> Any:
        """Try to build the schema's Pydantic model from field values."""
        raw = {}
        for name, field in fields.items():
            raw[name] = field.value

        try:
            return schema.parse_fields(raw)
        except Exception:
            return None

    def _determine_review(
        self,
        fields: dict[str, FieldResult],
        validation_errors: list[ValidationError],
    ) -> ReviewDecision:
        """Determine whether the result needs human review."""
        if not self.config.enable_hitl_routing:
            return ReviewDecision.AUTO_ACCEPT

        # Any validation errors = full review
        error_count = sum(1 for e in validation_errors if e.severity == "error")
        if error_count > 0:
            return ReviewDecision.FULL_REVIEW

        # Check field-level confidence
        low_conf_count = 0
        ungrounded_count = 0
        for field in fields.values():
            if field.value is None:
                continue
            if field.effective_confidence < self.config.auto_accept_threshold:
                low_conf_count += 1
            if field.is_grounded is False:
                ungrounded_count += 1

        if low_conf_count > 0 or ungrounded_count > 0:
            return ReviewDecision.PARTIAL_REVIEW

        return ReviewDecision.AUTO_ACCEPT
