"""Main Extractor — the public API for extracture."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from extracture.agentic.extractor import AgenticExtractor
from extracture.config import get_config
from extracture.consensus.engine import ConsensusEngine
from extracture.correction.router import HITLRouter, ReviewQueue
from extracture.correction.store import CorrectionStore
from extracture.models import ExtractionResult, FieldResult, ReviewDecision
from extracture.providers.registry import ProviderRegistry
from extracture.schema import CrossFieldRule, ExtractionSchema, FieldAnchor
from extracture.templates.engine import TemplateExtractor
from extracture.verification.calibration import ConfidenceCalibrator
from extracture.verification.grounding import GroundingVerifier
from extracture.verification.validator import CrossFieldValidator

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Extractor:
    """High-accuracy, schema-driven document extraction.

    Usage:
        from extracture import Extractor
        from pydantic import BaseModel, Field

        class Invoice(BaseModel):
            vendor: str = Field(description="Vendor name")
            total: float = Field(description="Total amount")

        extractor = Extractor(
            schema=Invoice,
            providers=["openai:gpt-4o", "anthropic:claude-sonnet-4-6-20250514"],
        )
        result = extractor.extract("invoice.pdf")
        print(result.data)  # Invoice(vendor="Acme", total=1500.0)
    """

    def __init__(
        self,
        schema: type[BaseModel],
        *,
        providers: list[str] | None = None,
        ocr_engine: str | None = None,
        consensus: str = "confidence_weighted",
        api_keys: dict[str, str] | None = None,
        # Schema options
        form_title: str | None = None,
        form_instructions: str = "",
        field_labels: dict[str, str] | None = None,
        field_sections: dict[str, list[str]] | None = None,
        # Validation
        validation_rules: list[Callable | tuple] | None = None,
        # Template anchors for known document types
        template_anchors: dict[str, FieldAnchor] | None = None,
        # Feature flags
        enable_grounding: bool = False,
        enable_nli_grounding: bool = False,
        enable_hitl: bool = True,
        enable_rag: bool = False,
        # Config overrides
        confidence_floor: float | None = None,
        reexamine_threshold: float | None = None,
        auto_accept_threshold: float | None = None,
        # Correction storage
        correction_store_path: str | Path | None = None,
        # Calibration
        calibration_path: str | Path | None = None,
    ):
        # Build config with overrides
        config_overrides: dict[str, Any] = {}
        if ocr_engine:
            config_overrides["default_ocr_engine"] = ocr_engine
        if consensus:
            config_overrides["consensus_strategy"] = consensus
        if confidence_floor is not None:
            config_overrides["confidence_floor"] = confidence_floor
        if reexamine_threshold is not None:
            config_overrides["reexamine_threshold"] = reexamine_threshold
        if auto_accept_threshold is not None:
            config_overrides["auto_accept_threshold"] = auto_accept_threshold
        config_overrides["enable_grounding"] = enable_grounding
        config_overrides["enable_hitl_routing"] = enable_hitl
        config_overrides["enable_rag"] = enable_rag

        self.config = get_config(**config_overrides)

        # Build schema
        cross_field_rules = []
        if validation_rules:
            for rule in validation_rules:
                if isinstance(rule, tuple) and len(rule) >= 3:
                    cross_field_rules.append(
                        CrossFieldRule(
                            name=rule[0],
                            fields=rule[1],
                            check=rule[2],
                            severity=rule[3] if len(rule) > 3 else "error",
                        )
                    )
                elif callable(rule):
                    cross_field_rules.append(
                        CrossFieldRule(
                            name=f"rule_{len(cross_field_rules)}",
                            fields=[],
                            check=lambda data, fn=rule: fn(data),
                        )
                    )

        self.schema = ExtractionSchema(
            model=schema,
            form_title=form_title,
            form_instructions=form_instructions,
            field_labels=field_labels,
            field_sections=field_sections,
            validation_rules=cross_field_rules,
            template_anchors=template_anchors,
        )

        # Build providers
        provider_specs = providers or [self.config.default_provider]
        registry = ProviderRegistry(self.config)
        self._extraction_providers, self._ocr_providers = registry.create_extraction_providers(
            provider_specs, api_keys
        )

        # Build components
        self._consensus = ConsensusEngine(strategy=self.config.consensus_strategy)

        self._grounding = (
            GroundingVerifier(config=self.config, use_nli=enable_nli_grounding)
            if enable_grounding
            else None
        )

        self._calibrator = ConfidenceCalibrator(config=self.config)
        if calibration_path and Path(calibration_path).exists():
            self._calibrator.load(calibration_path)

        self._validator = CrossFieldValidator()
        self._validator.auto_detect_format_rules(self.schema.field_names)

        self._template_extractor = TemplateExtractor()
        self._hitl_router = HITLRouter(config=self.config)

        self._correction_store = CorrectionStore(correction_store_path) if enable_rag else None

        # Build ingest router (lazy import to avoid heavy deps at import time)
        self._ingest_router = None
        self._ocr_engine = ocr_engine

    def extract(self, source: str | Path | bytes, file_type: str | None = None) -> ExtractionResult:
        """Extract structured data from a document.

        Args:
            source: File path, bytes, or URL
            file_type: Optional file type hint ("pdf", "png", etc.)

        Returns:
            ExtractionResult with .data, .fields, .confidence, etc.
        """
        return asyncio.run(self.aextract(source, file_type))

    async def aextract(
        self, source: str | Path | bytes, file_type: str | None = None
    ) -> ExtractionResult:
        """Async version of extract."""
        # Lazy init ingest router
        if self._ingest_router is None:
            from extracture.ingest.router import IngestRouter

            self._ingest_router = IngestRouter(config=self.config, ocr_engine=self._ocr_engine)

        # Load source bytes for OCR providers
        file_bytes = None
        if isinstance(source, bytes):
            file_bytes = source
        elif isinstance(source, (str, Path)):
            file_bytes = Path(source).read_bytes()

        # Step 1: Ingest
        logger.info("Ingesting document...")
        ingest_result = self._ingest_router.ingest(source, file_type)

        # Step 2: Try template extraction first (520x faster, 3700x cheaper)
        if self.schema.has_template:
            logger.info("Attempting template-based extraction...")
            template_fields = self._template_extractor.extract(
                self.schema,
                ingest_result.text_content,
                ingest_result.word_positions,
            )

            # If template extraction is high confidence for all fields, use it
            if template_fields:
                all_confident = all(
                    f.confidence >= 0.90 for f in template_fields.values()
                )
                coverage = len(template_fields) / max(len(self.schema.field_names), 1)

                if all_confident and coverage >= 0.8:
                    logger.info(
                        f"Template extraction sufficient: {len(template_fields)} fields, "
                        f"all ≥0.90 confidence"
                    )
                    return self._build_template_result(template_fields, ingest_result)

        # Step 3: Full agentic extraction
        agentic = AgenticExtractor(
            extraction_providers=self._extraction_providers,
            ocr_providers=self._ocr_providers,
            consensus_engine=self._consensus,
            grounding_verifier=self._grounding,
            calibrator=self._calibrator,
            validator=self._validator,
            config=self.config,
        )

        result = await agentic.extract(self.schema, ingest_result, file_bytes)
        return result

    def extract_batch(
        self,
        sources: list[str | Path | bytes],
        file_types: list[str | None] | None = None,
        max_concurrent: int = 5,
    ) -> list[ExtractionResult]:
        """Extract from multiple documents concurrently."""
        return asyncio.run(self._batch_extract(sources, file_types, max_concurrent))

    async def _batch_extract(
        self,
        sources: list[str | Path | bytes],
        file_types: list[str | None] | None,
        max_concurrent: int,
    ) -> list[ExtractionResult]:
        types = file_types or [None] * len(sources)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_one(src, ft):
            async with semaphore:
                return await self.aextract(src, ft)

        tasks = [extract_one(s, t) for s, t in zip(sources, types)]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def review(self, result: ExtractionResult) -> ReviewQueue:
        """Get the review queue for an extraction result."""
        return self._hitl_router.route(result)

    def learn_from_corrections(self, result: ExtractionResult) -> None:
        """Store corrections from a result for future RAG-based improvement."""
        if not self._correction_store:
            logger.warning("RAG not enabled. Set enable_rag=True to use corrections.")
            return

        if result.corrections:
            self._correction_store.add_corrections_from_result(
                document_type=self.schema.form_title,
                corrections=result.corrections,
                document_text=None,
            )

    def save_calibration(self, path: str | Path) -> None:
        """Save current confidence calibration parameters."""
        self._calibrator.save(path)

    def load_calibration(self, path: str | Path) -> None:
        """Load confidence calibration parameters."""
        self._calibrator.load(path)

    def register_template(
        self, field_anchors: dict[str, FieldAnchor]
    ) -> None:
        """Register template anchors for faster extraction."""
        self.schema.template_anchors.update(field_anchors)

    def _build_template_result(
        self, fields: dict[str, FieldResult], ingest_result
    ) -> ExtractionResult:
        """Build an ExtractionResult from template extraction."""
        from extracture.models import (
            ExtractionAudit,
            ExtractionMethod,
            ExtractionStatus,
        )

        non_null_confs = [
            f.confidence for f in fields.values() if f.value is not None
        ]
        overall = sum(non_null_confs) / len(non_null_confs) if non_null_confs else 0.0

        # Try to parse into schema model
        parsed = None
        try:
            raw = {name: f.value for name, f in fields.items()}
            parsed = self.schema.parse_fields(raw)
        except Exception:
            pass

        return ExtractionResult(
            data=parsed,
            fields=fields,
            overall_confidence=round(overall, 4),
            extraction_method=ExtractionMethod.TEMPLATE,
            status=ExtractionStatus.EXTRACTED,
            review_decision=ReviewDecision.AUTO_ACCEPT,
            audit=ExtractionAudit(
                extraction_method=ExtractionMethod.TEMPLATE,
                template_used=self.schema.form_title,
                preprocessing_steps=ingest_result.preprocessing_applied,
            ),
            page_count=ingest_result.page_count,
        )
