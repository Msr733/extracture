"""Configuration for extracture."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ExtractureConfig(BaseSettings):
    """Global configuration, overridable via environment variables."""

    model_config = {"env_prefix": "EXTRACTURE_"}

    # OCR
    default_ocr_engine: str = "pymupdf"
    ocr_dpi: int = 300
    digital_text_threshold: int = 50

    # Extraction
    default_provider: str = "openai:gpt-4o"
    consensus_strategy: str = "confidence_weighted"
    max_tokens: int = 4096

    # Confidence
    confidence_floor: float = 0.70
    reexamine_threshold: float = 0.85
    auto_accept_threshold: float = 0.95
    max_reexamine_fields: int = 5

    # Calibration
    default_temperature: float = 1.5

    # Grounding
    enable_grounding: bool = False
    grounding_similarity_threshold: float = 0.6
    grounding_nli_threshold: float = 0.7

    # Preprocessing
    enable_preprocessing: bool = True
    skew_correction_threshold: float = 2.0
    min_dpi_threshold: int = 200
    contrast_threshold: float = 0.4

    # HITL
    enable_hitl_routing: bool = True

    # Retrieval-augmented extraction
    enable_rag: bool = False
    rag_num_examples: int = 3

    # Performance
    max_concurrent_providers: int = 5
    provider_timeout_seconds: int = 120
    subprocess_timeout_seconds: int = 60

    # Costs
    track_costs: bool = True

    # Logging
    log_level: str = "INFO"
    log_raw_responses: bool = False


def get_config(**overrides: object) -> ExtractureConfig:
    return ExtractureConfig(**overrides)
