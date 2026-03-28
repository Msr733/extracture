"""Provider registry — parse provider strings and instantiate providers."""

from __future__ import annotations

import logging

from extracture.config import ExtractureConfig
from extracture.providers.base import ExtractionProvider, OCRProvider
from extracture.providers.litellm_provider import LiteLLMProvider
from extracture.providers.textract_provider import TextractProvider

logger = logging.getLogger(__name__)

# Provider string format: "provider_type:model_name" or just "model_name"
# Examples:
#   "openai:gpt-4o"
#   "anthropic:claude-sonnet-4-6-20250514"
#   "gemini/gemini-2.5-flash"  (LiteLLM format)
#   "aws-textract"
#   "ollama/llama3.2-vision"

OCR_PROVIDERS = {"aws-textract", "textract"}


class ProviderRegistry:
    """Parses provider configuration and instantiates providers."""

    def __init__(self, config: ExtractureConfig):
        self.config = config

    def create_extraction_providers(
        self,
        provider_specs: list[str],
        api_keys: dict[str, str] | None = None,
    ) -> tuple[list[ExtractionProvider], list[OCRProvider]]:
        """Create provider instances from spec strings.

        Returns:
            (extraction_providers, ocr_providers)
        """
        extraction_providers: list[ExtractionProvider] = []
        ocr_providers: list[OCRProvider] = []
        api_keys = api_keys or {}

        for spec in provider_specs:
            spec = spec.strip()

            if spec.lower() in OCR_PROVIDERS:
                provider = self._create_ocr_provider(spec, api_keys)
                if provider:
                    ocr_providers.append(provider)
                continue

            provider = self._create_llm_provider(spec, api_keys)
            if provider:
                extraction_providers.append(provider)

        return extraction_providers, ocr_providers

    def _create_llm_provider(
        self, spec: str, api_keys: dict[str, str]
    ) -> LiteLLMProvider | None:
        """Create a LiteLLM-based provider from spec."""
        # Parse "provider:model" format
        model = spec
        api_key = None

        if ":" in spec and not spec.startswith("ollama"):
            parts = spec.split(":", 1)
            provider_type = parts[0].lower()
            model_name = parts[1]

            # Map to LiteLLM format
            if provider_type == "openai":
                model = model_name
                api_key = api_keys.get("openai") or api_keys.get("OPENAI_API_KEY")
            elif provider_type == "anthropic":
                model = model_name
                api_key = api_keys.get("anthropic") or api_keys.get("ANTHROPIC_API_KEY")
            elif provider_type == "google" or provider_type == "gemini":
                model = f"gemini/{model_name}"
                api_key = api_keys.get("google") or api_keys.get("GEMINI_API_KEY")
            elif provider_type == "deepseek":
                model = f"deepseek/{model_name}"
                api_key = api_keys.get("deepseek") or api_keys.get("DEEPSEEK_API_KEY")
            else:
                model = spec

        logger.info(f"Creating LLM provider: {model}")
        return LiteLLMProvider(
            model=model,
            config=self.config,
            api_key=api_key,
        )

    def _create_ocr_provider(
        self, spec: str, api_keys: dict[str, str]
    ) -> TextractProvider | None:
        """Create an OCR provider."""
        if spec.lower() in OCR_PROVIDERS:
            logger.info("Creating AWS Textract provider")
            return TextractProvider(
                aws_access_key_id=api_keys.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=api_keys.get("AWS_SECRET_ACCESS_KEY"),
                aws_region=api_keys.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
        return None
