"""Universal LLM provider using LiteLLM — supports 100+ models."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any

from extracture.config import ExtractureConfig, get_config
from extracture.models import FieldResult, IngestResult, RawExtraction
from extracture.providers.base import ExtractionProvider
from extracture.schema import ExtractionSchema

logger = logging.getLogger(__name__)

# Cost per 1M tokens (input, output) for common models
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-sonnet-4-6-20250514": (3.0, 15.0),
    "claude-opus-4-6-20250901": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    "gemini/gemini-2.5-flash": (0.15, 0.60),
    "gemini/gemini-2.5-pro": (1.25, 10.0),
    "deepseek/deepseek-chat": (0.14, 0.28),
}


class LiteLLMProvider(ExtractionProvider):
    """Universal provider that works with any LLM supported by LiteLLM."""

    def __init__(
        self,
        model: str,
        config: ExtractureConfig | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        use_vision: bool = True,
        use_text: bool = True,
        temperature: float = 0.0,
    ):
        self.model = model
        self.config = config or get_config()
        self.api_key = api_key
        self.api_base = api_base
        self.use_vision = use_vision
        self.use_text = use_text
        self.temperature = temperature
        self.provider_name = model

    async def extract(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
    ) -> RawExtraction:
        """Extract using LLM — vision and/or text modes."""
        import litellm

        start = time.time()
        fields: dict[str, FieldResult] = {}
        raw_response = None
        cost = 0.0

        try:
            # Build messages based on available data and capabilities
            messages = self._build_messages(schema, ingest_result)

            # Use tool/function calling for structured output when possible
            tools = [schema.build_tool_schema()]
            tool_name = tools[0]["name"]

            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=self.temperature,
                max_tokens=schema.max_tokens,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.config.provider_timeout_seconds,
            )

            # Parse response
            raw_response = str(response)
            tool_calls = response.choices[0].message.tool_calls

            if tool_calls:
                tool_args = json.loads(tool_calls[0].function.arguments)
                fields = self._parse_tool_response(tool_args, schema)
            else:
                # Fallback: parse from message content
                content = response.choices[0].message.content or ""
                parsed = self._parse_json_response(content)
                if parsed:
                    fields = self._parse_tool_response(parsed, schema)

            # Calculate cost
            usage = response.usage
            if usage:
                cost = self.get_cost_estimate(usage.prompt_tokens, usage.completion_tokens)

        except Exception as e:
            logger.error(f"Extraction failed with {self.model}: {e}", exc_info=True)
            return RawExtraction(
                provider=self.provider_name,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"{self.model}: extracted {len(fields)} fields in {duration_ms:.0f}ms (${cost:.4f})"
        )

        return RawExtraction(
            provider=self.provider_name,
            fields=fields,
            raw_response=raw_response[:2000] if raw_response else None,
            duration_ms=duration_ms,
            cost_estimate_usd=cost,
        )

    async def reexamine(
        self,
        schema: ExtractionSchema,
        ingest_result: IngestResult,
        low_confidence_fields: dict[str, dict[str, Any]],
    ) -> RawExtraction:
        """Focused re-examination of specific fields."""
        import litellm

        start = time.time()

        prompt = schema.build_reexamination_prompt(
            low_confidence_fields,
            text_content=ingest_result.text_content,
        )

        messages: list[dict[str, Any]] = []

        if self.use_vision and ingest_result.page_images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_bytes in ingest_result.page_images[:3]:
                b64 = base64.b64encode(img_bytes).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
                api_key=self.api_key,
                api_base=self.api_base,
                timeout=self.config.provider_timeout_seconds,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            parsed = self._parse_json_response(content)
            fields = self._parse_tool_response(parsed, schema) if parsed else {}

            return RawExtraction(
                provider=f"{self.provider_name}_reexam",
                fields=fields,
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"Re-examination failed with {self.model}: {e}")
            return RawExtraction(
                provider=f"{self.provider_name}_reexam",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _build_messages(
        self, schema: ExtractionSchema, ingest_result: IngestResult
    ) -> list[dict[str, Any]]:
        """Build LLM messages with text and/or vision content."""
        messages: list[dict[str, Any]] = []

        # System message
        messages.append({
            "role": "system",
            "content": (
                "You are a precise document extraction system. Extract ONLY what you can see "
                "in the document. Never guess or fabricate values. Report confidence accurately."
            ),
        })

        # User message with content
        content: list[dict[str, Any]] = []

        # Text prompt
        if self.use_text and ingest_result.text_content:
            prompt = schema.build_extraction_prompt(
                text_content=ingest_result.text_content, mode="text"
            )
        elif self.use_vision and ingest_result.page_images:
            prompt = schema.build_extraction_prompt(mode="vision")
        else:
            prompt = schema.build_extraction_prompt(mode="text")

        content.append({"type": "text", "text": prompt})

        # Add page images for vision
        if self.use_vision and ingest_result.page_images:
            for img_bytes in ingest_result.page_images[:5]:  # Limit pages
                b64 = base64.b64encode(img_bytes).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        messages.append({"role": "user", "content": content})
        return messages

    def _parse_tool_response(
        self, raw: dict[str, Any], schema: ExtractionSchema
    ) -> dict[str, FieldResult]:
        """Parse tool call response into FieldResult objects."""
        fields: dict[str, FieldResult] = {}

        for field_name in schema.field_names:
            field_data = raw.get(field_name)
            if field_data is None:
                fields[field_name] = FieldResult(value=None, confidence=0.0)
                continue

            if isinstance(field_data, dict):
                value = field_data.get("value")
                confidence = float(field_data.get("confidence", 0.5))
                source_quote = field_data.get("source_quote")

                # Handle 0.0 confidence = not visible
                if confidence == 0.0:
                    value = None

                fields[field_name] = FieldResult(
                    value=value,
                    confidence=confidence,
                    source_quote=source_quote,
                    sources=[],
                )
            else:
                # Direct value (no confidence wrapper)
                fields[field_name] = FieldResult(
                    value=field_data,
                    confidence=0.5,
                )

        return fields

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, handling markdown fences and truncation."""
        # Strip markdown fences
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to repair truncated JSON
        try:
            open_braces = text.count("{") - text.count("}")
            open_brackets = text.count("[") - text.count("]")
            repaired = text + "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Try to extract first JSON object
        try:
            start = text.index("{")
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        except (ValueError, json.JSONDecodeError):
            pass

        logger.warning(f"Failed to parse JSON response: {text[:200]}")
        return None

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        model_key = self.model.split("/")[-1] if "/" in self.model else self.model
        costs = MODEL_COSTS.get(model_key, (0.0, 0.0))
        return (input_tokens * costs[0] + output_tokens * costs[1]) / 1_000_000
