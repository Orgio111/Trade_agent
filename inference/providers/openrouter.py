"""
OpenRouter Inference Provider.

OpenRouter aggregates 200+ models from multiple providers with a unified API.
Provides the largest selection of free and paid models.

Strengths:
  - Largest model selection (200+ models)
  - Many free models available
  - Acts as fallback aggregator for other providers
  - Good for cost-sensitive tasks

Weaknesses:
  - Variable latency depending on backend provider
  - Free models may be rate-limited or slower
  - No embedding support through free tier
"""

import os
import time
import asyncio
from typing import AsyncGenerator

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from .base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    ProviderUnavailable,
    ProviderTimeout,
    ProviderError,
)


# OpenRouter model IDs organized by capability
# Using :free suffix for free tier models

OPENROUTER_MODELS = {
    # Free tier models
    "reasoning_free": [
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-v4-flash:free",
        "qwen/qwq-32b:free",
    ],
    "fast_free": [
        "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
        "meta-llama/llama-4-scout:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ],
    "analysis_free": [
        "deepseek/deepseek-chat-v3-0324:free",
        "google/gemini-2.0-flash-exp:free",
    ],
    "coding_free": [
        "qwen/qwen3-coder-480b-a35b:free",
        "deepseek/deepseek-chat-v3-0324:free",
    ],
    # Default mapping (simplified keys for router)
    "reasoning": "deepseek/deepseek-v4-flash:free",
    "fast": "meta-llama/llama-4-scout:free",
    "analysis": "deepseek/deepseek-chat-v3-0324:free",
    "coding": "qwen/qwen3-coder-480b-a35b:free",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter inference provider.

    Aggregates 200+ models from multiple providers with a unified API.
    Used primarily as a fallback when primary providers fail.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        config = ProviderConfig(
            api_key=key,
            base_url=OPENROUTER_BASE_URL,
            timeout=60.0,  # Longer timeout for fallback provider
            max_retries=3,
            rate_limit_rps=20.0,
            free_tier=True,
            cost_per_mtok=0.0,  # Using free models
            models=dict(OPENROUTER_MODELS),
        )
        super().__init__(config)

        self._client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=self.config.api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/quantex/quantex",
                "X-Title": "QUANTEX Trading System",
            },
        )

    @property
    def name(self) -> str:
        return "openrouter"

    def _get_fallback_chain(self, task_type: str) -> list[str]:
        """Get ordered fallback chain for a task type."""
        # Check if there's a free model list for this type
        type_models = OPENROUTER_MODELS.get(f"{task_type}_free")
        if type_models:
            return type_models

        # Default fallback: try analysis models
        return OPENROUTER_MODELS.get("analysis_free", OPENROUTER_MODELS["fast_free"])

    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send request to OpenRouter with fallback chain."""
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        # Determine task type from model name for fallback chain
        task_type = "fast"
        for key in ["reasoning", "analysis", "coding", "fast"]:
            if key in model:
                task_type = key
                break

        models_to_try = [model] + self._get_fallback_chain(task_type)
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for m in models_to_try:
            try:
                # Strip :free suffix for the actual API call
                api_model = m.replace(":free", "")

                response = await self._client.chat.completions.create(
                    model=api_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.timeout,
                )

                latency_ms = (time.time() - t0) * 1000
                self.record_latency(latency_ms)

                usage = response.usage
                return ProviderResponse(
                    content=response.choices[0].message.content or "",
                    model=m,
                    provider="openrouter",
                    latency_ms=round(latency_ms, 1),
                    tokens_prompt=usage.prompt_tokens if usage else 0,
                    tokens_completion=usage.completion_tokens if usage else 0,
                    cost_usd=0.0,
                )

            except (RateLimitError, APIError, APITimeoutError) as e:
                last_error = e
                await asyncio.sleep(0.5)
                continue

            except Exception as e:
                last_error = ProviderError(f"OpenRouter error: {e}")
                continue

        self._error_count += 1
        raise last_error or ProviderUnavailable("All OpenRouter models failed")

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Streaming inference from OpenRouter."""
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        try:
            api_model = model.replace(":free", "")

            stream = await self._client.chat.completions.create(
                model=api_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                timeout=self.config.timeout,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            latency_ms = (time.time() - t0) * 1000
            self.record_latency(latency_ms)

        except Exception as e:
            self._error_count += 1
            raise ProviderError(f"OpenRouter streaming failed: {e}")

    async def is_available(self) -> bool:
        """Check if OpenRouter API key is configured."""
        return bool(self.config.api_key)
