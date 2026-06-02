"""
Groq LPU Inference Provider.

Groq's LPU (Language Processing Unit) delivers extremely fast inference
(800+ tok/s) for smaller models. Ideal for latency-sensitive trading agents.

Strengths:
  - Fastest inference of any provider (800+ tok/s)
  - Free tier available (30 req/min)
  - Excellent for real-time trading decisions

Weaknesses:
  - Limited to smaller models (8B-70B range)
  - Rate limited on free tier (30 req/min)
  - No embedding support
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


GROQ_DEFAULT_MODELS = {
    "fast": "llama3-8b-8192",
    "reasoning": "qwen/qwen3-32b",
    "analysis": "llama-3.3-70b-versatile",
    "tiny": "llama3-8b-8192",
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Free tier limits
GROQ_FREE_RATE_LIMIT = 30  # req/min
GROQ_FREE_TOKENS_PER_MIN = 7000  # tokens/min (for smaller models)


class GroqProvider(BaseProvider):
    """
    Groq LPU inference provider.

    Uses Groq's custom LPU hardware for ultra-fast inference.
    Best for latency-sensitive agents (execution, scalping, risk).
    """

    def __init__(self, api_key: str | None = None):
        config = ProviderConfig(
            api_key=api_key or os.getenv("GROQ_API_KEY", ""),
            base_url=GROQ_BASE_URL,
            timeout=15.0,  # Groq is fast, shorter timeout
            max_retries=3,
            rate_limit_rps=0.5,  # 30 req/min = 0.5 req/s on free tier
            free_tier=True,
            cost_per_mtok=0.0,  # Free tier
            models=dict(GROQ_DEFAULT_MODELS),
        )
        super().__init__(config)

        self._client = AsyncOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=self.config.api_key,
        )

    @property
    def name(self) -> str:
        return "groq"

    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send request to Groq LPU with automatic fallback models."""
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        # Try models in order: requested -> fallback chain
        models_to_try = [
            model,
            GROQ_DEFAULT_MODELS.get("fast"),
            GROQ_DEFAULT_MODELS.get("reasoning"),
        ]
        # Deduplicate while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        last_error = None
        for m in models_to_try:
            try:
                response = await self._client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.timeout,
                )

                latency_ms = (time.time() - t0) * 1000
                self.record_latency(latency_ms)

                return ProviderResponse(
                    content=response.choices[0].message.content or "",
                    model=m,
                    provider="groq",
                    latency_ms=round(latency_ms, 1),
                    tokens_prompt=response.usage.prompt_tokens if response.usage else 0,
                    tokens_completion=response.usage.completion_tokens if response.usage else 0,
                    cost_usd=0.0,  # Free tier
                )

            except RateLimitError as e:
                last_error = ProviderUnavailable(f"Groq rate limited: {e}")
                await asyncio.sleep(2)  # Wait before retry with next model
                continue

            except APITimeoutError:
                last_error = ProviderTimeout("Groq request timed out")
                continue

            except APIError as e:
                last_error = ProviderError(f"Groq API error: {e}")
                continue

        self._error_count += 1
        raise last_error or ProviderUnavailable("All Groq models failed")

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Streaming inference from Groq."""
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        try:
            stream = await self._client.chat.completions.create(
                model=model,
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
            raise ProviderError(f"Groq streaming failed: {e}")

    async def is_available(self) -> bool:
        """Check if Groq API key is configured and reachable."""
        if not self.config.api_key:
            return False
        try:
            # Quick model list fetch to verify connectivity
            await self._client.models.list()
            return True
        except Exception:
            return False
