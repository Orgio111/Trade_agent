"""
NVIDIA NIM Inference Provider.

NVIDIA NIM (NVIDIA Inference Microservices) provides optimized
LLM inference on NVIDIA's cloud infrastructure. Uses OpenAI-compatible API.

Strengths:
  - High quality models (Nemotron, Llama, DeepSeek)
  - FP8 optimized inference (NVLink fast)
  - Free credits available for testing
  - Embedding API support

Weaknesses:
  - Medium latency (400-1200ms)
  - Limited free tier after credits exhausted
  - Smaller model selection than OpenRouter
"""

import os
import time
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


NIM_DEFAULT_MODELS = {
    "reasoning": "deepseek-ai/deepseek-v4-flash",
    "analysis": "meta/llama-3.1-70b-instruct",
    "fast": "meta/llama-3.1-8b-instruct",
    "embedding": "nvidia/nv-embedqa-e5-v5",
    "coding": "qwen/qwen3-coder-480b-a35b-instruct",
    "multimodal": "nvidia/nemotron-nano-omni",
}

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# As of June 2026 — NVIDIA free tier provides $5 in initial credits
NIM_FREE_CREDITS = 5.0  # USD
NIM_COST_PER_MTK = 0.90  # $0.90 per million tokens (for most models)


class NvidiaNIMProvider(BaseProvider):
    """
    NVIDIA NIM cloud inference provider.

    Uses NVIDIA's hosted inference microservices for high-quality
    model inference. Supports chat completions, embeddings, and streaming.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("NVIDIA_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        config = ProviderConfig(
            api_key=key,
            base_url=NIM_BASE_URL,
            timeout=30.0,
            max_retries=2,
            rate_limit_rps=5.0,
            free_tier=True,  # Has free credits
            cost_per_mtok=NIM_COST_PER_MTK,
            models=dict(NIM_DEFAULT_MODELS),
        )
        super().__init__(config)

        self._client = AsyncOpenAI(
            base_url=NIM_BASE_URL,
            api_key=self.config.api_key,
        )

    @property
    def name(self) -> str:
        return "nvidia_nim"

    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send request to NVIDIA NIM."""
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        try:
            # Resolve model name (strip nvidia/ prefix if needed)
            actual_model = model.replace("nvidia/", "") if model.startswith("nvidia/") else model

            response = await self._client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.config.timeout,
            )

            latency_ms = (time.time() - t0) * 1000
            self.record_latency(latency_ms)

            usage = response.usage
            total_tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
            cost = (total_tokens / 1_000_000) * self.config.cost_per_mtok if total_tokens > 0 else 0.0

            return ProviderResponse(
                content=response.choices[0].message.content or "",
                model=actual_model,
                provider="nvidia_nim",
                latency_ms=round(latency_ms, 1),
                tokens_prompt=usage.prompt_tokens if usage else 0,
                tokens_completion=usage.completion_tokens if usage else 0,
                cost_usd=round(cost, 6),
            )

        except RateLimitError as e:
            self._error_count += 1
            raise ProviderUnavailable(f"NVIDIA NIM rate limited: {e}")
        except APITimeoutError:
            self._error_count += 1
            raise ProviderTimeout("NVIDIA NIM request timed out")
        except APIError as e:
            self._error_count += 1
            raise ProviderError(f"NVIDIA NIM API error: {e}")

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Streaming inference from NVIDIA NIM."""
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        try:
            actual_model = model.replace("nvidia/", "") if model.startswith("nvidia/") else model

            stream = await self._client.chat.completions.create(
                model=actual_model,
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
            raise ProviderError(f"NVIDIA NIM streaming failed: {e}")

    async def batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text list."""
        try:
            response = await self._client.embeddings.create(
                model=NIM_DEFAULT_MODELS["embedding"],
                input=texts,
                encoding_format="float",
            )
            return [e.embedding for e in response.data]
        except Exception as e:
            raise ProviderError(f"NVIDIA NIM embedding failed: {e}")

    async def is_available(self) -> bool:
        """Check if NIM API key is configured."""
        return bool(self.config.api_key)
