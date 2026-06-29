"""
QUANTEX vLLM Inference Provider — Self-hosted GPU inference via vLLM.

Provides a BaseProvider-compatible interface for self-hosted vLLM servers.
vLLM runs locally on GPU hardware, serving models like Qwen2.5, Mistral 7B,
DeepSeek, and Llama with optimized batching and KV-cache management.

Architecture:
  ┌──────────────────┐          ┌───────────────┐          ┌──────────┐
  │  InferenceRouter │  ──────→ │  vLLM Server  │  ──────→ │  GPU     │
  │  (task routing)  │          │  (localhost)  │          │  (CUDA)  │
  └──────────────────┘          └───────────────┘          └──────────┘

Requirements:
  - vLLM server running locally (pip install vllm)
  - GPU with sufficient VRAM (16GB+ for 7B models, 80GB+ for 70B)
  - OpenAI-compatible API endpoint (default: http://localhost:8000/v1)

Usage:
    provider = vLLMProvider(base_url="http://localhost:8000/v1")
    result = await provider.infer(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
    )

Setup:
    # Start vLLM server:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.90
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError

from .base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    ProviderUnavailable,
    ProviderTimeout,
    ProviderError,
)

logger = logging.getLogger("quantex.inference.vllm")

# ── Default Model Configuration ──────────────────────────────

VLLM_DEFAULT_MODELS = {
    "reasoning": "Qwen/Qwen2.5-7B-Instruct",
    "analysis": "Qwen/Qwen2.5-7B-Instruct",
    "fast": "Qwen/Qwen2.5-1.5B-Instruct",
    "coding": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "classification": "Qwen/Qwen2.5-1.5B-Instruct",
    "embedding": "BAAI/bge-small-en-v1.5",
}

VLLM_DEFAULT_URL = "http://localhost:8000/v1"
VLLM_COST_PER_MTK = 0.0  # Self-hosted = free inference (power costs only)


class vLLMProvider(BaseProvider):
    """
    Self-hosted vLLM inference provider.

    Connects to a local vLLM server via OpenAI-compatible API.
    Latency depends on GPU hardware and model size:
      - RTX 4090 (24GB): ~50-150ms for 7B model
      - A100 (80GB): ~20-80ms for 7B, ~100-300ms for 70B
      - H100: ~15-50ms for 7B, ~50-200ms for 70B

    Parameters
    ----------
    base_url:
        vLLM server URL (default: http://localhost:8000/v1)
    api_key:
        Optional API key (not used by default vLLM, but configurable)
    models:
        Optional model override dict mapping task types to model names
    timeout:
        Request timeout in seconds (default: 60.0 for local GPU)
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        models: Optional[dict[str, str]] = None,
        timeout: float = 60.0,
    ):
        url = base_url or os.getenv("VLLM_URL", "") or VLLM_DEFAULT_URL
        key = api_key or os.getenv("VLLM_API_KEY", "")

        config = ProviderConfig(
            api_key=key,
            base_url=url,
            timeout=timeout,
            max_retries=2,
            rate_limit_rps=20.0,  # vLLM can handle high concurrency
            free_tier=True,       # Self-hosted = free
            cost_per_mtok=VLLM_COST_PER_MTK,
            models=models or dict(VLLM_DEFAULT_MODELS),
        )
        super().__init__(config)

        self._client = AsyncOpenAI(
            base_url=url,
            api_key=key or "not-needed",
        )

    @property
    def name(self) -> str:
        return "vllm"

    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send a request to the local vLLM server.

        Args:
            model: Model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
            messages: OpenAI-format message list
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            ProviderResponse with content and latency.

        Raises:
            ProviderUnavailable: vLLM server is not running
            ProviderTimeout: Request timed out
            ProviderError: Other errors
        """
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        # Resolve model shortcut
        actual_model = self.config.models.get(model, model)

        try:
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
            tokens_prompt = usage.prompt_tokens if usage else 0
            tokens_completion = usage.completion_tokens if usage else 0

            logger.debug(
                "vLLM inference OK: model=%s latency=%.0fms tokens=%d+%d",
                actual_model, latency_ms, tokens_prompt, tokens_completion,
            )

            return ProviderResponse(
                content=response.choices[0].message.content or "",
                model=actual_model,
                provider="vllm",
                latency_ms=round(latency_ms, 1),
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                cost_usd=0.0,  # Self-hosted = free
            )

        except APITimeoutError:
            self._error_count += 1
            raise ProviderTimeout(
                f"vLLM request timed out after {self.config.timeout}s. "
                "Check if the vLLM server is running and the GPU is not overloaded."
            )
        except APIError as e:
            self._error_count += 1
            if "Connection refused" in str(e) or "Cannot connect" in str(e):
                raise ProviderUnavailable(
                    f"vLLM server not reachable at {self.config.base_url}. "
                    "Start with: python -m vllm.entrypoints.openai.api_server ..."
                )
            raise ProviderError(f"vLLM API error: {e}")
        except Exception as e:
            self._error_count += 1
            raise ProviderError(f"vLLM unexpected error: {e}")

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Streaming inference from vLLM.

        Yields tokens as they are generated, with minimal latency.
        Useful for real-time trading decisions where partial results
        can be acted upon incrementally.
        """
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        actual_model = self.config.models.get(model, model)

        try:
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

        except APITimeoutError:
            self._error_count += 1
            raise ProviderTimeout(f"vLLM stream timed out")
        except APIError as e:
            self._error_count += 1
            raise ProviderError(f"vLLM stream failed: {e}")

    async def is_available(self) -> bool:
        """Check if the vLLM server is reachable.

        Sends a lightweight model list request to verify the server
        is running and healthy.
        """
        try:
            response = await self._client.models.list()
            return len(response.data) > 0
        except Exception:
            return False

    async def health(self) -> dict:
        """Extended health report including GPU info if available."""
        health = await super().health()
        health["base_url"] = self.config.base_url
        health["models_available"] = list(self.config.models.values())

        # Try to get actual model list from vLLM server
        try:
            response = await self._client.models.list()
            health["server_models"] = [m.id for m in response.data]
        except Exception:
            health["server_models"] = []

        return health

    async def get_model_config(self, model_name: str) -> dict:
        """Get vLLM model configuration.

        Returns information about the model's max context length,
        supported parameters, etc.

        Args:
            model_name: Name of the deployed model

        Returns:
            Dict with model configuration.
        """
        try:
            response = await self._client.models.retrieve(model_name)
            return {
                "id": response.id,
                "created": response.created,
                "owned_by": response.owned_by,
            }
        except Exception as e:
            return {"error": str(e), "id": model_name}
