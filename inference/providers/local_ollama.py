"""
QUANTEX Local Ollama Provider — Local inference via Ollama.

Connects to a local Ollama server (http://localhost:11434) to run
quantized LLMs on local GPU/CPU. Perfect for RTX 4050 (6GB VRAM)
where 7B-8B quantized models fit comfortably.

Supported models (verified on RTX 4050 6GB):
  - qwen3:8b         ✅ ~4.5GB VRAM — best for reasoning/analysis
  - deepseek-r1:8b   ✅ ~4.7GB VRAM — strong reasoning
  - qwen2.5:3b       ✅ ~2GB VRAM   — very fast, low latency
  - phi3:mini        ✅ ~2.5GB VRAM — lightweight, fast

Requirements:
  - Ollama installed and running (http://localhost:11434)
  - Models pulled via `ollama pull <model>`

Usage:
    from inference.providers import LocalOllamaProvider

    provider = LocalOllamaProvider()
    result = await provider.infer("qwen3:8b", messages=[...])
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

import aiohttp

from .base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    ProviderUnavailable,
    ProviderTimeout,
    ProviderError,
)

logger = logging.getLogger("quantex.inference.ollama")

# ── Ollama API endpoints ────────────────────────────────────

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = "/api/chat"
OLLAMA_TAGS_ENDPOINT = "/api/tags"
OLLAMA_GENERATE_ENDPOINT = "/api/generate"

# Model configs with VRAM estimates for RTX 4050 6GB
OLLAMA_DEFAULT_MODELS: dict[str, dict] = {
    "reasoning": {
        "model": "qwen3:8b",
        "vram_gb": 4.5,
        "note": "Best overall for trading analysis on 6GB GPU",
    },
    "analysis": {
        "model": "qwen3:8b",
        "vram_gb": 4.5,
        "note": "Strong reasoning for market structure analysis",
    },
    "fast": {
        "model": "qwen2.5:3b",
        "vram_gb": 2.0,
        "note": "Ultra-fast, minimal latency for classifications",
    },
    "classification": {
        "model": "phi3:mini",
        "vram_gb": 2.5,
        "note": "Lightweight, good for text classification",
    },
    "deep_reasoning": {
        "model": "deepseek-r1:8b",
        "vram_gb": 4.7,
        "note": "Strong chain-of-thought reasoning",
    },
}


class LocalOllamaProvider(BaseProvider):
    """
    Local Ollama inference provider.

    Connects to a local Ollama server to run quantized models
    on local GPU. Zero API cost.

    Parameters
    ----------
    base_url:
        Ollama server URL (default: http://localhost:11434)
    models:
        Optional model config override dict
    timeout:
        Request timeout in seconds (default: 120.0 for local inference)
    """

    def __init__(
        self,
        base_url: str = "",
        models: Optional[dict] = None,
        timeout: float = 120.0,
    ):
        url = base_url or os.getenv("OLLAMA_URL", "") or OLLAMA_DEFAULT_URL

        config = ProviderConfig(
            api_key="",  # Ollama doesn't need an API key
            base_url=url,
            timeout=timeout,
            max_retries=2,
            rate_limit_rps=5.0,
            free_tier=True,
            cost_per_mtok=0.0,
            models=models or {k: v["model"] for k, v in OLLAMA_DEFAULT_MODELS.items()},
        )
        super().__init__(config)
        # Create session at init (consistent with other providers)
        self._session = aiohttp.ClientSession(
            base_url=self.config.base_url,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
        )
        self._installed_models: list[str] = []

    @property
    def name(self) -> str:
        return "local_ollama"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get the aiohttp session (always available after init)."""
        return self._session

    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """Send a chat completion request to Ollama.

        Args:
            model: Model name (e.g., "qwen3:8b", "deepseek-r1:8b")
            messages: OpenAI-format message list
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            ProviderResponse with content and latency.
        """
        await self._rate_limit_wait()
        t0 = time.time()
        self._total_requests += 1

        # Resolve model shortcut
        actual_model = self.config.models.get(model, model)

        # Build Ollama chat request
        payload = {
            "model": actual_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }

        try:
            session = await self._get_session()
            async with session.post(OLLAMA_CHAT_ENDPOINT, json=payload) as resp:
                latency_ms = (time.time() - t0) * 1000

                if resp.status == 404:
                    raise ProviderUnavailable(
                        f"Model '{actual_model}' not found. "
                        f"Pull it: ollama pull {actual_model}"
                    )
                if resp.status == 503:
                    raise ProviderUnavailable(
                        "Ollama is loading the model. "
                        "First inference is slow (~5-30s for model load)."
                    )
                if resp.status != 200:
                    error_text = await resp.text()
                    raise ProviderError(f"Ollama error {resp.status}: {error_text}")

                data = await resp.json()
                content = data.get("message", {}).get("content", "")

                # Parse Ollama metrics if available
                tokens_per_sec = data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001)
                tokens_prompt = data.get("prompt_eval_count", 0)
                tokens_completion = data.get("eval_count", 0)

                self.record_latency(latency_ms)

                logger.debug(
                    "Ollama inference OK: model=%s latency=%.0fms tokens=%d+%d (%.1f tok/s)",
                    actual_model, latency_ms, tokens_prompt, tokens_completion, tokens_per_sec,
                )

                return ProviderResponse(
                    content=content,
                    model=actual_model,
                    provider="local_ollama",
                    latency_ms=round(latency_ms, 1),
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    cost_usd=0.0,
                )

        except aiohttp.ClientConnectorError:
            self._error_count += 1
            raise ProviderUnavailable(
                f"Ollama server not reachable at {self.config.base_url}. "
                "Start with: ollama serve"
            )
        except aiohttp.ServerTimeoutError:
            self._error_count += 1
            raise ProviderTimeout(f"Ollama request timed out after {self.config.timeout}s")

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Streaming inference from Ollama.

        Yields tokens as they are generated by the local model.
        Useful for real-time display of reasoning.
        """
        await self._rate_limit_wait()
        self._total_requests += 1

        actual_model = self.config.models.get(model, model)

        payload = {
            "model": actual_model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True,
        }

        try:
            session = await self._get_session()
            async with session.post(OLLAMA_CHAT_ENDPOINT, json=payload) as resp:
                if resp.status != 200:
                    raise ProviderError(f"Ollama stream error {resp.status}")

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("done"):
                                break
                            if content := data.get("message", {}).get("content"):
                                yield content
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientConnectorError:
            raise ProviderUnavailable("Ollama server not reachable")
        except Exception as e:
            raise ProviderError(f"Ollama stream failed: {e}")

    async def is_available(self) -> bool:
        """Check if Ollama server is reachable and has models."""
        try:
            session = await self._get_session()
            async with session.get(OLLAMA_TAGS_ENDPOINT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._installed_models = [m["name"] for m in data.get("models", [])]
                    return len(self._installed_models) > 0
                return False
        except Exception:
            return False

    async def health(self) -> dict:
        """Extended health report with installed models."""
        health = await super().health()
        health["base_url"] = self.config.base_url

        # Check installed models
        try:
            session = await self._get_session()
            async with session.get(OLLAMA_TAGS_ENDPOINT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    health["installed_models"] = [
                        {
                            "name": m["name"],
                            "size_gb": round(m.get("size", 0) / 1e9, 2),
                            "modified_at": m.get("modified_at", ""),
                        }
                        for m in data.get("models", [])
                    ]
                    health["model_count"] = len(health["installed_models"])
                else:
                    health["installed_models"] = []
        except Exception:
            health["installed_models"] = []
            health["error"] = "Could not fetch model list"

        return health

    async def list_models(self) -> list[dict]:
        """Get list of installed Ollama models with details."""
        try:
            session = await self._get_session()
            async with session.get(OLLAMA_TAGS_ENDPOINT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        {
                            "name": m["name"],
                            "size_gb": round(m.get("size", 0) / 1e9, 2),
                            "digest": m.get("digest", "")[:12],
                            "modified_at": m.get("modified_at", ""),
                        }
                        for m in data.get("models", [])
                    ]
                return []
        except Exception:
            return []

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
