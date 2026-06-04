"""
QUANTEX Local Ollama Provider — Self-hosted LLM inference via Ollama.

Provides a local fallback when cloud providers (Groq, NVIDIA NIM, OpenRouter)
are unavailable. Runs on RTX 4050 (6GB VRAM) with:
  - phi-3.5:mini  (2.5 GB VRAM, 45 t/s) → Fast/Urgent tasks
  - qwen2.5:7b-q4 (4.5 GB VRAM, 25 t/s) → Reasoning/Analysis tasks

Architecture:
  Ollama (localhost:11434) ← HTTP API
       │
       ├── phi-3.5:mini     → FAST, CLASSIFY, URGENT
       └── qwen2.5:7b-q4_K_M → REASONING, ANALYSIS

Installation:
    # One-time setup:
    winget install Ollama.Ollama  # or download from ollama.com
    ollama pull phi-3.5:mini
    ollama pull qwen2.5:7b-q4_K_M

Usage:
    provider = LocalOllamaProvider()
    response = await provider.infer("fast", messages)
"""

import os
import json
import time
import logging
from typing import Optional, AsyncGenerator

import aiohttp
from .base import BaseProvider, ProviderResponse, ProviderUnavailable, ProviderTimeout, ProviderError

logger = logging.getLogger("quantex.inference.local_ollama")


class LocalOllamaProvider(BaseProvider):
    """
    Ollama-based local LLM provider.

    Uses local models running via Ollama server (localhost:11434).
    Falls back automatically when cloud providers are unavailable.

    Models:
      - phi-3.5:mini  → 2.5 GB VRAM, 45 t/s, good for fast/classification tasks
      - qwen2.5:7b-q4_K_M → 4.5 GB VRAM, 25 t/s, good for reasoning/analysis

    Requires Ollama to be installed and running.
    """

    OLLAMA_BASE_URL = "http://localhost:11434"

    # Model mapping: task_type -> Ollama model name
    MODEL_MAP = {
        "fast": "phi-3.5:mini",
        "urgent": "phi-3.5:mini",
        "classification": "phi-3.5:mini",
        "reasoning": "qwen2.5:7b-q4_K_M",
        "analysis": "qwen2.5:7b-q4_K_M",
        "coding": "qwen2.5:7b-q4_K_M",
        "embedding": "phi-3.5:mini",  # Fallback, not ideal for embeddings
    }

    def __init__(self, base_url: str = None):
        super().__init__()
        self.base_url = base_url or self.OLLAMA_BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._available: Optional[bool] = None
        self._available_models: list[str] = []

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def is_available(self) -> bool:
        """
        Check if Ollama server is running and has required models.
        Caches result for 60 seconds.
        """
        if self._available is not None:
            return self._available

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags", timeout=3.0) as resp:
                if resp.status != 200:
                    self._available = False
                    return False

                data = await resp.json()
                models = [m["name"] for m in data.get("models", [])]
                self._available_models = models

                # Check if we have at least one required model
                required = set(self.MODEL_MAP.values())
                available = set(models)
                if required & available:
                    self._available = True
                    logger.info(f"Ollama available: {len(models)} models loaded")
                    return True

                logger.warning(f"Ollama running but no required models. Have: {models}")
                self._available = False
                return False

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Ollama not available: {e}")
            self._available = False
            return False

    def _select_model(self, model_key: str) -> str:
        """Select the best available Ollama model for the task."""
        preferred = self.MODEL_MAP.get(model_key, "phi-3.5:mini")

        # Check if preferred model is available
        if preferred in self._available_models:
            return preferred

        # Fallback to any available model
        for m in self.MODEL_MAP.values():
            if m in self._available_models:
                return m

        # Default fallback
        return preferred

    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """
        Send inference request to local Ollama.

        Args:
            model: Task type key ("fast", "reasoning", etc.) or model name
            messages: OpenAI-format messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ProviderResponse with content and metadata

        Raises:
            ProviderUnavailable: Ollama not running
            ProviderTimeout: Request timed out
        """
        if not await self.is_available():
            raise ProviderUnavailable("Ollama not available")

        actual_model = self._select_model(model)
        session = await self._get_session()

        # Convert OpenAI-format messages to Ollama prompt
        prompt = self._messages_to_prompt(messages)
        t0 = time.time()

        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": actual_model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,  # Context window
                    },
                },
                timeout=30.0,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise ProviderError(f"Ollama error {resp.status}: {error_text}")

                data = await resp.json()
                content = data.get("response", "")
                latency_ms = (time.time() - t0) * 1000

                # Approximate token count (4 chars per token for English)
                tokens = len(content) // 4

                return ProviderResponse(
                    content=content,
                    model=actual_model,
                    provider="local_ollama",
                    latency_ms=round(latency_ms, 2),
                    tokens_prompt=len(prompt) // 4,
                    tokens_completion=tokens,
                    cost_usd=0.0,  # Free! Local inference
                )

        except asyncio.TimeoutError:
            raise ProviderTimeout("Ollama request timed out after 30s")
        except aiohttp.ClientError as e:
            raise ProviderUnavailable(f"Ollama connection error: {e}")

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from local Ollama."""
        if not await self.is_available():
            raise ProviderUnavailable("Ollama not available")

        actual_model = self._select_model(model)
        session = await self._get_session()
        prompt = self._messages_to_prompt(messages)

        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": actual_model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=60.0,
            ) as resp:
                if resp.status != 200:
                    raise ProviderError(f"Ollama stream error {resp.status}")

                async for line in resp.content:
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            raise ProviderTimeout("Ollama stream timed out")
        except aiohttp.ClientError as e:
            raise ProviderUnavailable(f"Ollama connection error: {e}")

    async def health(self) -> dict:
        """Get Ollama server health."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags", timeout=3.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return {
                        "provider": "local_ollama",
                        "available": True,
                        "models": models,
                        "model_count": len(models),
                        "latency_ms": 0,
                    }
        except Exception as e:
            pass

        return {
            "provider": "local_ollama",
            "available": False,
            "error": "Ollama server not running at localhost:11434",
            "setup_command": "ollama serve",
        }

    @staticmethod
    def _messages_to_prompt(messages: list) -> str:
        """Convert OpenAI-format messages to a single prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        parts.append("Assistant:")
        return "\n\n".join(parts)

    async def close(self):
        """Cleanup HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
