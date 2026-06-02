"""
Base provider interface for all inference providers.

All providers (Groq, NVIDIA NIM, OpenRouter, Ollama) must implement
this interface for drop-in compatibility with the InferenceRouter.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional


class ProviderError(Exception):
    """Base error for all provider failures."""
    pass


class ProviderUnavailable(ProviderError):
    """Provider is down, rate-limited, or unreachable."""
    pass


class ProviderTimeout(ProviderError):
    """Provider exceeded the timeout threshold."""
    pass


@dataclass
class ProviderResponse:
    """Normalized response from any provider."""
    content: str
    model: str
    provider: str
    latency_ms: float
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: Optional[str] = None


@dataclass
class ProviderConfig:
    """Configuration for a provider instance."""
    api_key: str = ""
    base_url: str = ""
    timeout: float = 30.0
    max_retries: int = 2
    retry_delay: float = 1.0
    models: dict = field(default_factory=dict)
    rate_limit_rps: float = 10.0
    cost_per_mtok: float = 0.0
    free_tier: bool = True


class BaseProvider(ABC):
    """
    Abstract base class for all inference providers.

    Subclasses must implement:
      - infer(): synchronous inference
      - infer_stream(): streaming inference (optional, default raises NotImplemented)
      - is_available(): health check
    """

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()
        self._latency_history: list[float] = []
        self._error_count: int = 0
        self._total_requests: int = 0
        self._last_request_time: float = 0.0

    @property
    def name(self) -> str:
        """Provider name (override in subclass)."""
        return self.__class__.__name__

    @abstractmethod
    async def infer(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> ProviderResponse:
        """
        Send a chat completion request.

        Args:
            model: Model identifier for this provider
            messages: OpenAI-format message list
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            ProviderResponse with content and metadata

        Raises:
            ProviderUnavailable: Provider is down
            ProviderTimeout: Request timed out
            ProviderError: Other errors
        """
        ...

    async def infer_stream(
        self,
        model: str,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming inference. Default implementation raises NotImplementedError.
        Override in providers that support streaming.
        """
        raise NotImplementedError(f"{self.name} does not support streaming")

    async def is_available(self) -> bool:
        """
        Quick health check. Returns True if provider appears operational.

        Default: check if API key is configured.
        Override for deeper checks (e.g., test endpoint call).
        """
        return bool(self.config.api_key)

    def record_latency(self, latency_ms: float):
        """Record a request latency for stats."""
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 1000:
            self._latency_history = self._latency_history[-500:]

    @property
    def avg_latency_ms(self) -> float:
        """Average latency over recent requests."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    @property
    def p95_latency_ms(self) -> float:
        """P95 latency over recent requests."""
        if not self._latency_history:
            return 0.0
        sorted_lat = sorted(self._latency_history)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[idx]

    @property
    def success_rate(self) -> float:
        """Success rate across all requests."""
        if self._total_requests == 0:
            return 1.0
        return 1.0 - (self._error_count / self._total_requests)

    async def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        if self.config.rate_limit_rps > 0:
            min_interval = 1.0 / self.config.rate_limit_rps
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    async def health(self) -> dict:
        """Full health report."""
        return {
            "provider": self.name,
            "available": await self.is_available(),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "success_rate": round(self.success_rate, 3),
            "total_requests": self._total_requests,
            "error_count": self._error_count,
            "free_tier": self.config.free_tier,
            "models": list(self.config.models.keys()),
        }
