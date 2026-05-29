"""Abstract provider interface for LLM and embedding providers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that every LLM/embedding provider must implement.

    Providers must implement three operations:
      - chat: free-form text completion
      - json: structured JSON completion (typically via response_format)
      - embed: text-to-vector embedding
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Single-turn chat completion.

        Parameters
        ----------
        messages:
            OpenAI-format message list ``[{"role": "system"|"user"|"assistant", "content": ...}]``.
        model:
            Model identifier override.  ``None`` = use the provider default.
        temperature:
            Sampling temperature (0.0 = deterministic).
        max_tokens:
            Maximum tokens in the response.
        response_format:
            Optional structured output format hint (e.g. ``{"type": "json_object"}``).
        """
        ...

    async def json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Chat completion with JSON response (parsed).

        Default implementation calls ``self.chat`` with ``response_format``
        set to ``{"type": "json_object"}`` and parses the result.
        """
        ...

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Batch text embedding.

        Parameters
        ----------
        texts:
            List of input strings to embed.
        model:
            Embedding model override.  ``None`` = use the provider default.
        """
        ...
