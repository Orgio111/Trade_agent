"""Backward-compatible NIM client — delegates to ``core.llm`` facade.

New code should import directly from ``core.llm``:

    from core.llm import llm_chat, llm_json, llm_embed

This module re-exports the legacy ``nim_chat``, ``nim_json``, and ``nim_embed``
names so that existing imports continue to work during the migration.
"""

from __future__ import annotations

import warnings
from typing import Any

from core.config import get_settings
from core.llm import llm_chat, llm_json, llm_embed
from core.llm.router import get_provider

__all__ = ["get_nim_client", "nim_chat", "nim_json", "nim_embed"]


def get_nim_client() -> Any:
    """Return the underlying NIM provider's OpenAI-compatible client.

    Deprecated — prefer using the ``core.llm`` facade directly.
    """
    warnings.warn(
        "get_nim_client() is deprecated. Use core.llm.llm_chat / llm_json / llm_embed instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from core.llm.providers.nim import NIMProvider

    provider = get_provider("nim")
    if isinstance(provider, NIMProvider):
        return provider.client
    raise RuntimeError("NIM provider not loaded")


async def nim_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    response_format: dict | None = None,
) -> str:
    """Deprecated alias for ``llm_chat`` — delegates to the NIM provider."""
    # If model is not specified, pass None to use the provider default
    return await llm_chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        provider="nim",
    )


async def nim_json(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Deprecated alias for ``llm_json`` — delegates to the NIM provider."""
    return await llm_json(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider="nim",
    )


async def nim_embed(texts: list[str]) -> list[list[float]]:
    """Deprecated alias for ``llm_embed`` — delegates to the NIM provider."""
    return await llm_embed(texts)
