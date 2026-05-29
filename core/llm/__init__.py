"""Multi-provider LLM abstraction layer.

Provides a unified, provider-agnostic interface for all LLM and embedding
operations across the Trade Agent.  Callers use the three top-level functions
and never touch provider implementations directly.

Usage
-----
    from core.llm import llm_chat, llm_json, llm_embed

    text = await llm_chat([{"role": "user", "content": "Hello"}])
    data = await llm_json([{"role": "user", "content": "Return JSON"}])
    vecs = await llm_embed(["text to embed"])

Multiple providers are supported out of the box:

======== ============== ==========  =========
Provider  Chat / JSON    Embedding   Config key
======== ============== ==========  =========
NIM      ✅             ✅          ``nim_*``
OpenAI   ✅             ✅          ``openai_*``
Anthropic ✅            ❌          ``anthropic_*``
======== ============== ==========  =========

The default provider is selected via the ``LLM_DEFAULT_PROVIDER`` environment
variable (default ``"nim"``).  Callers may override the provider per-call by
passing a ``model`` argument that is matched against known model prefixes
(``gpt-*`` → OpenAI, ``claude-*`` → Anthropic, etc.) via
:func:`core.llm.router.get_chat_provider`.
"""

from __future__ import annotations

import logging
from typing import Any

from core.llm.router import get_chat_provider, get_embed_provider

logger = logging.getLogger(__name__)


async def llm_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    response_format: dict[str, Any] | None = None,
    provider: str | None = None,
) -> str:
    """Single-turn chat completion across providers.

    Parameters
    ----------
    messages:
        OpenAI-format message list.
    model:
        Model name or alias.  If provided, the router maps it to the correct
        provider automatically (e.g. ``claude-sonnet-4-20250514`` →
        Anthropic).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the response.
    response_format:
        Optional structured output hint (e.g. ``{"type": "json_object"}``).
    provider:
        Explicit provider override (``"nim"``, ``"openai"``, ``"anthropic"``).
        Takes precedence over model-based routing.
    """
    if provider:
        from core.llm.router import get_provider as _get_provider
        p = _get_provider(provider)
    else:
        p = get_chat_provider(model)
    return await p.chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )


async def llm_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    provider: str | None = None,
) -> dict[str, Any]:
    """Chat completion returning a parsed JSON dict.

    See :func:`llm_chat` for parameter documentation.
    """
    if provider:
        from core.llm.router import get_provider as _get_provider
        p = _get_provider(provider)
    else:
        p = get_chat_provider(model)
    return await p.json(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def llm_embed(
    texts: list[str],
    *,
    model: str | None = None,
) -> list[list[float]]:
    """Batch text embedding.

    Unlike ``llm_chat``/``llm_json``, this always routes to a provider
    that supports embedding (see :func:`core.llm.router.get_embed_provider`).
    """
    p = get_embed_provider()
    return await p.embed(texts, model=model)


# ── Re-export key symbols for convenience ─────────────────────────────────────
__all__ = [
    "llm_chat",
    "llm_json",
    "llm_embed",
]
