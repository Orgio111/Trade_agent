"""Provider router — selects the right provider by model name, task type, or config.

Typical usage
-------------
    from core.llm.router import get_provider

    # Explicit model → provider resolution
    provider = get_provider(model="gpt-4o")

    # Default provider (from config)
    provider = get_provider()

The router maintains a singleton registry of provider instances and
resolves the correct backend for each request.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)

# ── Provider registry (lazy) ──────────────────────────────────────────────────

_PROVIDERS: dict[str, Any] = {}
"""Cache of instantiated providers keyed by short name (``"nim"``, ``"openai"``, …)."""


def _load_provider(name: str) -> Any:
    """Import and instantiate a provider by its short name.

    Uses lazy import so unused providers don't incur import overhead.
    """
    if name == "nim":
        from core.llm.providers.nim import NIMProvider

        return NIMProvider()
    elif name == "openai":
        from core.llm.providers.openai import OpenAIProvider

        return OpenAIProvider()
    elif name == "anthropic":
        from core.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    else:
        raise ValueError(f"Unknown provider: {name!r}.  Valid: nim, openai, anthropic")


def get_provider(name: str | None = None) -> Any:
    """Return a provider instance by its short name.

    Parameters
    ----------
    name:
        One of ``"nim"``, ``"openai"``, ``"anthropic"``.  If ``None``, the
        default provider configured in :attr:`Settings.llm_default_provider`
        is used.
    """
    if name is None:
        name = get_settings().llm_default_provider

    if name not in _PROVIDERS:
        _PROVIDERS[name] = _load_provider(name)
    return _PROVIDERS[name]


def get_chat_provider(model: str | None = None) -> Any:
    """Return the appropriate provider for a given model name.

    This inspects the model name (or config) to map it to a provider:

    =====================  ============
    Model starts with       Provider
    =====================  ============
    ``gpt-``, ``o3-``, ``o4-``, ``text-embedding-`` → OpenAI
    ``claude-``            → Anthropic
    ``meta/``, ``nvidia/``, ``mistralai/``, ``google/`` → NIM
    anything else          → Default provider from config
    =====================  ============
    """
    if model is None:
        return get_provider()

    model_lower = model.lower()

    if model_lower.startswith(("gpt-", "o3-", "o4-", "text-embedding-")):
        return get_provider("openai")
    elif model_lower.startswith("claude-"):
        return get_provider("anthropic")
    elif model_lower.startswith(("meta/", "nvidia/", "mistralai/", "google/")):
        return get_provider("nim")

    # Unknown model prefix — use default
    return get_provider()


def get_embed_provider() -> Any:
    """Return the provider suitable for embedding tasks.

    Falls back in priority order:
      1. Default provider (from config)
      2. NIM (if default doesn't support embed)
      3. OpenAI (last resort)
    """
    cfg = get_settings()
    default = cfg.llm_default_provider

    # If the default supports embed, use it
    if default != "anthropic":
        return get_provider(default)

    # Anthropic doesn't support embed — try NIM, then OpenAI
    logger.info("Default provider is Anthropic (no embed) — trying NIM for embedding")
    try:
        provider = get_provider("nim")
        # Quick sanity check
        return provider
    except Exception:
        logger.info("NIM not available for embedding — falling back to OpenAI")
        return get_provider("openai")


def get_all_providers() -> dict[str, Any]:
    """Return all registered provider instances, loading them if necessary."""
    for name in ("nim", "openai", "anthropic"):
        if name not in _PROVIDERS:
            _PROVIDERS[name] = _load_provider(name)
    return dict(_PROVIDERS)


def reset_providers() -> None:
    """Clear the provider cache (useful for testing or config reload)."""
    _PROVIDERS.clear()
