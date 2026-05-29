"""Anthropic Claude provider — Claude API via the LLMProvider interface.

Note: Anthropic does not expose an embedding endpoint, so ``embed()`` raises
``NotImplementedError``.  The router will automatically fall back to the
default embedding provider for embed calls when this provider is primary.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)

_MODEL_ALIASES: dict[str, str] = {
    "default": "claude-sonnet-4-20250514",
    "fast": "claude-haiku-3-5-20241022",
    "reasoning": "claude-opus-4-20250514",
}

# ── Lazily import anthropic so the package is optional ────────────────────────

_HAS_ANTHROPIC = False
_anthropic: Any = None


def _ensure_anthropic() -> Any:
    global _HAS_ANTHROPIC, _anthropic
    if not _HAS_ANTHROPIC:
        try:
            import anthropic  # type: ignore[import]

            _anthropic = anthropic
            _HAS_ANTHROPIC = True
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from None
    return _anthropic


class AnthropicProvider:
    """LLM provider backed by Anthropic's Claude API.

    Does **not** support embedding — callers should route embed tasks to
    a different provider (e.g. NIM or OpenAI).
    """

    def __init__(self) -> None:
        self._client: Any = None

    def _make_client(self) -> Any:
        anth = _ensure_anthropic()
        cfg = get_settings()
        return anth.AsyncAnthropic(
            api_key=cfg.anthropic_api_key,
            base_url=cfg.anthropic_base_url or None,
            timeout=cfg.anthropic_timeout,
            max_retries=3,
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _resolve_model(self, model: str | None) -> str:
        if model is None:
            return _MODEL_ALIASES["default"]
        return _MODEL_ALIASES.get(model, model)

    # ── LLMProvider interface ─────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        resolved = self._resolve_model(model)

        # Anthropic uses a different message format — extract system prompt
        system_msg = ""
        filtered_messages: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
            else:
                filtered_messages.append(m)

        kwargs: dict[str, Any] = {
            "model": resolved,
            "messages": filtered_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg

        try:
            resp = await self.client.messages.create(**kwargs)
            return resp.content[0].text if resp.content else ""
        except Exception as exc:
            logger.warning("Anthropic API error: %s", exc)
            raise

    async def json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        # Append JSON instruction to the last user message
        augmented = [dict(m) for m in messages]
        if augmented and            augmented[-1].get("role") == "user":
            augmented[-1]["content"] += (
                "\n\nYou MUST respond with valid JSON only, no surrounding text."
            )

        raw = await self.chat(
            augmented,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        raise NotImplementedError(
            "Anthropic does not provide an embedding API. "
            "Use NIM or OpenAI for embedding tasks."
        )
