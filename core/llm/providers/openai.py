"""OpenAI provider — direct OpenAI API calls."""

from __future__ import annotations

import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

from core.config import get_settings

logger = logging.getLogger(__name__)

_MODEL_ALIASES: dict[str, str] = {
    "default": "gpt-4o",
    "fast": "gpt-4o-mini",
    "reasoning": "o3-mini",
    "embed": "text-embedding-3-small",
}


class OpenAIProvider:
    """LLM + embedding provider backed by the OpenAI API.

    Reads configuration from ``openai_*`` fields in :class:`core.config.Settings`.
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _make_client(self) -> AsyncOpenAI:
        cfg = get_settings()
        return AsyncOpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url or None,
            timeout=cfg.openai_timeout,
            max_retries=3,
        )

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _resolve_model(self, model: str | None) -> str:
        """Resolve shorthand aliases to full model names."""
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
        kwargs: dict[str, Any] = {
            "model": self._resolve_model(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            resp = await self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except openai.RateLimitError:
            logger.warning("OpenAI rate limit — sleeping 10s")
            import asyncio
            await asyncio.sleep(10)
            resp = await self.client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

    async def json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        raw = await self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
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
        resp = await self.client.embeddings.create(
            model=self._resolve_model(model or "embed"),
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in resp.data]
