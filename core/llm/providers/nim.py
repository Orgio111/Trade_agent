"""NVIDIA NIM provider — OpenAI-compatible wrapper.

Ported from ``core/nim_client.py`` to the multi-provider ``LLMProvider`` interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

from core.config import get_settings

logger = logging.getLogger(__name__)


class NIMProvider:
    """LLM + embedding provider backed by NVIDIA NIM (OpenAI-compatible API).

    Reads configuration (API key, base URL, model names, timeout) from
    :class:`core.config.Settings`.
    """

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    # ── Client lifecycle ──────────────────────────────────────────────────────

    def _make_client(self) -> AsyncOpenAI:
        cfg = get_settings()
        return AsyncOpenAI(
            api_key=cfg.nim_api_key,
            base_url=cfg.nim_base_url,
            timeout=cfg.nim_timeout,
            max_retries=3,
        )

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = self._make_client()
        return self._client

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
        cfg = get_settings()
        kwargs: dict[str, Any] = {
            "model": model or cfg.nim_model,
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
            logger.warning("NIM rate limit — sleeping 10s")
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
            # Fallback: extract first { ... } block
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        cfg = get_settings()
        resp = await self.client.embeddings.create(
            model=model or cfg.nim_embed_model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in resp.data]
