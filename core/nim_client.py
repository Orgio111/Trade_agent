"""NVIDIA NIM client — OpenAI-compatible wrapper with async support."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

from core.config import get_settings

logger = logging.getLogger(__name__)


def _make_client() -> AsyncOpenAI:
    cfg = get_settings()
    return AsyncOpenAI(
        api_key=cfg.nim_api_key,
        base_url=cfg.nim_base_url,
        timeout=cfg.nim_timeout,
        max_retries=3,
    )


_client: AsyncOpenAI | None = None


def get_nim_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = _make_client()
    return _client


async def nim_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    response_format: dict | None = None,
) -> str:
    """Single-turn chat completion via NIM."""
    cfg = get_settings()
    client = get_nim_client()
    kwargs: dict[str, Any] = {
        "model": model or cfg.nim_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    try:
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except openai.RateLimitError:
        logger.warning("NIM rate limit — sleeping 10s")
        await asyncio.sleep(10)
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


async def nim_json(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """Chat completion returning parsed JSON."""
    raw = await nim_chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract first {...} block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])


async def nim_embed(texts: list[str]) -> list[list[float]]:
    """Batch embedding via NIM embedding endpoint."""
    cfg = get_settings()
    client = get_nim_client()
    resp = await client.embeddings.create(
        model=cfg.nim_embed_model,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in resp.data]
