"""Loopback-only Ollama JSON client for typed trade proposals."""

from __future__ import annotations

from typing import Any

import httpx

from packages.local_ai import normalize_local_http_url


_ALLOWED_DOCKER_HOST = "http://host.docker.internal:11434"


class OllamaCandidateClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 30,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise ValueError("timeout_seconds must be in (0, 120]")
        stripped = base_url.rstrip("/")
        self._base_url = (
            stripped
            if stripped == _ALLOWED_DOCKER_HOST
            else normalize_local_http_url(
                base_url,
                service="Ollama",
                default_port=11434,
            )
        )
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds), trust_env=False
        )
        self._owns_client = client is None

    async def complete(self, prompt: str) -> str:
        if not prompt or len(prompt) > 8_000:
            raise ValueError("candidate prompt must contain 1..8000 characters")
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": "qwen3:8b",
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "stream": False,
                    "keep_alive": "5m",
                    "options": {"temperature": 0, "num_predict": 512},
                },
            )
            response.raise_for_status()
            body: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError("local Ollama candidate request failed") from exc
        try:
            model = body["model"]
            done = body["done"]
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama candidate protocol mismatch") from exc
        if model != "qwen3:8b" or done is not True or not isinstance(content, str):
            raise RuntimeError("Ollama candidate protocol mismatch")
        return content

    async def model_digest(self) -> str:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            models = response.json()["models"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("local Ollama model inventory failed") from exc
        for model in models:
            if model.get("name") == "qwen3:8b":
                digest = model.get("digest")
                if (
                    isinstance(digest, str)
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest)
                ):
                    return digest
                raise RuntimeError("approved Ollama model digest is invalid")
        raise RuntimeError("approved Ollama model is not installed")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
