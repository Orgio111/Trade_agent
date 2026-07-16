"""Read-only Ollama inventory checks for runtime readiness."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .locality import normalize_local_http_url
from .models import LOCAL_MODEL_BY_ROLE, model_name_matches


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OllamaModel(_StrictModel):
    name: str = Field(min_length=1)
    digest: str = ""
    size_bytes: int = Field(default=0, ge=0)


class OllamaHealth(_StrictModel):
    healthy: bool
    base_url: str
    installed: tuple[OllamaModel, ...] = ()
    missing: tuple[str, ...] = ()
    running: tuple[str, ...] = ()
    detail: str


class LocalOllamaHealthClient:
    """Check the exact local model allowlist without exposing chat inference."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout_seconds: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = normalize_local_http_url(
            base_url,
            service="Ollama",
            default_port=11434,
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )
        self._closed = False

    async def health(self) -> OllamaHealth:
        """Return fail-closed reachability, inventory, and residency evidence."""

        if self._closed:
            return OllamaHealth(
                healthy=False,
                base_url=self.base_url,
                detail="Ollama health client is closed",
            )
        try:
            installed = await self._models("/api/tags")
            running = await self._running_models()
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            return OllamaHealth(
                healthy=False,
                base_url=self.base_url,
                detail=f"local Ollama health check failed: {type(exc).__name__}",
            )
        installed_names = {model.name for model in installed}
        missing = tuple(
            sorted(
                expected
                for expected in set(LOCAL_MODEL_BY_ROLE.values())
                if not any(
                    model_name_matches(actual, expected)
                    for actual in installed_names
                )
            )
        )
        return OllamaHealth(
            healthy=not missing,
            base_url=self.base_url,
            installed=installed,
            missing=missing,
            running=running,
            detail=(
                "all approved local models are installed"
                if not missing
                else "approved local models are missing"
            ),
        )

    async def _models(self, path: str) -> tuple[OllamaModel, ...]:
        response = await self._client.get(f"{self.base_url}{path}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("models"), list
        ):
            raise ValueError("Ollama inventory response is malformed")
        result: list[OllamaModel] = []
        for raw in payload["models"]:
            if not isinstance(raw, Mapping):
                raise TypeError("Ollama model entry must be an object")
            name = raw.get("name") or raw.get("model")
            if not isinstance(name, str) or not name:
                raise ValueError("Ollama model entry has no name")
            digest = raw.get("digest", "")
            size = raw.get("size", 0)
            if not isinstance(digest, str) or isinstance(size, bool) or not isinstance(size, int):
                raise ValueError("Ollama model metadata is malformed")
            result.append(OllamaModel(name=name, digest=digest, size_bytes=size))
        return tuple(sorted(result, key=lambda model: model.name))

    async def _running_models(self) -> tuple[str, ...]:
        try:
            models = await self._models("/api/ps")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ()
            raise
        return tuple(model.name for model in models)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> LocalOllamaHealthClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()
