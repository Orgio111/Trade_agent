"""Fail-closed async adapter for the loopback Ollama embedding API."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence

import httpx

from .errors import EmbeddingError
from .locality import normalize_loopback_http_url
from .models import ComponentHealth, EmbeddingSpec, ModelIdentity


_TRANSIENT_STATUS = frozenset({429, 502, 503, 504})


def validate_loopback_url(value: str) -> str:
    """Allow only unauthenticated local HTTP endpoints."""

    try:
        return normalize_loopback_http_url(
            value,
            service="Ollama",
            default_port=11434,
        )
    except ValueError as exc:
        raise EmbeddingError(str(exc)) from exc


class OllamaEmbedder:
    """Batch embeddings from local Ollama with no semantic fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not model.strip():
            raise EmbeddingError("embedding model cannot be empty")
        if dimension < 1:
            raise EmbeddingError("embedding dimension must be positive")
        self._base_url = validate_loopback_url(base_url)
        self._spec = EmbeddingSpec(model=model, dimension=dimension)
        self._max_retries = max(0, max_retries)
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(1)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        )
        self._closed = False

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def _request(
        self, endpoint: str, *, json: dict | None = None
    ) -> httpx.Response:
        if self._closed:
            raise EmbeddingError("Ollama embedder is closed")
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    "POST" if json is not None else "GET", endpoint, json=json
                )
            except httpx.RequestError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise EmbeddingError("local Ollama is unavailable") from exc
            else:
                if response.status_code not in _TRANSIENT_STATUS:
                    if response.is_error:
                        raise EmbeddingError(
                            f"Ollama {endpoint} failed with HTTP {response.status_code}"
                        )
                    return response
                last_error = EmbeddingError(
                    f"Ollama {endpoint} transient HTTP {response.status_code}"
                )
                if attempt >= self._max_retries:
                    raise last_error
            await self._sleep(min(0.1 * (2**attempt), 1.0))
        raise EmbeddingError("local Ollama request failed") from last_error

    async def identity(self) -> ModelIdentity:
        """Resolve the requested alias to the exact digest installed locally."""

        async with self._semaphore:
            response = await self._request("/api/tags")
        try:
            payload = response.json()
            models = payload["models"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Ollama model inventory is malformed") from exc
        if not isinstance(models, list):
            raise EmbeddingError("Ollama model inventory is malformed")

        requested = self._spec.model
        aliases = {requested, f"{requested}:latest"}
        for item in models:
            if not isinstance(item, dict):
                continue
            resolved = item.get("name") or item.get("model")
            if resolved not in aliases:
                continue
            digest = item.get("digest")
            if not isinstance(digest, str) or len(digest.strip()) < 12:
                raise EmbeddingError("Ollama model digest is missing or invalid")
            return ModelIdentity(
                requested_model=requested,
                resolved_model=resolved,
                digest=digest,
                dimension=self._spec.dimension,
            )
        raise EmbeddingError(
            f"required local Ollama model is not installed: {requested}"
        )

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed a true batch and validate every vector before returning."""

        batch = list(texts)
        if not batch or any(not isinstance(text, str) or not text for text in batch):
            raise EmbeddingError("embedding input must be a non-empty sequence of text")
        async with self._semaphore:
            response = await self._request(
                "/api/embed",
                json={
                    "model": self._spec.model,
                    "input": batch,
                    "truncate": False,
                },
            )
        try:
            payload = response.json()
            vectors = payload["embeddings"]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("Ollama embedding response is malformed") from exc
        response_model = payload.get("model")
        if response_model not in {self._spec.model, f"{self._spec.model}:latest"}:
            raise EmbeddingError("Ollama returned embeddings from an unexpected model")
        if not isinstance(vectors, list) or len(vectors) != len(batch):
            raise EmbeddingError(
                "Ollama embedding count does not match the input batch"
            )

        validated: list[tuple[float, ...]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self._spec.dimension:
                raise EmbeddingError(
                    f"embedding dimension drift: expected {self._spec.dimension}"
                )
            values: list[float] = []
            for item in vector:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise EmbeddingError("embedding contains a non-numeric value")
                value = float(item)
                if not math.isfinite(value):
                    raise EmbeddingError("embedding contains NaN or infinity")
                values.append(value)
            if not any(value != 0.0 for value in values):
                raise EmbeddingError("Ollama returned a zero vector")
            validated.append(tuple(values))
        return validated

    async def health(self) -> ComponentHealth:
        try:
            identity = await self.identity()
        except EmbeddingError as exc:
            return ComponentHealth(healthy=False, detail=str(exc))
        return ComponentHealth(
            healthy=True,
            detail=f"{identity.resolved_model}@{identity.digest[:12]}",
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()
