"""Dependency-free async ports for local embedding and vector storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .models import (
    ComponentHealth,
    EmbeddingSpec,
    ModelIdentity,
    Scalar,
    SearchHit,
    StoredRecord,
    VectorRecord,
)


class AsyncEmbedder(Protocol):
    """Local embedding boundary; implementations must fail closed."""

    @property
    def spec(self) -> EmbeddingSpec: ...

    async def identity(self) -> ModelIdentity: ...

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...

    async def health(self) -> ComponentHealth: ...

    async def aclose(self) -> None: ...


class AsyncVectorStore(Protocol):
    """Vector database boundary with caller-supplied embeddings only."""

    async def ensure_namespace(self, metadata: Mapping[str, Scalar]) -> None: ...

    async def require_namespace(self, metadata: Mapping[str, Scalar]) -> None: ...

    async def inventory(self) -> dict[str, StoredRecord]: ...

    async def namespace_metadata(self) -> Mapping[str, Scalar]: ...

    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...

    async def delete(self, ids: Sequence[str]) -> int: ...

    async def activate(self, metadata: Mapping[str, Scalar]) -> None: ...

    async def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Scalar] | None = None,
    ) -> list[SearchHit]: ...

    async def health(self) -> ComponentHealth: ...

    async def aclose(self) -> None: ...
