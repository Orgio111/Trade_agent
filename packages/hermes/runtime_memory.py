"""Local Chroma projection for durable Obsidian journal events."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from packages.knowledge.models import (
    ComponentHealth,
    Scalar,
    SearchHit,
    VectorRecord,
    json_scalar_mapping,
)
from packages.knowledge.ports import AsyncEmbedder, AsyncVectorStore

from .models import JournalEvent, JournalReceipt


RUNTIME_MEMORY_COLLECTION = "trade-agent-runtime-memory-v1"
RUNTIME_MEMORY_SCHEMA = "hermes-runtime-memory-v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeMemoryHit(_StrictModel):
    """Retrieval result that retains its canonical Markdown provenance."""

    id: str
    document: str
    distance: float = Field(ge=0)
    metadata: Mapping[str, Scalar]


class RuntimeMemoryQuery(_StrictModel):
    """Bounded query accepted by the local control-plane API."""

    query: str = Field(min_length=1, max_length=8_000)
    top_k: int = Field(default=5, ge=1, le=25)
    filters: Mapping[str, Scalar] | None = None


class RuntimeMemoryQueryResult(_StrictModel):
    query: str
    hits: tuple[RuntimeMemoryHit, ...]


class RuntimeIndexReceipt(_StrictModel):
    event_id: str
    record_id: str
    content_sha256: str
    source_path: str
    indexed_at: datetime


def _event_document(event: JournalEvent) -> str:
    sources = "\n".join(f"- {source}" for source in event.source_refs)
    source_section = f"\n\n## Sources\n{sources}" if sources else ""
    return (
        f"# {event.title}\n\n"
        f"Event kind: {event.kind.value}\n"
        f"Agent: {event.agent_id}\n"
        f"Occurred at: {event.occurred_at.isoformat()}\n\n"
        f"{event.body}{source_section}\n"
    )


class RuntimeMemoryIndex:
    """Project journal events into an isolated, rebuildable Chroma namespace."""

    def __init__(
        self,
        *,
        embedder: AsyncEmbedder,
        store: AsyncVectorStore,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._initialize_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            identity = await self._embedder.identity()
            await self._store.ensure_namespace(
                {
                    "schema": RUNTIME_MEMORY_SCHEMA,
                    "canonical_source": "obsidian-markdown",
                    "embedding_model": identity.resolved_model,
                    "embedding_model_digest": identity.digest,
                    "embedding_dimension": identity.dimension,
                }
            )
            self._initialized = True

    async def index_event(
        self,
        event: JournalEvent,
        journal: JournalReceipt,
    ) -> RuntimeIndexReceipt:
        """Idempotently upsert only after the canonical Markdown write succeeded."""

        await self.initialize()
        document = _event_document(event)
        vectors = await self._embedder.embed([document])
        if len(vectors) != 1:
            raise RuntimeError("local embedder returned an unexpected vector count")
        record_id = f"journal-event:{event.event_id}"
        source_path = str(journal.relative_path)
        metadata = json_scalar_mapping(
            {
                "event_id": str(event.event_id),
                "event_kind": event.kind.value,
                "agent_id": event.agent_id,
                "occurred_at": event.occurred_at.isoformat(),
                "content_sha256": event.content_sha256,
                "source_path": source_path,
                "schema": RUNTIME_MEMORY_SCHEMA,
            }
        )
        await self._store.upsert(
            [
                VectorRecord(
                    id=record_id,
                    embedding=vectors[0],
                    document=document,
                    metadata=metadata,
                )
            ]
        )
        return RuntimeIndexReceipt(
            event_id=str(event.event_id),
            record_id=record_id,
            content_sha256=event.content_sha256,
            source_path=source_path,
            indexed_at=datetime.now(UTC),
        )

    async def query(
        self,
        text: str,
        *,
        top_k: int,
        filters: Mapping[str, Scalar] | None = None,
    ) -> RuntimeMemoryQueryResult:
        await self.initialize()
        normalized = text.strip()
        if not normalized:
            raise ValueError("runtime-memory query cannot be empty")
        if not 1 <= top_k <= 25:
            raise ValueError("top_k must be between 1 and 25")
        vectors = await self._embedder.embed([normalized])
        if len(vectors) != 1:
            raise RuntimeError("local embedder returned an unexpected vector count")
        hits = await self._store.query(
            vectors[0],
            top_k=top_k,
            filters=json_scalar_mapping(filters) if filters else None,
        )
        return RuntimeMemoryQueryResult(
            query=normalized,
            hits=tuple(_runtime_hit(hit) for hit in hits),
        )

    async def health(self) -> Sequence[ComponentHealth]:
        return (await self._embedder.health(), await self._store.health())

    async def aclose(self) -> None:
        await asyncio.gather(
            self._embedder.aclose(),
            self._store.aclose(),
        )


def _runtime_hit(hit: SearchHit) -> RuntimeMemoryHit:
    return RuntimeMemoryHit(
        id=hit.id,
        document=hit.document,
        distance=hit.distance,
        metadata=hit.metadata,
    )
