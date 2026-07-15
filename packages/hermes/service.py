"""Application service joining canonical journal writes and vector projection."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from packages.knowledge.models import ComponentHealth, Scalar

from .errors import RuntimeProjectionError
from .models import JournalEvent, JournalReceipt
from .ports import ObsidianJournalPort, RuntimeMemoryPort
from .runtime_memory import RuntimeIndexReceipt, RuntimeMemoryQueryResult


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HermesIngestReceipt(_StrictModel):
    """Evidence that both canonical and derived writes completed."""

    journal: JournalReceipt
    index: RuntimeIndexReceipt


class HermesHealth(_StrictModel):
    healthy: bool
    components: tuple[ComponentHealth, ...]


class HermesMemoryService:
    """Persist first, then project; a failed projection never erases memory."""

    def __init__(
        self,
        *,
        journal: ObsidianJournalPort,
        memory: RuntimeMemoryPort,
    ) -> None:
        self._journal = journal
        self._memory = memory
        self._locks_guard = asyncio.Lock()
        self._event_locks: dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Fail startup when the local retrieval projection is unavailable."""

        await self._memory.initialize()

    async def _event_lock(self, event_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._event_locks.setdefault(event_id, asyncio.Lock())

    async def ingest(self, event: JournalEvent) -> HermesIngestReceipt:
        event_id = str(event.event_id)
        lock = await self._event_lock(event_id)
        try:
            async with lock:
                journal = await self._journal.append(event)
                try:
                    index = await self._memory.index_event(event, journal)
                except Exception as exc:
                    raise RuntimeProjectionError(
                        event_id=event_id,
                        journal_path=str(journal.relative_path),
                    ) from exc
                return HermesIngestReceipt(journal=journal, index=index)
        finally:
            async with self._locks_guard:
                if not lock.locked():
                    self._event_locks.pop(event_id, None)

    async def query(
        self,
        text: str,
        *,
        top_k: int,
        filters: Mapping[str, Scalar] | None = None,
    ) -> RuntimeMemoryQueryResult:
        result = await self._memory.query(
            text,
            top_k=top_k,
            filters=filters,
        )
        return result

    async def health(self) -> HermesHealth:
        journal_health = await self._journal.health()
        memory_health: Sequence[ComponentHealth] = await self._memory.health()
        components = (journal_health, *memory_health)
        return HermesHealth(
            healthy=all(component.healthy for component in components),
            components=components,
        )

    async def aclose(self) -> None:
        await self._memory.aclose()
