"""Async dependency boundaries for the local Hermes integration layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from packages.knowledge.models import ComponentHealth, Scalar

from .models import JournalEvent, JournalReceipt

if TYPE_CHECKING:
    from .runtime_memory import RuntimeIndexReceipt, RuntimeMemoryQueryResult


class ObsidianJournalPort(Protocol):
    """Append-only permanent-memory boundary."""

    async def append(self, event: JournalEvent) -> JournalReceipt: ...

    async def get_receipt(
        self,
        event_id: UUID,
        *,
        occurred_at: datetime | None = None,
    ) -> JournalReceipt | None: ...

    async def health(self) -> ComponentHealth: ...


class RuntimeMemoryPort(Protocol):
    """Disposable retrieval projection of journaled events."""

    async def initialize(self) -> None: ...

    async def index_event(
        self,
        event: JournalEvent,
        journal: JournalReceipt,
    ) -> RuntimeIndexReceipt: ...

    async def query(
        self,
        text: str,
        *,
        top_k: int,
        filters: Mapping[str, Scalar] | None = None,
    ) -> RuntimeMemoryQueryResult: ...

    async def health(self) -> Sequence[ComponentHealth]: ...

    async def aclose(self) -> None: ...
