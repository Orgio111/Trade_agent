"""Runtime journal-to-Chroma projection contract tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from packages.hermes.errors import RuntimeProjectionError
from packages.hermes.models import JournalEvent, JournalEventKind
from packages.hermes.obsidian import ObsidianJournalStore
from packages.hermes.runtime_memory import (
    RUNTIME_MEMORY_COLLECTION,
    RUNTIME_MEMORY_SCHEMA,
    RuntimeMemoryIndex,
)
from packages.hermes.service import HermesMemoryService
from tests.fakes.knowledge import DeterministicFakeEmbedder, InMemoryVectorStore


def _event() -> JournalEvent:
    return JournalEvent.create(
        occurred_at=datetime(2026, 7, 16, 9, 30, tzinfo=UTC),
        kind=JournalEventKind.BENCHMARK,
        agent_id="engineering-benchmark",
        title="Hermes projection benchmark",
        body="Validated local projection result.",
        tags=("benchmark", "hermes"),
        source_refs=("wiki/content/decisions/hermes-local-integration-v1.md",),
    )


@pytest.mark.asyncio
async def test_runtime_index_is_isolated_provenance_bearing_and_idempotent(
    tmp_path: Path,
) -> None:
    journal = ObsidianJournalStore(tmp_path)
    event = _event()
    journal_receipt = await journal.append(event)
    embedder = DeterministicFakeEmbedder(dimension=4)
    store = InMemoryVectorStore()
    index = RuntimeMemoryIndex(embedder=embedder, store=store)

    first = await index.index_event(event, journal_receipt)
    second = await index.index_event(event, journal_receipt)

    assert RUNTIME_MEMORY_COLLECTION == "trade-agent-runtime-memory-v1"
    assert first.record_id == second.record_id == f"journal-event:{event.event_id}"
    assert len(store.records) == 1
    assert store.metadata is not None
    assert store.metadata["schema"] == RUNTIME_MEMORY_SCHEMA
    assert store.metadata["canonical_source"] == "obsidian-markdown"
    record = store.records[first.record_id]
    assert record.metadata["source_path"] == str(journal_receipt.relative_path)
    assert record.metadata["content_sha256"] == event.content_sha256
    assert "Validated local projection result" in record.document

    result = await index.query(
        "projection result",
        top_k=5,
        filters={"event_kind": "benchmark"},
    )

    assert [hit.id for hit in result.hits] == [first.record_id]
    assert result.hits[0].metadata["source_path"] == str(journal_receipt.relative_path)


class _FailingMemoryIndex:
    async def initialize(self) -> None:
        return None

    async def index_event(self, event, journal):
        del event, journal
        raise RuntimeError("injected local index failure")

    async def query(self, text, *, top_k, filters=None):
        del text, top_k, filters
        raise AssertionError("not used")

    async def health(self):
        return ()

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_projection_failure_never_rolls_back_canonical_markdown(
    tmp_path: Path,
) -> None:
    event = _event()
    journal = ObsidianJournalStore(tmp_path)
    service = HermesMemoryService(journal=journal, memory=_FailingMemoryIndex())

    with pytest.raises(RuntimeProjectionError) as failure:
        await service.ingest(event)

    assert failure.value.event_id == str(event.event_id)
    receipt = await journal.get_receipt(event.event_id)
    assert receipt is not None
    assert (tmp_path / receipt.relative_path).is_file()
