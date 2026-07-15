"""Security and append-only behavior tests for the Obsidian journal."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from packages.hermes.errors import (
    JournalConflictError,
    JournalPathError,
    JournalSecretError,
)
from packages.hermes.models import JournalEvent, JournalEventKind
from packages.hermes.obsidian import ObsidianJournalStore


def _event(**overrides: object) -> JournalEvent:
    values: dict[str, object] = {
        "occurred_at": datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC),
        "kind": JournalEventKind.INCIDENT,
        "agent_id": "engineering-agent",
        "title": "Local service incident",
        "body": "A local dependency became unavailable and recovered.",
        "tags": ["incident"],
        "source_refs": ["system.health.local"],
        "metadata": {"recovered": True},
    }
    values.update(overrides)
    return JournalEvent.create(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_append_is_create_only_and_idempotent(tmp_path: Path) -> None:
    store = ObsidianJournalStore(tmp_path)
    event = _event()

    first = await store.append(event)
    second = await store.append(event)
    journal_path = tmp_path.joinpath(*first.relative_path.parts)

    assert first.created is True
    assert second.created is False
    assert first.relative_path == second.relative_path
    assert journal_path.is_file()
    assert journal_path.read_text(encoding="utf-8").startswith("---\n")
    assert f'content_sha256: "{event.content_sha256}"' in journal_path.read_text(
        encoding="utf-8"
    )
    receipt = await store.get_receipt(event.event_id)
    assert receipt is not None
    assert receipt.content_sha256 == event.content_sha256


@pytest.mark.asyncio
async def test_same_external_id_with_different_content_conflicts(
    tmp_path: Path,
) -> None:
    store = ObsidianJournalStore(tmp_path)
    event_id = uuid4()
    await store.append(_event(event_id=event_id, body="First durable fact."))

    with pytest.raises(JournalConflictError, match="different content"):
        await store.append(_event(event_id=event_id, body="Conflicting fact."))


@pytest.mark.asyncio
async def test_secret_like_material_is_rejected_without_a_file(
    tmp_path: Path,
) -> None:
    store = ObsidianJournalStore(tmp_path)
    secret = "sk-THIS_VALUE_MUST_NOT_BE_WRITTEN_1234567890"

    with pytest.raises(JournalSecretError) as failure:
        await store.append(_event(body=f'Reported API_KEY = "{secret}"'))

    assert secret not in str(failure.value)
    assert not store.events_root.exists()


@pytest.mark.asyncio
async def test_symlink_or_junction_escape_is_rejected(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"outside-{uuid4()}"
    outside.mkdir()
    store = ObsidianJournalStore(tmp_path)
    namespace = tmp_path / "trade-agent" / "memory"
    namespace.mkdir(parents=True)
    link = namespace / "events"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks require elevated privileges on this host")

    try:
        with pytest.raises(JournalPathError, match="symlink|junction|escapes"):
            await store.append(_event())
        assert not any(outside.iterdir())
    finally:
        link.unlink(missing_ok=True)
        outside.rmdir()


@pytest.mark.asyncio
async def test_journal_path_is_fixed_by_utc_event_date(tmp_path: Path) -> None:
    store = ObsidianJournalStore(tmp_path)
    event = _event()

    receipt = await store.append(event)

    assert receipt.relative_path.as_posix() == (
        f"trade-agent/memory/events/2026/07/16/{event.event_id}.md"
    )
    assert (await store.health()).healthy is True
