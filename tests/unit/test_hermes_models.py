"""Integrity and local-model policy tests for Hermes contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from packages.hermes.models import (
    LOCAL_MODEL_BY_ROLE,
    JournalEvent,
    JournalEventKind,
    LocalModelRole,
)


def _event(**overrides: object) -> JournalEvent:
    values: dict[str, object] = {
        "occurred_at": datetime(2026, 7, 16, 1, 2, 3, tzinfo=UTC),
        "kind": JournalEventKind.OBSERVATION,
        "agent_id": "research-agent",
        "title": "Observed system state",
        "body": "The upstream component reported a completed observation.",
        "tags": ["hermes", "memory"],
        "source_refs": ["wiki/content/concepts/multi-agent-pipeline.md"],
        "metadata": {"attempt": 1, "verified": True},
    }
    values.update(overrides)
    return JournalEvent.create(**values)  # type: ignore[arg-type]


def test_local_model_registry_is_exact_and_immutable() -> None:
    assert dict(LOCAL_MODEL_BY_ROLE) == {
        LocalModelRole.REASONING: "qwen3:8b",
        LocalModelRole.FAST: "phi3:3.8b",
        LocalModelRole.RESEARCH: "deepseek-r1:8b",
        LocalModelRole.VISION: "moondream",
        LocalModelRole.TOOL_FORMATTING: "mistral",
        LocalModelRole.EMBEDDING: "nomic-embed-text",
    }
    with pytest.raises(TypeError):
        LOCAL_MODEL_BY_ROLE[LocalModelRole.REASONING] = "cloud-model"  # type: ignore[index]


def test_event_normalizes_to_utc_and_is_content_stable() -> None:
    east = datetime(
        2026,
        7,
        16,
        9,
        2,
        3,
        tzinfo=timezone(timedelta(hours=8)),
    )
    first = _event(occurred_at=east, tags=["memory", "hermes", "memory"])
    second = _event()

    assert first.occurred_at == second.occurred_at
    assert first.event_id == second.event_id
    assert first.content_sha256 == second.content_sha256
    assert first.tags == ("hermes", "memory")
    assert len(first.content_sha256) == 64


def test_external_uuid_supports_source_idempotency_keys() -> None:
    source_id = uuid4()

    assert _event(event_id=source_id).event_id == source_id


def test_naive_timestamp_and_non_scalar_metadata_are_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _event(occurred_at=datetime(2026, 7, 16, 1, 2, 3))

    with pytest.raises((TypeError, ValidationError), match="scalar JSON"):
        _event(metadata={"nested": {"not": "allowed"}})


def test_tampered_checksum_and_unknown_fields_are_rejected() -> None:
    event = _event()
    payload = event.model_dump()
    payload["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match"):
        JournalEvent.model_validate(payload)

    payload = event.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        JournalEvent.model_validate(payload)
