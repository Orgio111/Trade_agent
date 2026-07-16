"""Strict, provider-neutral contracts for local Hermes memory integration."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.domain.events import canonical_json, compute_payload_checksum
from packages.knowledge.models import Scalar
from packages.local_ai.models import (
    LOCAL_MODEL_BY_ROLE as LOCAL_MODEL_BY_ROLE,
    LocalModelRole as LocalModelRole,
)


class StrictModel(BaseModel):
    """Immutable contract that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class JournalEventKind(str, Enum):
    """Facts Hermes may record without interpreting or authorizing them."""

    OBSERVATION = "observation"
    DECISION = "decision"
    EXECUTION = "execution"
    RESEARCH = "research"
    INCIDENT = "incident"
    BENCHMARK = "benchmark"


_ZERO_UUID = UUID(int=0)
_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_METADATA_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_RECEIPT_PATH_PATTERN = re.compile(
    r"^trade-agent/memory/events/\d{4}/\d{2}/\d{2}/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}\.md$"
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _normalized_text(value: str, *, field: str, allow_newlines: bool) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    if "\x00" in normalized:
        raise ValueError(f"{field} cannot contain NUL bytes")
    if not allow_newlines and "\n" in normalized:
        raise ValueError(f"{field} cannot contain newlines")
    return normalized


def _validate_metadata(value: Any) -> dict[str, Scalar]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping of scalar JSON values")
    if len(value) > 64:
        raise ValueError("metadata cannot contain more than 64 entries")

    result: dict[str, Scalar] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _METADATA_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("metadata keys must be bounded identifiers")
        if not isinstance(item, (str, int, float, bool)):
            raise TypeError("metadata values must be scalar JSON values")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("metadata numbers must be finite")
        if isinstance(item, str):
            if len(item) > 4_096 or "\x00" in item:
                raise ValueError("metadata strings must be bounded and NUL-free")
        result[key] = item
    return dict(sorted(result.items()))


class JournalEvent(StrictModel):
    """One externally produced fact prepared for permanent Markdown storage.

    This envelope records a decision or execution supplied by another component;
    it never creates, validates, or authorizes either one.
    """

    schema_version: Literal["1.0"] = "1.0"
    event_id: UUID = Field(default=_ZERO_UUID)
    occurred_at: datetime
    kind: JournalEventKind
    agent_id: str = Field(
        min_length=2,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    )
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=1_000_000)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    tags: tuple[str, ...] = Field(default=(), max_length=32)
    source_refs: tuple[str, ...] = Field(default=(), max_length=64)
    metadata: Mapping[str, Scalar] = Field(default_factory=dict)
    content_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$")

    @field_validator("occurred_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return _normalized_text(value, field="title", allow_newlines=False)

    @field_validator("body")
    @classmethod
    def normalize_body(cls, value: str) -> str:
        return _normalized_text(value, field="body", allow_newlines=True)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalized_text(value, field="trace_id", allow_newlines=False)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(value)))
        if any(_TAG_PATTERN.fullmatch(tag) is None for tag in normalized):
            raise ValueError("tags must be lowercase slug identifiers")
        return normalized

    @field_validator("source_refs")
    @classmethod
    def normalize_source_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for source in value:
            item = _normalized_text(
                source,
                field="source reference",
                allow_newlines=False,
            )
            if len(item) > 1_024:
                raise ValueError("source references cannot exceed 1024 characters")
            normalized.append(item)
        return tuple(sorted(set(normalized)))

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: Any) -> dict[str, Scalar]:
        return _validate_metadata(value)

    @model_validator(mode="after")
    def bind_integrity_fields(self) -> JournalEvent:
        expected_checksum = compute_payload_checksum(self.content_payload())
        if self.content_sha256 and self.content_sha256 != expected_checksum:
            raise ValueError("content_sha256 does not match journal event content")
        if not self.content_sha256:
            object.__setattr__(self, "content_sha256", expected_checksum)
        if self.event_id == _ZERO_UUID:
            stable_id = uuid5(
                NAMESPACE_URL,
                f"urn:trade-agent:hermes:journal-event:v1:{expected_checksum}",
            )
            object.__setattr__(self, "event_id", stable_id)
        return self

    @classmethod
    def create(
        cls,
        *,
        occurred_at: datetime,
        kind: JournalEventKind,
        agent_id: str,
        title: str,
        body: str,
        event_id: UUID | None = None,
        trace_id: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        source_refs: tuple[str, ...] | list[str] = (),
        metadata: Mapping[str, Scalar] | None = None,
    ) -> JournalEvent:
        """Build a normalized event with a stable content-derived UUID by default."""

        return cls(
            event_id=event_id or _ZERO_UUID,
            occurred_at=occurred_at,
            kind=kind,
            agent_id=agent_id,
            title=title,
            body=body,
            trace_id=trace_id,
            tags=tuple(tags),
            source_refs=tuple(source_refs),
            metadata=metadata or {},
        )

    def content_payload(self) -> dict[str, Any]:
        """Return the canonical content covered by ``content_sha256``."""

        return {
            "schema_version": self.schema_version,
            "occurred_at": self.occurred_at,
            "kind": self.kind,
            "agent_id": self.agent_id,
            "title": self.title,
            "body": self.body,
            "trace_id": self.trace_id,
            "tags": self.tags,
            "source_refs": self.source_refs,
            "metadata": self.metadata,
        }

    def canonical_json(self) -> str:
        """Return deterministic transport JSON for this complete envelope."""

        return canonical_json(self)


class JournalReceipt(StrictModel):
    """Evidence for one create-only or idempotently reused Markdown write."""

    event_id: UUID
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    relative_path: PurePosixPath
    created: bool
    written_at: datetime

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: PurePosixPath) -> PurePosixPath:
        text = value.as_posix()
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("journal receipt path must be repository-relative")
        if _RECEIPT_PATH_PATTERN.fullmatch(text) is None:
            raise ValueError("journal receipt path is outside the event namespace")
        return value

    @field_validator("written_at")
    @classmethod
    def normalize_written_at(cls, value: datetime) -> datetime:
        return _utc(value)
