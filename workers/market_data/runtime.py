"""Durable raw-ingress runtime for the canonical market-data quality gate."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import re
from typing import Any, Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from packages.domain import MarketEvent, SourceMode, canonical_json
from packages.event_bus import (
    ConsumerFailure,
    CoreSubject,
    InMemoryEventBus,
    JetStreamEventBus,
)
from workers.config import WorkerSettings
from workers.market_data.main import MarketDataWorker
from workers.nats_runtime import consumer_settings


_LOGGER = logging.getLogger(__name__)
_ZERO_UUID = UUID(int=0)
_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MarketRejectionEvent(BaseModel):
    """Payload-free, deterministic rejection record for one raw ingress item."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    schema_version: Literal["1.0"] = "1.0"
    event_type: Literal["market.rejected"] = "market.rejected"
    event_id: UUID = _ZERO_UUID
    source_event_id: UUID | None = None
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    runtime_mode: SourceMode
    source_mode: SourceMode | None = None
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_codes: tuple[str, ...] = Field(min_length=1)
    reason_fields: tuple[str, ...] = ()

    @field_validator("reason_codes")
    @classmethod
    def normalize_reason_codes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(_REASON_CODE.fullmatch(code) is None for code in value):
            raise ValueError("rejection reason codes must be uppercase identifiers")
        return tuple(sorted(set(value)))

    @field_validator("reason_fields")
    @classmethod
    def normalize_reason_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def bind_identity(self) -> MarketRejectionEvent:
        expected = uuid5(
            NAMESPACE_URL,
            "urn:trade-agent:market-rejection:v1:"
            + hashlib.sha256(
                canonical_json(self.content_payload()).encode("utf-8")
            ).hexdigest(),
        )
        if self.event_id not in {_ZERO_UUID, expected}:
            raise ValueError("event_id does not match rejection content")
        if self.event_id == _ZERO_UUID:
            object.__setattr__(self, "event_id", expected)
        return self

    def content_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "source_event_id": self.source_event_id,
            "trace_id": self.trace_id,
            "runtime_mode": self.runtime_mode,
            "source_mode": self.source_mode,
            "payload_sha256": self.payload_sha256,
            "reason_codes": self.reason_codes,
            "reason_fields": self.reason_fields,
        }

    def canonical_json(self) -> str:
        return canonical_json(self)


@dataclass(frozen=True, slots=True)
class MarketIngressOutcome:
    """Observable result after the durable output publication succeeds."""

    subject: CoreSubject
    event_id: UUID


class DurableMarketRouter(Protocol):
    async def route(
        self,
        *,
        source_event_id: str,
        source_subject: CoreSubject,
        source_payload: dict[str, Any],
        payload_checksum: str,
        output_event_id: str,
        aggregate_type: str,
        aggregate_id: str,
        subject: CoreSubject,
        payload: bytes,
        message_id: str,
    ) -> bool: ...


class MarketDataRuntime:
    """Parse, mode-check, validate, and durably route raw market events."""

    def __init__(
        self,
        settings: WorkerSettings,
        bus: JetStreamEventBus,
        *,
        clock: Callable[[], datetime] = _utc_now,
        max_retry_cache_entries: int = 1024,
        durable_router: DurableMarketRouter | None = None,
    ) -> None:
        if max_retry_cache_entries < 64:
            raise ValueError("retry cache must cover all pending consumer messages")
        self._settings = settings
        self._bus = bus
        self._clock = clock
        self._worker = MarketDataWorker(InMemoryEventBus())
        self._lock = asyncio.Lock()
        self._accepted_retry_cache: OrderedDict[str, tuple[UUID, bytes]] = OrderedDict()
        self._max_retry_cache_entries = max_retry_cache_entries
        self._durable_router = durable_router

    async def start(self) -> object:
        """Attach the exact raw subject using explicit durable/manual ACK."""

        return await self._bus.subscribe(
            CoreSubject.MARKET_RAW,
            self._consume,
            settings=consumer_settings(self._settings),
            on_failure=self._record_consumer_failure,
        )

    async def _consume(self, payload: bytes) -> None:
        await self.handle(payload)

    async def handle(self, payload: bytes) -> MarketIngressOutcome:
        """Process one ingress payload; publication failure propagates for NAK."""

        payload_sha256 = hashlib.sha256(payload).hexdigest()
        async with self._lock:
            cached = self._accepted_retry_cache.get(payload_sha256)
            if cached is not None:
                event_id, encoded = cached
                self._accepted_retry_cache.move_to_end(payload_sha256)
                await self._publish_validated(
                    event_id, encoded, payload_sha256=payload_sha256
                )
                return MarketIngressOutcome(CoreSubject.MARKET_VALIDATED, event_id)

            try:
                event = MarketEvent.model_validate_json(payload)
            except ValidationError:
                return await self._reject(
                    payload_sha256=payload_sha256,
                    reason_codes=("CONTRACT_INVALID",),
                )

            if event.source_mode != self._settings.mode:
                return await self._reject(
                    payload_sha256=payload_sha256,
                    source_event=event,
                    reason_codes=("SOURCE_MODE_MISMATCH",),
                )

            result = self._worker.process(event, now=self._clock())
            if not result.verdict.accepted:
                return await self._reject(
                    payload_sha256=payload_sha256,
                    source_event=event,
                    reason_codes=tuple(
                        reason.code.value for reason in result.verdict.reasons
                    ),
                    reason_fields=tuple(
                        reason.field
                        for reason in result.verdict.reasons
                        if reason.field is not None
                    ),
                )

            encoded = event.canonical_json().encode("utf-8")
            self._remember_accepted(payload_sha256, event.event_id, encoded)
            await self._publish_validated(
                event.event_id, encoded, payload_sha256=payload_sha256
            )
            return MarketIngressOutcome(CoreSubject.MARKET_VALIDATED, event.event_id)

    async def _publish_validated(
        self, event_id: UUID, payload: bytes, *, payload_sha256: str
    ) -> None:
        if self._durable_router is not None:
            await self._durable_router.route(
                source_event_id=str(event_id),
                source_subject=CoreSubject.MARKET_RAW,
                source_payload=json.loads(payload),
                payload_checksum=payload_sha256,
                output_event_id=f"market-validated:{event_id}",
                aggregate_type="market",
                aggregate_id=str(event_id),
                subject=CoreSubject.MARKET_VALIDATED,
                payload=payload,
                message_id=f"market-validated:{event_id}",
            )
            return
        await self._bus.publish(
            CoreSubject.MARKET_VALIDATED,
            payload,
            message_id=f"market-validated:{event_id}",
            stream=self._settings.stream_name,
        )

    async def _reject(
        self,
        *,
        payload_sha256: str,
        reason_codes: tuple[str, ...],
        source_event: MarketEvent | None = None,
        reason_fields: tuple[str, ...] = (),
    ) -> MarketIngressOutcome:
        rejection = MarketRejectionEvent(
            source_event_id=(
                source_event.event_id if source_event is not None else None
            ),
            trace_id=(source_event.trace_id if source_event is not None else None),
            runtime_mode=self._settings.mode,
            source_mode=(
                source_event.source_mode if source_event is not None else None
            ),
            payload_sha256=payload_sha256,
            reason_codes=reason_codes,
            reason_fields=reason_fields,
        )
        await self._bus.publish(
            CoreSubject.MARKET_REJECTED,
            rejection.canonical_json().encode("utf-8"),
            message_id=f"market-rejected:{rejection.event_id}",
            stream=self._settings.stream_name,
        )
        return MarketIngressOutcome(CoreSubject.MARKET_REJECTED, rejection.event_id)

    def _remember_accepted(
        self,
        payload_sha256: str,
        event_id: UUID,
        encoded: bytes,
    ) -> None:
        self._accepted_retry_cache[payload_sha256] = (event_id, encoded)
        self._accepted_retry_cache.move_to_end(payload_sha256)
        while len(self._accepted_retry_cache) > self._max_retry_cache_entries:
            self._accepted_retry_cache.popitem(last=False)

    async def _record_consumer_failure(self, failure: ConsumerFailure) -> None:
        _LOGGER.warning(
            "market ingress failed subject=%s durable=%s delivery=%d "
            "terminal=%s error_type=%s",
            failure.subject,
            failure.durable_name,
            failure.delivery_count,
            failure.terminal,
            failure.error_type,
        )
