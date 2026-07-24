"""Feature snapshot to typed local candidate runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from packages.event_bus import CoreSubject
from workers.config import WorkerSettings
from workers.durability import PostgresInboxOutbox
from workers.features import FeatureSnapshot
from workers.nats_runtime import consumer_settings

from .market import BinanceBookTickerClient
from .runtime import CandidateProducer


class CandidateBus(Protocol):
    async def subscribe(
        self, subject: CoreSubject, handler: Any, *, settings: Any
    ) -> Any: ...


class CandidateRuntime:
    def __init__(
        self,
        settings: WorkerSettings,
        bus: CandidateBus,
        producer: CandidateProducer,
        market: BinanceBookTickerClient,
        durability: PostgresInboxOutbox,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._producer = producer
        self._market = market
        self._durability = durability

    async def handle(self, payload: bytes) -> object | None:
        feature = FeatureSnapshot.model_validate_json(payload)
        source_id = str(feature.input_event_id)
        source_payload = feature.model_dump(mode="json")
        reserved = await self._durability.reserve(
            source_event_id=source_id,
            source_subject=CoreSubject.FEATURES_READY,
            source_payload=source_payload,
            payload_checksum=feature.checksum,
        )
        if not reserved:
            if await self._durability.reservation_processed(source_id):
                return None
            raise RuntimeError("candidate reservation is still active")
        market = await self._market.fetch(feature.symbol)
        candidate = await self._producer.generate(
            feature,
            market,
            generated_at=datetime.now(UTC),
            source_mode=self._settings.mode,
        )
        if candidate is None:
            await self._durability.complete_reserved_without_output(source_id)
            return None
        encoded = candidate.canonical_json().encode()
        output_id = f"signal-candidate:{candidate.event_id}"
        await self._durability.complete_reserved(
            source_event_id=source_id,
            output_event_id=output_id,
            aggregate_type="candidate",
            aggregate_id=str(candidate.event_id),
            subject=CoreSubject.SIGNAL_CANDIDATE,
            payload=encoded,
            message_id=output_id,
        )
        return candidate

    async def start(self) -> Any:
        return await self._bus.subscribe(
            CoreSubject.FEATURES_READY,
            self.handle,
            settings=consumer_settings(self._settings),
        )
