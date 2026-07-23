"""Focused runtime tests for raw market-data validation and durable routing."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from packages.domain import CandlePayload, MarketEvent, SourceMode
from packages.event_bus import CoreSubject, JetStreamEventBus, JetStreamTransportError
from workers.config import WorkerSettings
from workers.market_data.runtime import MarketDataRuntime, MarketRejectionEvent


NOW = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


class FakeContext:
    def __init__(self) -> None:
        self.publications: list[tuple[str, bytes, dict[str, object]]] = []
        self.subscription: tuple[str, dict[str, object]] | None = None
        self.callback = None
        self.fail_next_publish = False

    async def publish(self, subject: str, payload: bytes = b"", **kwargs):
        self.publications.append((subject, payload, kwargs))
        if self.fail_next_publish:
            self.fail_next_publish = False
            raise ConnectionError("offline")
        return SimpleNamespace(stream="QUANTEX_CORE", seq=len(self.publications))

    async def subscribe(self, subject: str, **kwargs):
        self.subscription = (subject, kwargs)
        self.callback = kwargs["cb"]
        return "subscription"


class FakeMessage:
    def __init__(self, payload: bytes, *, delivery: int = 1) -> None:
        self.subject = CoreSubject.MARKET_RAW.value
        self.data = payload
        self.metadata = SimpleNamespace(num_delivered=delivery)
        self.acked = 0
        self.nacked = 0
        self.terminated = 0

    async def ack(self) -> None:
        self.acked += 1

    async def nak(self, *, delay=None) -> None:
        self.nacked += 1

    async def term(self) -> None:
        self.terminated += 1


class FakeDurableRouter:
    def __init__(self, bus: JetStreamEventBus) -> None:
        self.bus = bus
        self.events: dict[str, tuple[CoreSubject, bytes, str]] = {}

    async def route(self, **event: object) -> bool:
        source_event_id = str(event["source_event_id"])
        created = source_event_id not in self.events
        self.events.setdefault(
            source_event_id,
            (event["subject"], event["payload"], str(event["message_id"])),
        )
        if created:
            subject, payload, message_id = self.events[source_event_id]
            await self.bus.publish(subject, payload, message_id=message_id)
        return created


def settings(*, mode: SourceMode = SourceMode.PAPER_LIVE) -> WorkerSettings:
    return WorkerSettings(
        service_name="market-data-worker",
        mode=mode,
        nats_url="nats://127.0.0.1:4222",
        database_url=SecretStr("postgresql://127.0.0.1/quantex"),
        durable_name="market-data-worker-v1",
    )


def market_event(
    *,
    mode: SourceMode = SourceMode.PAPER_LIVE,
    exchange_ts: datetime = NOW - timedelta(milliseconds=100),
) -> MarketEvent:
    payload = CandlePayload(
        interval="1m",
        open_time=exchange_ts - timedelta(minutes=1),
        close_time=exchange_ts,
        open="60000",
        high="60100",
        low="59950",
        close="60080",
        volume="12.3",
    )
    return MarketEvent.create(
        trace_id="trace-market-runtime-1",
        event_type="market.candle.closed",
        venue="binance",
        market_type="spot",
        instrument_id="BTCUSDT",
        exchange_ts=exchange_ts,
        received_ts=exchange_ts + timedelta(milliseconds=50),
        sequence_start=100,
        sequence_end=100,
        source_mode=mode,
        ingest_run_id="paper-live-runtime-1",
        payload=payload,
    )


def build_runtime(
    *, mode: SourceMode = SourceMode.PAPER_LIVE, durable: bool = False
) -> tuple[MarketDataRuntime, FakeContext]:
    context = FakeContext()
    bus = JetStreamEventBus(context)
    runtime = MarketDataRuntime(
        settings(mode=mode),
        bus,
        clock=lambda: NOW,
        durable_router=FakeDurableRouter(bus) if durable else None,
    )
    return runtime, context


@pytest.mark.asyncio
async def test_valid_event_is_canonical_and_acked_after_durable_publish() -> None:
    runtime, context = build_runtime()
    raw = market_event().canonical_json().encode()
    assert await runtime.start() == "subscription"
    message = FakeMessage(raw)

    await context.callback(message)

    assert context.subscription[0] == CoreSubject.MARKET_RAW.value
    assert context.subscription[1]["manual_ack"] is True
    assert message.acked == 1
    assert message.nacked == 0
    subject, payload, publish = context.publications[0]
    assert subject == CoreSubject.MARKET_VALIDATED.value
    assert payload == raw
    assert publish["headers"] == {
        "Nats-Msg-Id": f"market-validated:{market_event().event_id}"
    }


@pytest.mark.asyncio
async def test_contract_failure_publishes_payload_free_deterministic_rejection() -> (
    None
):
    runtime, context = build_runtime()
    raw = b'{"not":"a market event","secret":"must-not-leak"}'

    first = await runtime.handle(raw)
    second = await runtime.handle(raw)

    assert first == second
    assert first.subject is CoreSubject.MARKET_REJECTED
    rejection = MarketRejectionEvent.model_validate_json(context.publications[0][1])
    assert rejection.reason_codes == ("CONTRACT_INVALID",)
    assert rejection.payload_sha256 == hashlib.sha256(raw).hexdigest()
    assert b"must-not-leak" not in context.publications[0][1]
    assert context.publications[0][1] == context.publications[1][1]


@pytest.mark.asyncio
async def test_source_mode_mismatch_fails_closed_without_quality_processing() -> None:
    runtime, context = build_runtime(mode=SourceMode.REPLAY)
    source = market_event(mode=SourceMode.PAPER_LIVE)

    outcome = await runtime.handle(source.canonical_json().encode())

    assert outcome.subject is CoreSubject.MARKET_REJECTED
    rejection = MarketRejectionEvent.model_validate_json(context.publications[0][1])
    assert rejection.reason_codes == ("SOURCE_MODE_MISMATCH",)
    assert rejection.runtime_mode is SourceMode.REPLAY
    assert rejection.source_mode is SourceMode.PAPER_LIVE
    assert rejection.source_event_id == source.event_id


@pytest.mark.asyncio
async def test_quality_rejection_contains_machine_codes_not_raw_payload() -> None:
    runtime, context = build_runtime()
    source = market_event(exchange_ts=NOW - timedelta(seconds=30))
    raw = source.canonical_json().encode()

    await runtime.handle(raw)

    rejection = MarketRejectionEvent.model_validate_json(context.publications[0][1])
    assert "STALE_EVENT" in rejection.reason_codes
    assert rejection.reason_fields == ("exchange_ts",)
    assert rejection.payload_sha256 == hashlib.sha256(raw).hexdigest()
    encoded = json.loads(context.publications[0][1])
    assert "payload" not in encoded


@pytest.mark.asyncio
async def test_publish_retry_reuses_validated_result_instead_of_rejecting_sequence() -> (
    None
):
    runtime, context = build_runtime()
    raw = market_event().canonical_json().encode()
    context.fail_next_publish = True

    with pytest.raises(JetStreamTransportError, match="publish failed"):
        await runtime.handle(raw)
    outcome = await runtime.handle(raw)

    assert outcome.subject is CoreSubject.MARKET_VALIDATED
    assert [item[0] for item in context.publications] == [
        CoreSubject.MARKET_VALIDATED.value,
        CoreSubject.MARKET_VALIDATED.value,
    ]
    assert context.publications[0][1] == context.publications[1][1]


@pytest.mark.asyncio
async def test_durable_callback_naks_failed_output_then_acks_retry() -> None:
    runtime, context = build_runtime()
    raw = market_event().canonical_json().encode()
    await runtime.start()
    context.fail_next_publish = True
    first = FakeMessage(raw)
    retry = FakeMessage(raw, delivery=2)

    await context.callback(first)
    await context.callback(retry)

    assert first.acked == 0
    assert first.nacked == 1
    assert first.terminated == 0
    assert retry.acked == 1
    assert retry.nacked == 0
    assert all(
        publication[0] == CoreSubject.MARKET_VALIDATED.value
        for publication in context.publications
    )


@pytest.mark.asyncio
async def test_durable_router_suppresses_duplicate_after_runtime_restart() -> None:
    context = FakeContext()
    bus = JetStreamEventBus(context)
    router = FakeDurableRouter(bus)
    raw = market_event().canonical_json().encode()

    first = MarketDataRuntime(settings(), bus, clock=lambda: NOW, durable_router=router)
    second = MarketDataRuntime(settings(), bus, clock=lambda: NOW, durable_router=router)
    await first.handle(raw)
    await second.handle(raw)

    assert len(context.publications) == 1
