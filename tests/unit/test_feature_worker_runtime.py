"""Durable feature-worker transport boundary."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from packages.domain import CandlePayload, MarketEvent, SourceMode
from packages.event_bus import CoreSubject
from workers.config import WorkerSettings
from workers.features import IncrementalFeatureEngine
from workers.features.service import FeatureRuntime


NOW = datetime(2026, 7, 15, 1, 0, tzinfo=UTC)


def source(*, mode: SourceMode = SourceMode.PAPER_LIVE) -> MarketEvent:
    return MarketEvent.create(
        trace_id="feature-runtime",
        event_type="market.candle",
        venue="binance",
        market_type="spot",
        instrument_id="BTCUSDT",
        exchange_ts=NOW,
        received_ts=NOW + timedelta(milliseconds=10),
        source_mode=mode,
        ingest_run_id="binance-public-live-v1",
        payload=CandlePayload(
            interval="1m",
            open_time=NOW - timedelta(minutes=1),
            close_time=NOW,
            open="99",
            high="101",
            low="98",
            close="100",
            volume="10",
        ),
    )


class FakeRepository:
    def __init__(self) -> None:
        self.engine = IncrementalFeatureEngine()
        self.calls = 0

    async def process(self, event: MarketEvent):
        self.calls += 1
        return self.engine.update(event)


class FakeBus:
    def __init__(self) -> None:
        self.subscription = None

    async def subscribe(self, subject, handler, *, settings, on_failure=None):
        self.subscription = (subject, handler, settings, on_failure)
        return SimpleNamespace()


def settings() -> WorkerSettings:
    return WorkerSettings(
        service_name="feature-worker",
        nats_url="nats://nats:4222",
        database_url=SecretStr("postgresql://quantex:***@postgres:5432/quantex"),
        durable_name="feature-worker-v1",
    )


@pytest.mark.asyncio
async def test_validated_event_is_processed_by_durable_repository() -> None:
    repository, bus = FakeRepository(), FakeBus()
    runtime = FeatureRuntime(settings(), bus, repository)

    snapshot = await runtime.handle(source().canonical_json().encode())

    assert repository.calls == 1
    assert snapshot.input_event_id == source().event_id
    assert snapshot.verify_checksum()


@pytest.mark.asyncio
async def test_mode_mismatch_fails_before_repository() -> None:
    repository, bus = FakeRepository(), FakeBus()
    runtime = FeatureRuntime(settings(), bus, repository)
    event = source(mode=SourceMode.REPLAY)

    with pytest.raises(ValueError, match="mode"):
        await runtime.handle(event.canonical_json().encode())
    assert repository.calls == 0


@pytest.mark.asyncio
async def test_start_uses_feature_durable_on_validated_subject() -> None:
    repository, bus = FakeRepository(), FakeBus()
    runtime = FeatureRuntime(settings(), bus, repository)

    await runtime.start()

    assert bus.subscription[0] is CoreSubject.MARKET_VALIDATED
    assert bus.subscription[2].durable_name == "feature-worker-v1"
