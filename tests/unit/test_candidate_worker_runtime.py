"""Candidate runtime reservation and fail-closed behavior."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import SecretStr

from packages.execution import MarketSnapshot
from packages.event_bus import CoreSubject
from workers.candidate.service import CandidateRuntime
from workers.config import WorkerSettings
from workers.features import FeatureSnapshot


class FakeDurability:
    def __init__(self, *, reserved: bool = True, processed: bool = False) -> None:
        self.reserved = reserved
        self.processed = processed
        self.completed_without_output = 0
        self.completed = 0

    async def reserve(self, **_kwargs):
        return self.reserved

    async def reservation_processed(self, _source_id: str):
        return self.processed

    async def complete_reserved_without_output(self, _source_id: str):
        self.completed_without_output += 1

    async def complete_reserved(self, **_kwargs):
        self.completed += 1


class FakeProducer:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        return self.result


class FakeMarket:
    async def fetch(self, symbol: str):
        return MarketSnapshot(
            venue="binance",
            market_type="spot",
            instrument=symbol,
            bid="99",
            ask="101",
            observed_at=datetime.now(UTC),
        )


class FakeBus:
    async def subscribe(self, subject, handler, *, settings):
        self.call = (subject, handler, settings)
        return SimpleNamespace()


def feature() -> FeatureSnapshot:
    return FeatureSnapshot(
        input_event_id=uuid4(),
        state_version=1,
        symbol="BTCUSDT",
        observed_at=datetime.now(UTC),
        candle_count=1,
        session_high="101",
        session_low="99",
        session_vwap="100",
        atr="2",
        atr_median="2",
        volatility="0.02",
        ema_9="100",
        ema_21="100",
        ema_50="100",
        rsi_14="50",
        trend_direction="flat",
        trend_strength="0",
        support="99",
        resistance="101",
        liquidity_proxy="1000",
        market_data_age_seconds="0",
        fresh=True,
    )


def settings() -> WorkerSettings:
    return WorkerSettings(
        service_name="candidate-worker",
        nats_url="nats://nats:4222",
        database_url=SecretStr("postgresql://quantex:password@postgres:5432/quantex"),
        durable_name="candidate-worker-v1",
    )


@pytest.mark.asyncio
async def test_hold_or_invalid_output_completes_reservation_without_event() -> None:
    durability, producer = FakeDurability(), FakeProducer(None)
    runtime = CandidateRuntime(settings(), FakeBus(), producer, FakeMarket(), durability)

    assert await runtime.handle(feature().canonical_json().encode()) is None
    assert producer.calls == 1
    assert durability.completed_without_output == 1
    assert durability.completed == 0


@pytest.mark.asyncio
async def test_processed_redelivery_never_calls_market_or_model() -> None:
    durability, producer = FakeDurability(reserved=False, processed=True), FakeProducer()
    runtime = CandidateRuntime(settings(), FakeBus(), producer, FakeMarket(), durability)

    assert await runtime.handle(feature().canonical_json().encode()) is None
    assert producer.calls == 0


@pytest.mark.asyncio
async def test_active_reservation_is_retried_not_acked_as_done() -> None:
    durability = FakeDurability(reserved=False, processed=False)
    runtime = CandidateRuntime(settings(), FakeBus(), FakeProducer(), FakeMarket(), durability)

    with pytest.raises(RuntimeError, match="reservation"):
        await runtime.handle(feature().canonical_json().encode())


@pytest.mark.asyncio
async def test_start_consumes_feature_subject_with_candidate_durable() -> None:
    bus = FakeBus()
    runtime = CandidateRuntime(
        settings(), bus, FakeProducer(), FakeMarket(), FakeDurability()
    )
    await runtime.start()
    assert bus.call[0] is CoreSubject.FEATURES_READY
    assert bus.call[2].durable_name == "candidate-worker-v1"
