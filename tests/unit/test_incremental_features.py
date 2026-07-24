"""Deterministic incremental feature-state behavior."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib
import json

import pytest

from packages.domain import CandlePayload, MarketEvent, SourceMode
from workers.features.runtime import IncrementalFeatureEngine
from workers.features.postgres import PostgresFeatureRepository


def event(index: int, close: str, *, receive_offset_ms: int = 100) -> MarketEvent:
    opened = datetime(2026, 7, 15, tzinfo=UTC) + timedelta(minutes=index)
    price = Decimal(close)
    exchange_ts = opened + timedelta(minutes=1)
    return MarketEvent.create(
        trace_id="feature-replay",
        event_type="market.candle",
        venue="binance",
        market_type="spot",
        instrument_id="BTCUSDT",
        exchange_ts=exchange_ts,
        received_ts=exchange_ts + timedelta(milliseconds=receive_offset_ms),
        source_mode=SourceMode.REPLAY,
        ingest_run_id="feature-fixture",
        sequence_start=index + 1,
        sequence_end=index + 1,
        payload=CandlePayload(
            interval="1m",
            open_time=opened,
            close_time=opened + timedelta(minutes=1),
            open=price - Decimal("0.5"),
            high=price + Decimal("1"),
            low=price - Decimal("1"),
            close=price,
            volume=Decimal("10") + index,
        ),
    )


def test_incremental_snapshot_contains_required_bounded_state() -> None:
    engine = IncrementalFeatureEngine()
    snapshot = None
    for index in range(205):
        snapshot = engine.update(event(index, str(100 + index)))

    assert snapshot is not None
    assert snapshot.candle_count == 200
    assert snapshot.session_high == Decimal("305")
    assert snapshot.session_low == Decimal("99")
    assert snapshot.session_vwap > 0
    assert snapshot.atr > 0
    assert snapshot.atr_median > 0
    assert snapshot.volatility >= 0
    assert snapshot.ema_9 > snapshot.ema_21 > snapshot.ema_50
    assert Decimal("0") <= snapshot.rsi_14 <= Decimal("100")
    assert snapshot.trend_direction == "up"
    assert snapshot.fresh is True
    assert snapshot.verify_checksum()


def test_duplicate_event_is_idempotent() -> None:
    engine = IncrementalFeatureEngine()
    source = event(0, "100")

    first = engine.update(source)
    duplicate = engine.update(source)

    assert duplicate == first
    assert duplicate.state_version == 1


def test_checkpoint_round_trip_preserves_next_snapshot() -> None:
    original = IncrementalFeatureEngine()
    for index in range(20):
        original.update(event(index, str(100 + index)))

    restored = IncrementalFeatureEngine.from_checkpoint(original.checkpoint_json())
    expected = original.update(event(20, "120"))
    actual = restored.update(event(20, "120"))

    assert actual == expected
    assert actual.input_event_id == event(20, "120").event_id


def test_checkpoint_restart_accepts_last_event_redelivery_idempotently() -> None:
    original = IncrementalFeatureEngine()
    source = event(0, "100")
    expected = original.update(source)

    restored = IncrementalFeatureEngine.from_checkpoint(original.checkpoint_json())

    assert restored.update(source) == expected


def test_postgres_jsonb_formatting_does_not_change_checkpoint_identity() -> None:
    original = IncrementalFeatureEngine()
    original.update(event(0, "100"))
    canonical_checkpoint = original.checkpoint_json()
    database_jsonb_text = json.dumps(
        json.loads(canonical_checkpoint),
        indent=2,
        sort_keys=True,
    )

    restored = PostgresFeatureRepository._restore(
        {
            "checkpoint": database_jsonb_text,
            "checkpoint_checksum": hashlib.sha256(
                canonical_checkpoint.encode("utf-8")
            ).hexdigest(),
        }
    )

    assert restored.update(event(1, "101")).state_version == 2


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FeatureConnection:
    def __init__(self, checkpoint: dict[str, object]) -> None:
        self._checkpoint = checkpoint

    def transaction(self) -> _AsyncContext:
        return _AsyncContext(self)

    async def execute(self, *_args: object) -> str:
        return "OK"

    async def fetchval(self, *_args: object) -> str:
        return "inserted"

    async def fetchrow(self, *_args: object) -> dict[str, object]:
        return self._checkpoint


class _FeaturePool:
    def __init__(self, connection: _FeatureConnection) -> None:
        self._connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self._connection)


@pytest.mark.asyncio
async def test_postgres_repository_uses_processing_time_after_accepted_clock_skew() -> None:
    original = IncrementalFeatureEngine()
    first = original.update(event(0, "100"))
    canonical_checkpoint = original.checkpoint_json()
    second = event(1, "101", receive_offset_ms=-20)
    processing_time = second.exchange_ts + timedelta(milliseconds=50)
    checkpoint = {
        "checkpoint": json.dumps(json.loads(canonical_checkpoint), indent=2),
        "checkpoint_checksum": hashlib.sha256(
            canonical_checkpoint.encode("utf-8")
        ).hexdigest(),
        "feature_snapshot": first.model_dump(mode="json"),
    }
    repository = PostgresFeatureRepository(
        _FeaturePool(_FeatureConnection(checkpoint)),  # type: ignore[arg-type]
        stream_name="QUANTEX_CORE",
        durable_name="feature-worker-v1",
        clock=lambda: processing_time,
    )

    snapshot = await repository.process(second)

    assert snapshot.state_version == 2
    assert snapshot.market_data_age_seconds == Decimal("0.05")


def test_stale_event_is_marked_unfresh() -> None:
    source = event(0, "100")
    snapshot = IncrementalFeatureEngine(max_age_seconds=30).update(
        source,
        evaluated_at=source.exchange_ts + timedelta(seconds=31),
    )

    assert snapshot.fresh is False
    assert snapshot.market_data_age_seconds == Decimal("31")
