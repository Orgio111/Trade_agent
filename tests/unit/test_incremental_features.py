"""Deterministic incremental feature-state behavior."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from packages.domain import CandlePayload, MarketEvent, SourceMode
from workers.features.runtime import IncrementalFeatureEngine


def event(index: int, close: str) -> MarketEvent:
    opened = datetime(2026, 7, 15, tzinfo=UTC) + timedelta(minutes=index)
    price = Decimal(close)
    return MarketEvent.create(
        trace_id="feature-replay",
        event_type="market.candle",
        venue="binance",
        market_type="spot",
        instrument_id="BTCUSDT",
        exchange_ts=opened + timedelta(minutes=1),
        received_ts=opened + timedelta(minutes=1, milliseconds=100),
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


def test_stale_event_is_marked_unfresh() -> None:
    source = event(0, "100")
    snapshot = IncrementalFeatureEngine(max_age_seconds=30).update(
        source,
        evaluated_at=source.exchange_ts + timedelta(seconds=31),
    )

    assert snapshot.fresh is False
    assert snapshot.market_data_age_seconds == Decimal("31")
