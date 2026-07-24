"""Explicit deterministic baseline candidate provenance and safety."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal
from uuid import UUID

import pytest

from packages.execution import MarketSnapshot
from workers.candidate.fallback import DeterministicBaselineCandidateProducer
from workers.contracts import DETERMINISTIC_BASELINE_MODEL
from workers.features import FeatureSnapshot


NOW = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)


def feature(
    *,
    trend: Literal["up", "down", "flat"] = "up",
    candle_count: int = 10,
    fresh: bool = True,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        input_event_id=UUID("5fd83a44-6283-4560-93c6-a3c8fe3329e2"),
        state_version=candle_count,
        symbol="BTCUSDT",
        observed_at=NOW,
        candle_count=candle_count,
        session_high=Decimal("105"),
        session_low=Decimal("95"),
        session_vwap=Decimal("100"),
        atr=Decimal("1"),
        atr_median=Decimal("1"),
        volatility=Decimal("0.01"),
        ema_9=Decimal("102"),
        ema_21=Decimal("101"),
        ema_50=Decimal("100"),
        rsi_14=Decimal("60"),
        trend_direction=trend,
        trend_strength=Decimal("0.02"),
        support=Decimal("98"),
        resistance=Decimal("104"),
        liquidity_proxy=Decimal("100"),
        market_data_age_seconds=Decimal("0.1"),
        fresh=fresh,
    )


def market() -> MarketSnapshot:
    return MarketSnapshot(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        bid=Decimal("99.99"),
        ask=Decimal("100.01"),
        observed_at=NOW,
    )


@pytest.mark.asyncio
async def test_baseline_candidate_has_explicit_non_model_provenance() -> None:
    producer = DeterministicBaselineCandidateProducer(artifact_digest="b" * 64)

    event = await producer.generate(
        feature(),
        market(),
        generated_at=NOW + timedelta(milliseconds=100),
    )

    assert event is not None
    assert event.provider == "deterministic_baseline"
    assert event.model_role is None
    assert event.model == DETERMINISTIC_BASELINE_MODEL
    assert event.model_digest == "b" * 64
    assert event.candidate.strategy_id == DETERMINISTIC_BASELINE_MODEL
    assert event.candidate.side == "buy"
    assert event.candidate.requested_risk_fraction == Decimal("0.0025")
    assert event.candidate.reference_price % Decimal("0.01") == 0
    assert event.candidate.stop_price % Decimal("0.01") == 0
    assert event.candidate.take_profit_price is not None
    assert event.candidate.take_profit_price % Decimal("0.01") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        feature(trend="flat"),
        feature(candle_count=4),
        feature(fresh=False),
    ],
)
async def test_baseline_candidate_fails_closed_without_established_fresh_trend(
    snapshot: FeatureSnapshot,
) -> None:
    producer = DeterministicBaselineCandidateProducer(artifact_digest="b" * 64)

    assert (
        await producer.generate(
            snapshot,
            market(),
            generated_at=NOW + timedelta(milliseconds=100),
        )
        is None
    )
