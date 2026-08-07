"""Canonical alpha replay uses next-bar, costed, conservative execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import hashlib

import pytest

from packages.alpha_certification.data import HistoricalCandle
from packages.alpha_certification.replay import (
    CanonicalAlphaReplay,
    CanonicalCandidateAdapter,
    ReplayConfig,
    ReplayCostModel,
)
from packages.domain import SourceMode
from packages.execution import MarketSnapshot
from packages.risk import (
    CandidateSignal,
    InstrumentConstraints,
    RiskPolicy,
)
from workers.candidate.fallback import DeterministicBaselineCandidateProducer
from workers.features import FeatureSnapshot


def _candle(
    at: datetime,
    *,
    open_price: str,
    high: str,
    low: str,
    close: str,
) -> HistoricalCandle:
    return HistoricalCandle(
        open_time=at,
        close_time=at + timedelta(hours=1),
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("100"),
        quote_volume=Decimal(close) * Decimal("100"),
        trade_count=10,
    )


def _config(max_holding_bars: int = 4) -> ReplayConfig:
    return ReplayConfig(
        symbol="BTCUSDT",
        interval="1h",
        interval_seconds=3_600,
        constraints=InstrumentConstraints(
            venue="binance",
            market_type="spot",
            instrument="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.00001"),
            min_quantity=Decimal("0.00001"),
            min_notional=Decimal("10"),
        ),
        risk_policy=RiskPolicy(
            version="alpha-test",
            min_confidence=Decimal("0.55"),
        ),
        costs=ReplayCostModel(
            taker_fee_bps=Decimal("4"),
            spread_bps=Decimal("4"),
            slippage_bps=Decimal("3"),
        ),
        max_holding_bars=max_holding_bars,
    )


class _OneCandidate:
    def __init__(self) -> None:
        self.sent = False

    async def generate_signal(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
    ) -> CandidateSignal | None:
        if self.sent:
            return None
        self.sent = True
        return CandidateSignal(
            trace_id=f"one:{feature.input_event_id}",
            signal_id=f"sig_{str(feature.input_event_id).replace('-', '')[:24]}",
            strategy_id="test-one",
            strategy_version="1",
            venue=market.venue,
            market_type=market.market_type,
            instrument=market.instrument,
            side="buy",
            reference_price=Decimal("100"),
            stop_price=Decimal("99"),
            take_profit_price=Decimal("101"),
            confidence=Decimal("0.8"),
            spread_bps=Decimal("4"),
            expected_slippage_bps=Decimal("2"),
            data_age_seconds=Decimal(
                str((generated_at - market.observed_at).total_seconds())
            ),
            source_mode=SourceMode.REPLAY,
            requested_risk_fraction=Decimal("0.005"),
        )


class _FeatureCountRecorder:
    def __init__(self) -> None:
        self.candle_counts: list[int] = []

    async def generate_signal(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
    ) -> CandidateSignal | None:
        self.candle_counts.append(feature.candle_count)
        return None


class _LastBarCandidate(_OneCandidate):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate_signal(
        self,
        feature: FeatureSnapshot,
        market: MarketSnapshot,
        *,
        generated_at: datetime,
    ) -> CandidateSignal | None:
        self.calls += 1
        if self.calls < 2:
            return None
        return await super().generate_signal(
            feature,
            market,
            generated_at=generated_at,
        )


@pytest.mark.asyncio
async def test_next_bar_fill_and_same_bar_stop_first_are_enforced() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = (
        _candle(start, open_price="100", high="100.5", low="99.5", close="100"),
        _candle(
            start + timedelta(hours=1),
            open_price="100",
            high="102",
            low="98",
            close="100",
        ),
        _candle(
            start + timedelta(hours=2),
            open_price="100",
            high="101",
            low="99",
            close="100",
        ),
    )
    result = await CanonicalAlphaReplay(_config()).run(candles, _OneCandidate())

    assert result.metrics.trades == 1
    trade = result.trades[0]
    assert trade.entry_time == candles[1].open_time
    assert trade.exit_reason == "stop"
    assert trade.exit_price < Decimal("99")
    assert trade.fees > 0
    assert result.metrics.next_bar_fill_violations == 0
    assert result.metrics.duplicate_fills == 0
    assert result.metrics.reconciliation_mismatches == 0


@pytest.mark.asyncio
async def test_replay_warms_features_without_evaluating_warmup_candles() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = tuple(
        _candle(
            start + timedelta(hours=index),
            open_price=str(100 + index),
            high=str(101 + index),
            low=str(99 + index),
            close=str(100 + index),
        )
        for index in range(52)
    )
    producer = _FeatureCountRecorder()

    result = await CanonicalAlphaReplay(_config()).run(
        candles[50:],
        producer,
        warmup_candles=candles[:50],
    )

    assert producer.candle_counts == [51, 52]
    assert result.metrics.bars == 2
    assert result.metrics.candidates == 0


@pytest.mark.asyncio
async def test_last_bar_candidate_is_never_filled_in_the_same_bar() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = (
        _candle(start, open_price="100", high="101", low="99", close="100"),
        _candle(
            start + timedelta(hours=1),
            open_price="100",
            high="101",
            low="99",
            close="100",
        ),
    )

    result = await CanonicalAlphaReplay(_config()).run(
        candles,
        _LastBarCandidate(),
    )

    assert result.metrics.candidates == 1
    assert result.metrics.trades == 0
    assert result.metrics.stale_pending_orders == 1
    assert result.metrics.next_bar_fill_violations == 0


@pytest.mark.asyncio
async def test_canonical_candidate_adapter_is_deterministic() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = tuple(
        _candle(
            start + timedelta(hours=index),
            open_price=str(100 + index),
            high=str(101.5 + index),
            low=str(99.5 + index),
            close=str(101 + index),
        )
        for index in range(30)
    )
    digest = hashlib.sha256(b"deterministic-baseline-test").hexdigest()

    async def run_once():
        producer = DeterministicBaselineCandidateProducer(artifact_digest=digest)
        return await CanonicalAlphaReplay(_config(max_holding_bars=3)).run(
            candles,
            CanonicalCandidateAdapter(producer),
        )

    first = await run_once()
    second = await run_once()
    assert first.trace_digest == second.trace_digest
    assert first.metrics.duplicate_signals == 0
    assert first.metrics.duplicate_fills == 0
    assert first.metrics.next_bar_fill_violations == 0
