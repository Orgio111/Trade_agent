"""Validated candle to deterministic risk-decision integration tests."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from packages.domain import CandlePayload, MarketEvent, SourceMode
from packages.event_bus import CoreSubject, InMemoryEventBus
from packages.risk import (
    InstrumentConstraints,
    PortfolioState,
    RiskEngine,
    RiskPolicy,
)
from packages.strategies import BaselineSmaStrategy
from workers.decision import DecisionWorker
from workers.market_data import MarketDataWorker


def _event(index: int, close: str) -> MarketEvent:
    open_time = datetime(2026, 7, 15, 0, index, tzinfo=UTC)
    close_time = open_time + timedelta(minutes=1)
    close_price = Decimal(close)
    payload = CandlePayload(
        interval="1m",
        open_time=open_time,
        close_time=close_time,
        open=close_price - Decimal("0.25"),
        high=close_price + Decimal("0.50"),
        low=close_price - Decimal("0.50"),
        close=close_price,
        volume=Decimal("100"),
        quote_volume=close_price * Decimal("100"),
        trade_count=10,
    )
    return MarketEvent.create(
        trace_id="replay-btc-001",
        event_type="market.candle",
        venue="binance",
        market_type="spot",
        instrument_id="BTCUSDT",
        exchange_ts=close_time,
        received_ts=close_time + timedelta(milliseconds=100),
        source_mode=SourceMode.REPLAY,
        ingest_run_id="fixture-001",
        payload=payload,
        sequence_start=index + 1,
        sequence_end=index + 1,
    )


def _portfolio(at: datetime) -> PortfolioState:
    return PortfolioState(
        account_id="paper-main",
        state_id=f"portfolio-{at.timestamp()}",
        equity=Decimal("10000"),
        cash=Decimal("10000"),
        daily_pnl=Decimal("0"),
        weekly_pnl=Decimal("0"),
        consecutive_losses=0,
        open_positions=0,
        gross_exposure=Decimal("0"),
        reconciled_at=at,
        kill_switch_active=False,
    )


def test_validated_candles_produce_one_approved_candidate() -> None:
    bus = InMemoryEventBus()
    market_worker = MarketDataWorker(bus)
    constraints = InstrumentConstraints(
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.001"),
        min_quantity=Decimal("0.001"),
        min_notional=Decimal("10"),
    )
    decision_worker = DecisionWorker(
        bus,
        BaselineSmaStrategy(),
        RiskEngine(
            RiskPolicy(
                version="paper-v1",
                min_confidence=Decimal("0.55"),
            )
        ),
    )

    outcomes = []
    for index, close in enumerate(("100", "101", "102", "103", "104")):
        event = _event(index, close)
        validated = market_worker.process(event, now=event.received_ts)
        assert validated.verdict.accepted
        outcome = decision_worker.process(
            event,
            portfolio=_portfolio(event.received_ts),
            constraints=constraints,
            evaluated_at=event.received_ts,
            spread_bps=Decimal("2"),
            expected_slippage_bps=Decimal("3"),
        )
        if outcome is not None:
            outcomes.append(outcome)

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.candidate.side == "buy"
    assert outcome.decision.approved
    assert outcome.decision.approved_quantity > 0
    assert len(bus.events_for(CoreSubject.MARKET_VALIDATED)) == 5
    assert len(bus.events_for(CoreSubject.SIGNAL_CANDIDATE)) == 1
    assert len(bus.events_for(CoreSubject.RISK_APPROVED)) == 1
    assert len(bus.events_for(CoreSubject.RISK_REJECTED)) == 0


def test_sequence_gap_stops_the_decision_path() -> None:
    bus = InMemoryEventBus()
    market_worker = MarketDataWorker(bus)
    first = _event(0, "100")
    assert market_worker.process(first, now=first.received_ts).verdict.accepted

    gap = MarketEvent.create(
        trace_id=first.trace_id,
        event_type=first.event_type,
        venue=first.venue,
        market_type=first.market_type,
        instrument_id=first.instrument_id,
        exchange_ts=first.exchange_ts + timedelta(minutes=2),
        received_ts=first.received_ts + timedelta(minutes=2),
        source_mode=first.source_mode,
        ingest_run_id=first.ingest_run_id,
        payload=CandlePayload(
            interval="1m",
            open_time=first.payload.open_time + timedelta(minutes=2),
            close_time=first.payload.close_time + timedelta(minutes=2),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("101"),
            volume=Decimal("100"),
        ),
        sequence_start=3,
        sequence_end=3,
    )
    result = market_worker.process(gap, now=gap.received_ts)

    assert not result.verdict.accepted
    assert "SEQUENCE_GAP" in result.verdict.reason_codes
    assert len(bus.events_for(CoreSubject.MARKET_REJECTED)) == 1
