"""Golden replay tests for the deterministic paper vertical slice."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

from packages.domain import CandlePayload, MarketEvent, SourceMode
from packages.replay import ReplayHarness
from packages.risk import InstrumentConstraints, RiskPolicy


def _events(mode: SourceMode = SourceMode.REPLAY) -> list[MarketEvent]:
    events: list[MarketEvent] = []
    for index, close_text in enumerate(("100", "101", "102", "103", "104", "105")):
        open_time = datetime(2026, 7, 15, 1, index, tzinfo=UTC)
        close_time = open_time + timedelta(minutes=1)
        close = Decimal(close_text)
        events.append(
            MarketEvent.create(
                trace_id="golden-replay-001",
                event_type="market.candle",
                venue="binance",
                market_type="spot",
                instrument_id="BTCUSDT",
                exchange_ts=close_time,
                received_ts=close_time + timedelta(milliseconds=50),
                source_mode=mode,
                ingest_run_id="golden-fixture-001",
                payload=CandlePayload(
                    interval="1m",
                    open_time=open_time,
                    close_time=close_time,
                    open=close - Decimal("0.25"),
                    high=close + Decimal("0.50"),
                    low=close - Decimal("0.50"),
                    close=close,
                    volume=Decimal("100"),
                    quote_volume=close * Decimal("100"),
                    trade_count=10,
                ),
                sequence_start=index + 1,
                sequence_end=index + 1,
            )
        )
    return events


def _harness() -> ReplayHarness:
    return ReplayHarness(
        constraints=InstrumentConstraints(
            venue="binance",
            market_type="spot",
            instrument="BTCUSDT",
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.001"),
            min_quantity=Decimal("0.001"),
            min_notional=Decimal("10"),
        ),
        risk_policy=RiskPolicy(
            version="paper-v1",
            min_confidence=Decimal("0.55"),
        ),
    )


def test_golden_replay_is_deterministic_and_reconciled() -> None:
    first = _harness().run(_events())
    second = _harness().run(_events())

    assert first.accepted_market_events == 6
    assert first.rejected_market_events == 0
    assert first.candidates == 1
    assert first.approved_decisions == 1
    assert first.rejected_decisions == 0
    assert first.orders == 1
    assert first.fills == 1
    assert first.reconciliation_clean
    assert first.trace_digest == second.trace_digest


def test_replay_is_independent_of_ambient_decimal_precision() -> None:
    expected = _harness().run(_events())
    for precision in (6, 10, 28, 50):
        with localcontext() as context:
            context.prec = precision
            actual = _harness().run(_events())
        assert actual.trace_digest == expected.trace_digest


def test_live_event_is_rejected_by_risk_before_execution() -> None:
    summary = _harness().run(_events(SourceMode.LIVE))

    assert summary.candidates == 1
    assert summary.approved_decisions == 0
    assert summary.rejected_decisions == 1
    assert summary.orders == 0
    assert summary.fills == 0
    assert summary.reconciliation_clean
