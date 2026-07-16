"""Deterministic domain fixtures for durable-adapter tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from packages.domain import SourceMode
from packages.execution.models import OrderIntent
from packages.risk import CandidateSignal, PortfolioState, RiskDecision, RiskReason


NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def candidate(*, signal_id: str = "sig-durable-001") -> CandidateSignal:
    return CandidateSignal(
        trace_id="trace-durable-001",
        signal_id=signal_id,
        strategy_id="fixture",
        strategy_version="v1",
        venue="binance",
        market_type="spot",
        instrument="BTCUSDT",
        side="buy",
        reference_price=Decimal("100"),
        stop_price=Decimal("99"),
        take_profit_price=Decimal("102"),
        confidence=Decimal("0.8"),
        spread_bps=Decimal("1"),
        expected_slippage_bps=Decimal("1"),
        data_age_seconds=Decimal("0"),
        source_mode=SourceMode.PAPER_LIVE,
        requested_risk_fraction=Decimal("0.005"),
    )


def portfolio(*, reconciled_at: datetime = NOW) -> PortfolioState:
    return PortfolioState(
        account_id="paper-main",
        state_id="state-durable-001",
        equity=Decimal("10000"),
        cash=Decimal("10000"),
        daily_pnl=Decimal("0"),
        weekly_pnl=Decimal("0"),
        consecutive_losses=0,
        open_positions=0,
        gross_exposure=Decimal("0"),
        reconciled_at=reconciled_at,
        kill_switch_active=False,
    )


def decision(
    *,
    candidate_value: CandidateSignal | None = None,
    approved: bool = True,
) -> RiskDecision:
    signal = candidate_value or candidate()
    return RiskDecision(
        decision_id="risk_0123456789abcdef01234567",
        trace_id=signal.trace_id,
        signal_id=signal.signal_id,
        account_id="paper-main",
        candidate_hash=signal.content_hash(),
        approved=approved,
        reason_codes=(RiskReason.APPROVED,)
        if approved
        else (RiskReason.KILL_SWITCH_ACTIVE,),
        approved_quantity=Decimal("1") if approved else Decimal("0"),
        approved_notional=Decimal("100") if approved else Decimal("0"),
        approved_risk_amount=Decimal("1") if approved else Decimal("0"),
        approved_risk_fraction=Decimal("0.0001") if approved else Decimal("0"),
        policy_version="policy-v1",
        portfolio_state_id="state-durable-001",
        evaluated_at=NOW,
    )


def intent() -> OrderIntent:
    signal = candidate()
    return OrderIntent.from_approved_decision(
        decision(candidate_value=signal),
        signal,
        account_id="paper-main",
        created_at=NOW,
    )
