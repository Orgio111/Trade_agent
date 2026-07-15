"""Table-driven boundary tests for the canonical deterministic risk engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from packages.domain.events import SourceMode
from packages.risk import (
    CandidateSignal,
    InstrumentConstraints,
    PortfolioState,
    RiskEngine,
    RiskPolicy,
    RiskReason,
)


D = Decimal
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def make_policy(**overrides: object) -> RiskPolicy:
    values: dict[str, object] = {
        "version": "paper-v1",
        "paper_only": True,
        "max_risk_per_trade_fraction": D("0.005"),
        "max_daily_loss_fraction": D("0.02"),
        "max_weekly_loss_fraction": D("0.04"),
        "max_consecutive_losses": 3,
        "max_open_positions": 3,
        "max_leverage": D("1"),
        "max_data_age_seconds": D("5"),
        "max_portfolio_age_seconds": D("30"),
        "max_spread_bps": D("10"),
        "max_slippage_bps": D("10"),
        "min_confidence": D("0.60"),
        "min_notional": D("0"),
    }
    values.update(overrides)
    return RiskPolicy(**values)


def make_candidate(**overrides: object) -> CandidateSignal:
    values: dict[str, object] = {
        "trace_id": "trace-001",
        "signal_id": "signal-001",
        "strategy_id": "baseline",
        "strategy_version": "1.0.0",
        "venue": "binance",
        "market_type": "spot",
        "instrument": "BTCUSDT",
        "side": "buy",
        "reference_price": D("100"),
        "stop_price": D("95"),
        "take_profit_price": D("110"),
        "confidence": D("0.80"),
        "spread_bps": D("2"),
        "expected_slippage_bps": D("3"),
        "data_age_seconds": D("1"),
        "source_mode": SourceMode.PAPER_LIVE,
        "requested_risk_fraction": D("0.005"),
    }
    values.update(overrides)
    return CandidateSignal(**values)


def make_portfolio(**overrides: object) -> PortfolioState:
    values: dict[str, object] = {
        "account_id": "paper-main",
        "state_id": "portfolio-001",
        "equity": D("10000"),
        "cash": D("10000"),
        "daily_pnl": D("0"),
        "weekly_pnl": D("0"),
        "consecutive_losses": 0,
        "open_positions": 0,
        "gross_exposure": D("0"),
        "reconciled_at": NOW - timedelta(seconds=5),
        "kill_switch_active": False,
    }
    values.update(overrides)
    return PortfolioState(**values)


def make_constraints(**overrides: object) -> InstrumentConstraints:
    values: dict[str, object] = {
        "venue": "binance",
        "market_type": "spot",
        "instrument": "BTCUSDT",
        "tick_size": D("0.1"),
        "step_size": D("0.01"),
        "min_quantity": D("0.001"),
        "min_notional": D("10"),
    }
    values.update(overrides)
    return InstrumentConstraints(**values)


def evaluate(
    *,
    policy: RiskPolicy | None = None,
    candidate: CandidateSignal | None = None,
    portfolio: PortfolioState | None = None,
    constraints: InstrumentConstraints | None = None,
    manual_risk_fraction: Decimal | None = None,
):
    return RiskEngine(policy or make_policy()).evaluate(
        candidate or make_candidate(),
        portfolio or make_portfolio(),
        constraints or make_constraints(),
        evaluated_at=NOW,
        manual_risk_fraction=manual_risk_fraction,
    )


def assert_rejected(decision, reason: RiskReason) -> None:
    assert decision.approved is False
    assert decision.reason_codes == (reason,)
    assert decision.approved_quantity == D("0")
    assert decision.approved_notional == D("0")
    assert decision.approved_risk_amount == D("0")
    assert decision.approved_risk_fraction == D("0")


def test_correct_decimal_stop_distance_sizing() -> None:
    decision = evaluate()

    # 10,000 * 0.5% = 50 risk; 100 - 95 = 5 stop distance; qty = 10.
    assert decision.approved is True
    assert decision.reason_codes == (RiskReason.APPROVED,)
    assert decision.approved_quantity == D("10")
    assert decision.approved_notional == D("1000")
    assert decision.approved_risk_amount == D("50")
    assert decision.approved_risk_fraction == D("0.005")
    assert decision.policy_version == "paper-v1"
    assert decision.portfolio_state_id == "portfolio-001"
    assert decision.account_id == "paper-main"
    assert decision.candidate_hash == make_candidate().content_hash()


def test_sell_uses_absolute_stop_distance_and_directional_stop() -> None:
    candidate = make_candidate(
        side="sell",
        stop_price=D("105"),
        take_profit_price=D("90"),
    )
    decision = evaluate(candidate=candidate)

    assert decision.approved
    assert decision.approved_quantity == D("10")
    assert decision.approved_risk_amount == D("50")


def test_quantity_is_rounded_down_never_up() -> None:
    decision = evaluate(
        portfolio=make_portfolio(equity=D("1000"), cash=D("1000")),
        candidate=make_candidate(stop_price=D("97")),
    )

    assert decision.approved
    assert decision.approved_quantity == D("1.66")
    assert decision.approved_risk_amount == D("4.98")
    assert decision.approved_risk_fraction == D("0.00498")
    assert RiskReason.QUANTITY_STEP_ROUNDED in decision.reason_codes


@pytest.mark.parametrize(
    ("candidate", "portfolio", "expected"),
    [
        (
            make_candidate(source_mode=SourceMode.HISTORICAL),
            make_portfolio(),
            RiskReason.NON_EXECUTABLE_SOURCE_MODE,
        ),
        (
            make_candidate(source_mode=SourceMode.LIVE),
            make_portfolio(kill_switch_active=True),
            RiskReason.LIVE_MODE_BLOCKED,
        ),
        (
            make_candidate(),
            make_portfolio(kill_switch_active=True),
            RiskReason.KILL_SWITCH_ACTIVE,
        ),
        (
            make_candidate(data_age_seconds=None),
            make_portfolio(),
            RiskReason.MISSING_MARKET_DATA,
        ),
        (
            make_candidate(data_age_seconds=D("5.0001")),
            make_portfolio(),
            RiskReason.STALE_MARKET_DATA,
        ),
        (
            make_candidate(),
            make_portfolio(reconciled_at=None),
            RiskReason.MISSING_PORTFOLIO_STATE,
        ),
        (
            make_candidate(),
            make_portfolio(reconciled_at=NOW - timedelta(seconds=31)),
            RiskReason.STALE_PORTFOLIO_STATE,
        ),
        (
            make_candidate(),
            make_portfolio(reconciled_at=NOW + timedelta(microseconds=1)),
            RiskReason.PORTFOLIO_STATE_FROM_FUTURE,
        ),
    ],
)
def test_ordered_mode_kill_and_freshness_gates(
    candidate: CandidateSignal,
    portfolio: PortfolioState,
    expected: RiskReason,
) -> None:
    assert_rejected(
        evaluate(candidate=candidate, portfolio=portfolio),
        expected,
    )


def test_live_is_allowed_only_when_policy_is_explicitly_promoted() -> None:
    decision = evaluate(
        policy=make_policy(paper_only=False),
        candidate=make_candidate(source_mode=SourceMode.LIVE),
    )
    assert decision.approved


@pytest.mark.parametrize(
    ("portfolio_overrides", "expected"),
    [
        ({"daily_pnl": D("-200")}, RiskReason.DAILY_LOSS_LIMIT_REACHED),
        ({"weekly_pnl": D("-400")}, RiskReason.WEEKLY_LOSS_LIMIT_REACHED),
        (
            {"consecutive_losses": 3},
            RiskReason.CONSECUTIVE_LOSS_LIMIT_REACHED,
        ),
        ({"open_positions": 3}, RiskReason.OPEN_POSITION_LIMIT_REACHED),
        ({"gross_exposure": D("10000")}, RiskReason.MAX_LEVERAGE_EXCEEDED),
    ],
)
def test_portfolio_limits_are_hard_at_the_boundary(
    portfolio_overrides: dict[str, object], expected: RiskReason
) -> None:
    assert_rejected(
        evaluate(portfolio=make_portfolio(**portfolio_overrides)),
        expected,
    )


@pytest.mark.parametrize(
    ("candidate_overrides", "expected"),
    [
        ({"confidence": None}, RiskReason.MISSING_CONFIDENCE),
        ({"confidence": D("0.5999")}, RiskReason.CONFIDENCE_BELOW_MINIMUM),
        ({"spread_bps": None}, RiskReason.MISSING_SPREAD),
        ({"spread_bps": D("10.0001")}, RiskReason.SPREAD_LIMIT_EXCEEDED),
        ({"expected_slippage_bps": None}, RiskReason.MISSING_SLIPPAGE),
        (
            {"expected_slippage_bps": D("10.0001")},
            RiskReason.SLIPPAGE_LIMIT_EXCEEDED,
        ),
    ],
)
def test_market_quality_is_fail_closed(
    candidate_overrides: dict[str, object], expected: RiskReason
) -> None:
    assert_rejected(
        evaluate(candidate=make_candidate(**candidate_overrides)),
        expected,
    )


@pytest.mark.parametrize(
    "candidate",
    [
        make_candidate(stop_price=D("100")),
        make_candidate(stop_price=D("101")),
        make_candidate(side="sell", stop_price=D("99"), take_profit_price=D("90")),
    ],
)
def test_invalid_directional_stop_is_rejected(candidate: CandidateSignal) -> None:
    assert_rejected(evaluate(candidate=candidate), RiskReason.INVALID_STOP)


@pytest.mark.parametrize(
    "candidate",
    [
        make_candidate(take_profit_price=D("99")),
        make_candidate(
            side="sell", stop_price=D("105"), take_profit_price=D("101")
        ),
    ],
)
def test_invalid_directional_take_profit_is_rejected(
    candidate: CandidateSignal,
) -> None:
    assert_rejected(evaluate(candidate=candidate), RiskReason.INVALID_TAKE_PROFIT)


def test_executable_prices_must_be_tick_aligned() -> None:
    decision = evaluate(candidate=make_candidate(stop_price=D("95.05")))
    assert_rejected(decision, RiskReason.PRICE_NOT_TICK_ALIGNED)


def test_risk_sized_order_below_venue_minimum_is_not_rounded_up() -> None:
    decision = evaluate(
        portfolio=make_portfolio(equity=D("10"), cash=D("10")),
    )
    assert_rejected(decision, RiskReason.MIN_NOTIONAL_NOT_MET)


def test_quantity_below_venue_minimum_is_rejected() -> None:
    decision = evaluate(
        constraints=make_constraints(min_quantity=D("11")),
    )
    assert_rejected(decision, RiskReason.MIN_QUANTITY_NOT_MET)


def test_policy_minimum_notional_can_be_stricter_than_venue() -> None:
    decision = evaluate(policy=make_policy(min_notional=D("1001")))
    assert_rejected(decision, RiskReason.MIN_NOTIONAL_NOT_MET)


def test_exposure_cap_reduces_quantity_without_crossing_max_leverage() -> None:
    portfolio = make_portfolio(gross_exposure=D("9990"))
    decision = evaluate(portfolio=portfolio)

    assert decision.approved
    assert decision.approved_quantity == D("0.1")
    assert decision.approved_notional == D("10.0")
    assert portfolio.gross_exposure + decision.approved_notional == D("10000.0")
    assert RiskReason.EXPOSURE_CAPPED in decision.reason_codes


def test_policy_caps_requested_risk() -> None:
    decision = evaluate(
        candidate=make_candidate(requested_risk_fraction=D("0.01"))
    )

    assert decision.approved
    assert decision.approved_quantity == D("10")
    assert RiskReason.POLICY_RISK_CAPPED in decision.reason_codes


def test_manual_risk_cap_only_reduces_quantity() -> None:
    decision = evaluate(manual_risk_fraction=D("0.002"))

    assert decision.approved
    assert decision.approved_quantity == D("4")
    assert decision.approved_risk_amount == D("20")
    assert decision.approved_risk_fraction == D("0.002")
    assert RiskReason.MANUAL_RISK_REDUCED in decision.reason_codes


@pytest.mark.parametrize(
    ("manual_risk", "expected"),
    [
        (D("0.006"), RiskReason.MANUAL_RISK_INCREASE_NOT_ALLOWED),
        (D("0"), RiskReason.MANUAL_RISK_DISABLED),
        (D("-0.001"), RiskReason.INVALID_MANUAL_RISK),
        (D("NaN"), RiskReason.INVALID_MANUAL_RISK),
        ("0.002", RiskReason.INVALID_MANUAL_RISK),
    ],
)
def test_manual_override_cannot_increase_or_bypass_risk(
    manual_risk: object, expected: RiskReason
) -> None:
    decision = RiskEngine(make_policy()).evaluate(
        make_candidate(),
        make_portfolio(),
        make_constraints(),
        evaluated_at=NOW,
        manual_risk_fraction=manual_risk,  # type: ignore[arg-type]
    )
    assert_rejected(decision, expected)


def test_instrument_constraints_must_match_candidate() -> None:
    decision = evaluate(constraints=make_constraints(instrument="ETHUSDT"))
    assert_rejected(decision, RiskReason.INSTRUMENT_MISMATCH)


@pytest.mark.parametrize(
    ("constraints", "expected"),
    [
        (make_constraints(venue="bybit"), RiskReason.VENUE_MISMATCH),
        (make_constraints(market_type="linear"), RiskReason.MARKET_TYPE_MISMATCH),
    ],
)
def test_venue_identity_must_match_constraints(
    constraints: InstrumentConstraints, expected: RiskReason
) -> None:
    assert_rejected(evaluate(constraints=constraints), expected)


def test_decision_is_frozen_and_deterministic() -> None:
    first = evaluate()
    second = evaluate()

    assert first == second
    assert first.decision_id == second.decision_id
    assert first.decision_id.startswith("risk_")
    with pytest.raises(ValidationError):
        first.approved_quantity = D("999")  # type: ignore[misc]


@pytest.mark.parametrize("precision", [6, 10, 28, 50])
def test_risk_decision_is_independent_of_ambient_decimal_context(
    precision: int,
) -> None:
    expected = evaluate()
    with localcontext() as context:
        context.prec = precision
        actual = evaluate()
    assert actual == expected
    assert actual.decision_id == expected.decision_id


def test_decision_schema_rejects_inconsistent_verdict_values() -> None:
    values = evaluate().model_dump()
    values["approved"] = False
    with pytest.raises(ValidationError, match="rejected decision"):
        type(evaluate())(**values)


def test_a_policy_change_changes_the_decision_id() -> None:
    first = evaluate()
    second = evaluate(policy=make_policy(version="paper-v2"))
    assert first.decision_id != second.decision_id


@pytest.mark.parametrize(
    ("factory", "overrides"),
    [
        (make_candidate, {"reference_price": D("0")}),
        (make_candidate, {"requested_risk_fraction": D("0")}),
        (make_candidate, {"confidence": D("1.01")}),
        (make_candidate, {"data_age_seconds": D("-0.1")}),
        (make_candidate, {"side": "hold"}),
        (make_portfolio, {"equity": D("0")}),
        (make_portfolio, {"gross_exposure": D("-1")}),
        (make_portfolio, {"daily_pnl": D("NaN")}),
        (make_portfolio, {"reconciled_at": datetime(2026, 7, 15, 12, 0)}),
        (make_constraints, {"step_size": D("0")}),
        (make_policy, {"max_leverage": D("0")}),
    ],
)
def test_invalid_numeric_and_enum_inputs_never_reach_the_engine(
    factory, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        factory(**overrides)


def test_missing_required_input_is_rejected_by_schema() -> None:
    values = make_candidate().model_dump()
    del values["spread_bps"]
    with pytest.raises(ValidationError):
        CandidateSignal(**values)


def test_extra_input_is_forbidden_instead_of_silently_ignored() -> None:
    values = make_candidate().model_dump()
    values["broker_override"] = "bypass"
    with pytest.raises(ValidationError):
        CandidateSignal(**values)


def test_evaluated_at_must_be_explicit_and_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RiskEngine(make_policy()).evaluate(
            make_candidate(),
            make_portfolio(),
            make_constraints(),
            evaluated_at=datetime(2026, 7, 15, 12, 0),
        )


def test_reason_code_values_are_stable_strings() -> None:
    assert RiskReason.APPROVED.value == "APPROVED"
    assert RiskReason.KILL_SWITCH_ACTIVE.value == "KILL_SWITCH_ACTIVE"
    assert RiskReason.MIN_NOTIONAL_NOT_MET.value == "MIN_NOTIONAL_NOT_MET"
