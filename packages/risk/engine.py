"""Pure, ordered, fail-closed risk evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN, localcontext
from typing import Any

from packages.domain.events import SourceMode

from .policy import InstrumentConstraints, RiskPolicy
from .state import CandidateSignal, PortfolioState, RiskDecision, RiskReason


ZERO = Decimal("0")
ONE = Decimal("1")


class RiskEngine:
    """Evaluate candidate signals without I/O, clocks, brokers, or models.

    The caller supplies ``evaluated_at`` so the same inputs always produce the
    same decision and decision ID. Gates return at the first hard failure.
    """

    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        candidate: CandidateSignal,
        portfolio: PortfolioState,
        constraints: InstrumentConstraints,
        *,
        evaluated_at: datetime,
        manual_risk_fraction: Decimal | None = None,
    ) -> RiskDecision:
        """Evaluate inside a fixed context, independent of ambient precision."""

        with localcontext() as context:
            context.prec = 50
            return self._evaluate(
                candidate,
                portfolio,
                constraints,
                evaluated_at=evaluated_at,
                manual_risk_fraction=manual_risk_fraction,
            )

    def _evaluate(
        self,
        candidate: CandidateSignal,
        portfolio: PortfolioState,
        constraints: InstrumentConstraints,
        *,
        evaluated_at: datetime,
        manual_risk_fraction: Decimal | None = None,
    ) -> RiskDecision:
        """Return a deterministic approval or reason-coded rejection."""

        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        evaluated_at = evaluated_at.astimezone(UTC)

        # 1. Deployment mode and promotion gate.
        if candidate.source_mode == SourceMode.HISTORICAL:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.NON_EXECUTABLE_SOURCE_MODE,
            )
        if self.policy.paper_only and candidate.source_mode == SourceMode.LIVE:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.LIVE_MODE_BLOCKED
            )

        # 2. Durable kill switch / manual lock.
        if portfolio.kill_switch_active:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.KILL_SWITCH_ACTIVE
            )

        # 3. Required data and staleness.
        if candidate.data_age_seconds is None:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.MISSING_MARKET_DATA
            )
        if candidate.data_age_seconds > self.policy.max_data_age_seconds:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.STALE_MARKET_DATA
            )

        # 4. Account reconciliation freshness.
        if portfolio.reconciled_at is None:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.MISSING_PORTFOLIO_STATE
            )
        portfolio_age = Decimal(
            str((evaluated_at - portfolio.reconciled_at).total_seconds())
        )
        if portfolio_age < ZERO:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.PORTFOLIO_STATE_FROM_FUTURE,
            )
        if portfolio_age > self.policy.max_portfolio_age_seconds:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.STALE_PORTFOLIO_STATE
            )

        # 5. Instrument / venue permission and constraint match.
        if candidate.instrument != constraints.instrument:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.INSTRUMENT_MISMATCH
            )
        if candidate.venue != constraints.venue:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.VENUE_MISMATCH
            )
        if candidate.market_type != constraints.market_type:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.MARKET_TYPE_MISMATCH
            )

        # 6. Portfolio loss and position-count limits.
        if portfolio.daily_pnl <= -(
            portfolio.equity * self.policy.max_daily_loss_fraction
        ):
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.DAILY_LOSS_LIMIT_REACHED,
            )
        if portfolio.weekly_pnl <= -(
            portfolio.equity * self.policy.max_weekly_loss_fraction
        ):
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.WEEKLY_LOSS_LIMIT_REACHED,
            )
        if portfolio.consecutive_losses >= self.policy.max_consecutive_losses:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.CONSECUTIVE_LOSS_LIMIT_REACHED,
            )
        if portfolio.open_positions >= self.policy.max_open_positions:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.OPEN_POSITION_LIMIT_REACHED,
            )

        exposure_limit = portfolio.equity * self.policy.max_leverage
        exposure_capacity = exposure_limit - portfolio.gross_exposure
        if exposure_capacity <= ZERO:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.MAX_LEVERAGE_EXCEEDED,
            )

        # 7. Market-quality inputs and hard limits.
        if candidate.confidence is None:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.MISSING_CONFIDENCE
            )
        if candidate.confidence < self.policy.min_confidence:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.CONFIDENCE_BELOW_MINIMUM,
            )
        if candidate.spread_bps is None:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.MISSING_SPREAD
            )
        if candidate.spread_bps > self.policy.max_spread_bps:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.SPREAD_LIMIT_EXCEEDED
            )
        if candidate.expected_slippage_bps is None:
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.MISSING_SLIPPAGE
            )
        if candidate.expected_slippage_bps > self.policy.max_slippage_bps:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.SLIPPAGE_LIMIT_EXCEEDED,
            )

        # 8. Directional stop/target validity and executable price precision.
        if not self._valid_stop(candidate):
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.INVALID_STOP
            )
        if not self._valid_take_profit(candidate):
            return self._reject(
                candidate, portfolio, evaluated_at, RiskReason.INVALID_TAKE_PROFIT
            )
        if not self._prices_are_tick_aligned(candidate, constraints.tick_size):
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.PRICE_NOT_TICK_ALIGNED,
            )

        automatic_risk_fraction = min(
            candidate.requested_risk_fraction,
            self.policy.max_risk_per_trade_fraction,
        )
        notes: list[RiskReason] = []
        if candidate.requested_risk_fraction > automatic_risk_fraction:
            notes.append(RiskReason.POLICY_RISK_CAPPED)

        if manual_risk_fraction is not None:
            if (
                not isinstance(manual_risk_fraction, Decimal)
                or not manual_risk_fraction.is_finite()
                or manual_risk_fraction < ZERO
                or manual_risk_fraction > ONE
            ):
                return self._reject(
                    candidate,
                    portfolio,
                    evaluated_at,
                    RiskReason.INVALID_MANUAL_RISK,
                )
            if manual_risk_fraction > automatic_risk_fraction:
                return self._reject(
                    candidate,
                    portfolio,
                    evaluated_at,
                    RiskReason.MANUAL_RISK_INCREASE_NOT_ALLOWED,
                )
            if manual_risk_fraction == ZERO:
                return self._reject(
                    candidate,
                    portfolio,
                    evaluated_at,
                    RiskReason.MANUAL_RISK_DISABLED,
                )
            if manual_risk_fraction < automatic_risk_fraction:
                notes.append(RiskReason.MANUAL_RISK_REDUCED)
            approved_risk_fraction = manual_risk_fraction
        else:
            approved_risk_fraction = automatic_risk_fraction

        # 9. Stop-distance sizing followed by leverage/exposure capping.
        stop_distance = abs(candidate.reference_price - candidate.stop_price)
        risk_budget = portfolio.equity * approved_risk_fraction
        risk_sized_quantity = risk_budget / stop_distance
        exposure_capped_quantity = exposure_capacity / candidate.reference_price
        quantity_before_rounding = min(
            risk_sized_quantity,
            exposure_capped_quantity,
        )
        if exposure_capped_quantity < risk_sized_quantity:
            notes.append(RiskReason.EXPOSURE_CAPPED)

        approved_quantity = self._floor_to_step(
            quantity_before_rounding, constraints.step_size
        )
        if approved_quantity != quantity_before_rounding:
            notes.append(RiskReason.QUANTITY_STEP_ROUNDED)
        if approved_quantity <= ZERO:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.QUANTITY_ROUNDS_TO_ZERO,
            )

        # 10. Venue and policy minima. Never round a risk-sized order up.
        if approved_quantity < constraints.min_quantity:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.MIN_QUANTITY_NOT_MET,
            )
        approved_notional = approved_quantity * candidate.reference_price
        required_notional = max(
            constraints.min_notional,
            self.policy.min_notional,
        )
        if approved_notional < required_notional:
            return self._reject(
                candidate,
                portfolio,
                evaluated_at,
                RiskReason.MIN_NOTIONAL_NOT_MET,
            )

        approved_risk_amount = approved_quantity * stop_distance
        actual_risk_fraction = approved_risk_amount / portfolio.equity
        return self._decision(
            candidate=candidate,
            portfolio=portfolio,
            evaluated_at=evaluated_at,
            approved=True,
            reason_codes=(RiskReason.APPROVED, *notes),
            approved_quantity=approved_quantity,
            approved_notional=approved_notional,
            approved_risk_amount=approved_risk_amount,
            approved_risk_fraction=actual_risk_fraction,
        )

    @staticmethod
    def _valid_stop(candidate: CandidateSignal) -> bool:
        if candidate.side == "buy":
            return candidate.stop_price < candidate.reference_price
        return candidate.stop_price > candidate.reference_price

    @staticmethod
    def _valid_take_profit(candidate: CandidateSignal) -> bool:
        if candidate.take_profit_price is None:
            return True
        if candidate.side == "buy":
            return candidate.take_profit_price > candidate.reference_price
        return candidate.take_profit_price < candidate.reference_price

    @staticmethod
    def _prices_are_tick_aligned(
        candidate: CandidateSignal, tick_size: Decimal
    ) -> bool:
        prices = [candidate.reference_price, candidate.stop_price]
        if candidate.take_profit_price is not None:
            prices.append(candidate.take_profit_price)
        return all(price % tick_size == ZERO for price in prices)

    @staticmethod
    def _floor_to_step(quantity: Decimal, step_size: Decimal) -> Decimal:
        steps = (quantity / step_size).to_integral_value(rounding=ROUND_DOWN)
        return steps * step_size

    def _reject(
        self,
        candidate: CandidateSignal,
        portfolio: PortfolioState,
        evaluated_at: datetime,
        reason: RiskReason,
    ) -> RiskDecision:
        return self._decision(
            candidate=candidate,
            portfolio=portfolio,
            evaluated_at=evaluated_at,
            approved=False,
            reason_codes=(reason,),
            approved_quantity=ZERO,
            approved_notional=ZERO,
            approved_risk_amount=ZERO,
            approved_risk_fraction=ZERO,
        )

    def _decision(
        self,
        *,
        candidate: CandidateSignal,
        portfolio: PortfolioState,
        evaluated_at: datetime,
        approved: bool,
        reason_codes: tuple[RiskReason, ...],
        approved_quantity: Decimal,
        approved_notional: Decimal,
        approved_risk_amount: Decimal,
        approved_risk_fraction: Decimal,
    ) -> RiskDecision:
        payload: dict[str, Any] = {
            "candidate": candidate.model_dump(mode="json"),
            "portfolio": portfolio.model_dump(mode="json"),
            "policy": self.policy.model_dump(mode="json"),
            "evaluated_at": evaluated_at.isoformat(),
            "approved": approved,
            "reason_codes": [reason.value for reason in reason_codes],
            "approved_quantity": str(approved_quantity),
            "approved_notional": str(approved_notional),
            "approved_risk_amount": str(approved_risk_amount),
            "approved_risk_fraction": str(approved_risk_fraction),
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        decision_id = f"risk_{hashlib.sha256(canonical).hexdigest()[:24]}"
        return RiskDecision(
            decision_id=decision_id,
            trace_id=candidate.trace_id,
            signal_id=candidate.signal_id,
            account_id=portfolio.account_id,
            candidate_hash=candidate.content_hash(),
            approved=approved,
            reason_codes=reason_codes,
            approved_quantity=approved_quantity,
            approved_notional=approved_notional,
            approved_risk_amount=approved_risk_amount,
            approved_risk_fraction=approved_risk_fraction,
            policy_version=self.policy.version,
            portfolio_state_id=portfolio.state_id,
            evaluated_at=evaluated_at,
        )
