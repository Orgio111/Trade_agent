"""Risk-approved, paper-only execution worker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

from packages.event_bus import CoreSubject, InMemoryEventBus
from packages.execution import (
    ExecutionResult,
    ExecutionService,
    MarketSnapshot,
    OrderType,
    UnapprovedDecisionError,
)
from packages.risk import CandidateSignal, RiskDecision


class PriceDeviationExceeded(RuntimeError):
    """Raised before intent creation when the executable price is too far away."""


class StaleExecutionContext(RuntimeError):
    """Risk approval or market snapshot is too old/future-dated to execute."""


def validate_execution_context(
    decision: RiskDecision,
    candidate: CandidateSignal,
    *,
    market: MarketSnapshot,
    execution_at: datetime,
    max_price_deviation_bps: Decimal,
    max_decision_age: timedelta,
    max_market_age: timedelta,
) -> datetime:
    """Apply the shared pre-intent safety gate for sync and async workers."""

    if not decision.approved:
        raise UnapprovedDecisionError("execution requires an approved risk decision")
    if execution_at.tzinfo is None or execution_at.utcoffset() is None:
        raise ValueError("execution_at must be timezone-aware")
    execution_utc = execution_at.astimezone(UTC)
    decision_age = execution_utc - decision.evaluated_at
    market_age = execution_utc - market.observed_at
    if (
        decision_age < timedelta(0)
        or decision_age > max_decision_age
        or market_age < timedelta(0)
        or market_age > max_market_age
    ):
        raise StaleExecutionContext(
            "risk decision or market snapshot is outside the execution TTL"
        )
    if (
        market.venue != candidate.venue
        or market.market_type != candidate.market_type
        or market.instrument != candidate.instrument
    ):
        raise ValueError("market snapshot identity does not match candidate")
    executable_price = market.ask if candidate.side == "buy" else market.bid
    with localcontext() as context:
        context.prec = 50
        deviation_bps = (
            abs(executable_price - candidate.reference_price)
            / candidate.reference_price
            * Decimal("10000")
        )
    if deviation_bps > max_price_deviation_bps:
        raise PriceDeviationExceeded(
            f"price deviation {deviation_bps} bps exceeds "
            f"{max_price_deviation_bps} bps"
        )
    return execution_utc


class ExecutionWorker:
    def __init__(
        self,
        bus: InMemoryEventBus,
        service: ExecutionService,
        *,
        account_id: str,
        max_price_deviation_bps: Decimal = Decimal("20"),
        max_decision_age: timedelta = timedelta(seconds=2),
        max_market_age: timedelta = timedelta(seconds=1),
    ) -> None:
        if max_price_deviation_bps < 0:
            raise ValueError("max_price_deviation_bps must be non-negative")
        if max_decision_age < timedelta(0) or max_market_age < timedelta(0):
            raise ValueError("execution freshness limits must be non-negative")
        self.bus = bus
        self.service = service
        self.account_id = account_id
        self.max_price_deviation_bps = max_price_deviation_bps
        self.max_decision_age = max_decision_age
        self.max_market_age = max_market_age

    def process(
        self,
        decision: RiskDecision,
        candidate: CandidateSignal,
        *,
        market: MarketSnapshot,
        execution_at: datetime,
        order_type: OrderType = OrderType.MARKET,
    ) -> ExecutionResult:
        execution_utc = validate_execution_context(
            decision,
            candidate,
            market=market,
            execution_at=execution_at,
            max_price_deviation_bps=self.max_price_deviation_bps,
            max_decision_age=self.max_decision_age,
            max_market_age=self.max_market_age,
        )

        result = self.service.submit(
            decision,
            candidate,
            account_id=self.account_id,
            market=market,
            order_type=order_type,
            created_at=execution_utc,
        )
        if not result.duplicate:
            self.bus.publish(CoreSubject.ORDER_INTENT, result.intent)
            self.bus.publish(CoreSubject.ORDER_UPDATED, result)
        return result
