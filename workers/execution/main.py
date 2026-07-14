"""Risk-approved, paper-only execution worker."""

from __future__ import annotations

from decimal import Decimal

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


class ExecutionWorker:
    def __init__(
        self,
        bus: InMemoryEventBus,
        service: ExecutionService,
        *,
        account_id: str,
        max_price_deviation_bps: Decimal = Decimal("20"),
    ) -> None:
        if max_price_deviation_bps < 0:
            raise ValueError("max_price_deviation_bps must be non-negative")
        self.bus = bus
        self.service = service
        self.account_id = account_id
        self.max_price_deviation_bps = max_price_deviation_bps

    def process(
        self,
        decision: RiskDecision,
        candidate: CandidateSignal,
        *,
        market: MarketSnapshot,
        order_type: OrderType = OrderType.MARKET,
    ) -> ExecutionResult:
        if not decision.approved:
            raise UnapprovedDecisionError("execution requires an approved risk decision")
        executable_price = market.ask if candidate.side == "buy" else market.bid
        deviation_bps = (
            abs(executable_price - candidate.reference_price)
            / candidate.reference_price
            * Decimal("10000")
        )
        if deviation_bps > self.max_price_deviation_bps:
            raise PriceDeviationExceeded(
                f"price deviation {deviation_bps} bps exceeds "
                f"{self.max_price_deviation_bps} bps"
            )

        result = self.service.submit(
            decision,
            candidate,
            account_id=self.account_id,
            market=market,
            order_type=order_type,
            created_at=market.observed_at,
        )
        if not result.duplicate:
            self.bus.publish(CoreSubject.ORDER_INTENT, result.intent)
            self.bus.publish(CoreSubject.ORDER_UPDATED, result)
        return result

