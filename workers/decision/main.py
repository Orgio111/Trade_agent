"""Candidate generation followed by the only canonical risk gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from packages.domain.events import MarketEvent
from packages.event_bus import CoreSubject, InMemoryEventBus
from packages.risk import (
    CandidateSignal,
    InstrumentConstraints,
    PortfolioState,
    RiskDecision,
    RiskEngine,
)
from packages.strategies import BaselineSmaStrategy


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    candidate: CandidateSignal
    decision: RiskDecision


class DecisionWorker:
    def __init__(
        self,
        bus: InMemoryEventBus,
        strategy: BaselineSmaStrategy,
        risk_engine: RiskEngine,
    ) -> None:
        self.bus = bus
        self.strategy = strategy
        self.risk_engine = risk_engine

    def process(
        self,
        event: MarketEvent,
        *,
        portfolio: PortfolioState,
        constraints: InstrumentConstraints,
        evaluated_at: datetime,
        spread_bps: Decimal | None,
        expected_slippage_bps: Decimal | None,
    ) -> DecisionOutcome | None:
        candidate = self.strategy.on_market_event(event, constraints)
        if candidate is None:
            return None
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        evaluated_utc = evaluated_at.astimezone(UTC)
        age_seconds = Decimal(str((evaluated_utc - event.exchange_ts).total_seconds()))
        candidate_values = candidate.model_dump(mode="python")
        candidate_values.update(
            data_age_seconds=age_seconds if age_seconds >= 0 else None,
            spread_bps=spread_bps,
            expected_slippage_bps=expected_slippage_bps,
        )
        candidate = CandidateSignal.model_validate(candidate_values)
        self.bus.publish(CoreSubject.SIGNAL_CANDIDATE, candidate)
        decision = self.risk_engine.evaluate(
            candidate,
            portfolio,
            constraints,
            evaluated_at=evaluated_utc,
        )
        subject = (
            CoreSubject.RISK_APPROVED
            if decision.approved
            else CoreSubject.RISK_REJECTED
        )
        self.bus.publish(subject, decision)
        return DecisionOutcome(candidate=candidate, decision=decision)
