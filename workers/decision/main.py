"""Candidate generation followed by the only canonical risk gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from packages.domain.events import MarketEvent
from packages.event_bus import CoreSubject, InMemoryEventBus
from packages.execution import InMemoryDecisionStore
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
        decision_store: InMemoryDecisionStore,
    ) -> None:
        self.bus = bus
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.decision_store = decision_store

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
        age = evaluated_utc - event.exchange_ts
        age_microseconds = (
            age.days * 86_400_000_000
            + age.seconds * 1_000_000
            + age.microseconds
        )
        age_seconds = Decimal(age_microseconds) / Decimal("1000000")
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
        self.decision_store.record(decision)
        subject = (
            CoreSubject.RISK_APPROVED
            if decision.approved
            else CoreSubject.RISK_REJECTED
        )
        self.bus.publish(subject, decision)
        return DecisionOutcome(candidate=candidate, decision=decision)
