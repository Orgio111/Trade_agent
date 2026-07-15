"""Replay canonical candles through quality, strategy, risk, and paper execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, localcontext
import hashlib
import json
from typing import Iterable

from packages.brokers import PaperBroker, PaperBrokerConfig
from packages.domain import MarketEvent
from packages.event_bus import CoreSubject, InMemoryEventBus
from packages.execution import (
    ExecutionResult,
    ExecutionService,
    InMemoryDecisionStore,
    InMemoryExecutionLedger,
    MarketSnapshot,
    Reconciler,
)
from packages.risk import (
    CandidateSignal,
    InstrumentConstraints,
    PortfolioState,
    RiskDecision,
    RiskEngine,
    RiskPolicy,
)
from packages.strategies import BaselineSmaStrategy
from workers.decision import DecisionWorker
from workers.execution import ExecutionWorker
from workers.market_data import MarketDataWorker


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    event_id: str
    market_accepted: bool
    candidate: CandidateSignal | None = None
    decision: RiskDecision | None = None
    execution: ExecutionResult | None = None


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    records: tuple[ReplayRecord, ...]
    accepted_market_events: int
    rejected_market_events: int
    candidates: int
    approved_decisions: int
    rejected_decisions: int
    orders: int
    fills: int
    reconciliation_clean: bool
    trace_digest: str


class ReplayHarness:
    """A self-contained deterministic paper pipeline for golden fixtures."""

    def __init__(
        self,
        *,
        constraints: InstrumentConstraints,
        risk_policy: RiskPolicy,
        initial_equity: Decimal = Decimal("10000"),
        account_id: str = "paper-replay",
        fee_rate: Decimal = Decimal("0.001"),
        slippage_bps: Decimal = Decimal("0"),
    ) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        self.constraints = constraints
        self.initial_equity = initial_equity
        self.account_id = account_id
        self.bus = InMemoryEventBus()
        self.ledger = InMemoryExecutionLedger()
        self.decision_store = InMemoryDecisionStore()
        self.broker = PaperBroker(
            PaperBrokerConfig(
                fee_rate=fee_rate,
                slippage_bps=slippage_bps,
            )
        )
        self.market_worker = MarketDataWorker(self.bus)
        self.decision_worker = DecisionWorker(
            self.bus,
            BaselineSmaStrategy(),
            RiskEngine(risk_policy),
            self.decision_store,
        )
        self.execution_worker = ExecutionWorker(
            self.bus,
            ExecutionService(self.ledger, self.broker, self.decision_store),
            account_id=account_id,
        )

    def run(self, events: Iterable[MarketEvent]) -> ReplaySummary:
        records: list[ReplayRecord] = []
        for event in events:
            market_result = self.market_worker.process(event, now=event.received_ts)
            if not market_result.verdict.accepted:
                records.append(
                    ReplayRecord(
                        event_id=str(event.event_id),
                        market_accepted=False,
                    )
                )
                continue

            outcome = self.decision_worker.process(
                event,
                portfolio=self._portfolio_state(event.received_ts),
                constraints=self.constraints,
                evaluated_at=event.received_ts,
                spread_bps=Decimal("2"),
                expected_slippage_bps=Decimal("3"),
            )
            if outcome is None:
                records.append(
                    ReplayRecord(
                        event_id=str(event.event_id),
                        market_accepted=True,
                    )
                )
                continue

            execution = None
            if outcome.decision.approved:
                execution = self.execution_worker.process(
                    outcome.decision,
                    outcome.candidate,
                    market=self._market_snapshot(outcome.candidate, event.received_ts),
                    execution_at=event.received_ts,
                )
            records.append(
                ReplayRecord(
                    event_id=str(event.event_id),
                    market_accepted=True,
                    candidate=outcome.candidate,
                    decision=outcome.decision,
                    execution=execution,
                )
            )

        reconciliation = Reconciler(self.ledger, self.broker).reconcile()
        accepted = sum(record.market_accepted for record in records)
        candidates = [record for record in records if record.candidate is not None]
        approved = [
            record
            for record in candidates
            if record.decision is not None and record.decision.approved
        ]
        rejected = [
            record
            for record in candidates
            if record.decision is not None and not record.decision.approved
        ]
        return ReplaySummary(
            records=tuple(records),
            accepted_market_events=accepted,
            rejected_market_events=len(records) - accepted,
            candidates=len(candidates),
            approved_decisions=len(approved),
            rejected_decisions=len(rejected),
            orders=len(self.ledger.list_orders()),
            fills=len(self.ledger.fills),
            reconciliation_clean=reconciliation.clean,
            trace_digest=self._trace_digest(records),
        )

    def _portfolio_state(self, at: datetime) -> PortfolioState:
        with localcontext() as context:
            context.prec = 50
            return self._portfolio_state_with_context(at)

    def _portfolio_state_with_context(self, at: datetime) -> PortfolioState:
        orders = self.ledger.list_orders()
        gross_exposure = sum(
            (
                order.filled_quantity * order.average_fill_price
                for order in orders
                if order.average_fill_price is not None
            ),
            Decimal("0"),
        )
        spent = sum(
            (
                fill.notional + fill.fee
                if fill.side.value == "buy"
                else -(fill.notional - fill.fee)
                for fill in self.ledger.fills
            ),
            Decimal("0"),
        )
        return PortfolioState(
            account_id=self.account_id,
            state_id=f"replay-state-{len(orders)}-{len(self.ledger.fills)}",
            equity=self.initial_equity,
            cash=max(Decimal("0"), self.initial_equity - spent),
            daily_pnl=Decimal("0"),
            weekly_pnl=Decimal("0"),
            consecutive_losses=0,
            open_positions=sum(order.filled_quantity > 0 for order in orders),
            gross_exposure=gross_exposure,
            reconciled_at=at,
            kill_switch_active=False,
        )

    @staticmethod
    def _market_snapshot(
        candidate: CandidateSignal,
        observed_at: datetime,
    ) -> MarketSnapshot:
        with localcontext() as context:
            context.prec = 50
            half_spread = candidate.spread_bps / Decimal("20000")
            return MarketSnapshot(
                venue=candidate.venue,
                market_type=candidate.market_type,
                instrument=candidate.instrument,
                bid=candidate.reference_price * (Decimal("1") - half_spread),
                ask=candidate.reference_price * (Decimal("1") + half_spread),
                last=candidate.reference_price,
                observed_at=observed_at,
            )

    def _trace_digest(self, records: list[ReplayRecord]) -> str:
        payload = {
            "market": [
                {
                    "event_id": record.event_id,
                    "accepted": record.market_accepted,
                    "signal_id": (
                        record.candidate.signal_id
                        if record.candidate is not None
                        else None
                    ),
                    "decision_id": (
                        record.decision.decision_id
                        if record.decision is not None
                        else None
                    ),
                    "approved": (
                        record.decision.approved
                        if record.decision is not None
                        else None
                    ),
                }
                for record in records
            ],
            "bus": [
                {
                    "sequence": event.sequence,
                    "subject": event.subject.value,
                }
                for event in self.bus.events
            ],
            "ledger_events": [
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "intent_id": event.intent_id,
                    "occurred_at": event.occurred_at.isoformat(),
                    "details": dict(event.details),
                }
                for event in self.ledger.events
            ],
            "fills": [
                {
                    "fill_id": fill.fill_id,
                    "quantity": str(fill.quantity),
                    "price": str(fill.price),
                    "fee": str(fill.fee),
                    "filled_at": fill.filled_at.isoformat(),
                }
                for fill in self.ledger.fills
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
