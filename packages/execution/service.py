"""Idempotent bridge from risk approval to broker submission."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from packages.brokers.base import (
    BrokerError,
    DefinitiveBrokerError,
    DuplicateBrokerIntent,
    ExecutionBroker,
)
from packages.execution.ledger import (
    DuplicateFill,
    DuplicateLedgerIdentity,
    InMemoryExecutionLedger,
    LedgerInvariantViolation,
)
from packages.execution.decision_store import (
    DecisionAuthorizationError,
    InMemoryDecisionStore,
)
from packages.execution.models import (
    ExecutionResult,
    MarketSnapshot,
    OrderIntent,
    OrderStatus,
    OrderType,
)
from packages.domain import SourceMode


class ExecutionServiceError(RuntimeError):
    pass


class UnapprovedDecisionError(ExecutionServiceError):
    pass


class BrokerLedgerConflict(ExecutionServiceError):
    pass


class AmbiguousSubmissionError(ExecutionServiceError):
    """Submission outcome is unknown and automatic resubmission is forbidden."""


class ExecutionService:
    """Persist-before-submit orchestration with deterministic idempotency."""

    def __init__(
        self,
        ledger: InMemoryExecutionLedger,
        broker: ExecutionBroker,
        decision_store: InMemoryDecisionStore,
    ) -> None:
        self.ledger = ledger
        self.broker = broker
        self.decision_store = decision_store

    def submit(
        self,
        decision: object,
        candidate: object,
        *,
        account_id: str,
        market: MarketSnapshot,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: Decimal | int | str | None = None,
        created_at: datetime | None = None,
    ) -> ExecutionResult:
        """Submit an approved decision exactly once.

        A duplicate invocation returns the existing ledger result and never
        sends another broker request. The candidate is supplied separately so
        the service can verify that the decision approved that exact signal.
        """

        try:
            self.decision_store.require_exact_approval(decision)
        except DecisionAuthorizationError as exc:
            raise UnapprovedDecisionError(str(exc)) from exc
        if getattr(decision, "approved", False) is not True:
            raise UnapprovedDecisionError("execution requires approved=True")
        if getattr(candidate, "source_mode", None) not in {
            SourceMode.REPLAY,
            SourceMode.PAPER_LIVE,
        }:
            raise UnapprovedDecisionError(
                "core execution is hard-gated to replay and paper_live modes"
            )

        try:
            intent = OrderIntent.from_approved_decision(
                decision,
                candidate,
                account_id=account_id,
                order_type=order_type,
                limit_price=limit_price,
                created_at=created_at,
            )
        except ValueError as exc:
            if "not approved" in str(exc):
                raise UnapprovedDecisionError(str(exc)) from exc
            raise

        existing = self.ledger.get_order(intent.intent_id)
        if existing is not None:
            if existing.status in {
                OrderStatus.PENDING_SUBMIT,
                OrderStatus.AMBIGUOUS,
            }:
                broker_record = self.broker.get_order(intent.client_order_id)
                if broker_record is None:
                    if existing.status is OrderStatus.PENDING_SUBMIT:
                        self.ledger.record_status(
                            intent.intent_id, OrderStatus.AMBIGUOUS
                        )
                    raise AmbiguousSubmissionError(
                        "persisted intent has no definitive broker outcome; "
                        "reconciliation is required"
                    )
                self._synchronize_record(broker_record)
                return self.ledger.result_for(intent.intent_id, duplicate=True)
            return self.ledger.result_for(intent.intent_id, duplicate=True)

        try:
            self.ledger.persist_intent(intent)
        except DuplicateLedgerIdentity:
            # Another worker may have won the race after the initial lookup.
            existing = self.ledger.get_order(intent.intent_id)
            if existing is None:
                raise
            return self.ledger.result_for(intent.intent_id, duplicate=True)

        try:
            broker_record = self.broker.submit_order(intent, market)
        except DuplicateBrokerIntent as exc:
            # A process restart can retain broker state while rebuilding this
            # in-memory reference ledger. Recover only the broker's exact
            # deterministic client ID; never submit a replacement order.
            broker_record = self.broker.get_order(intent.client_order_id)
            if broker_record is None or broker_record.intent_id != intent.intent_id:
                self.ledger.record_status(intent.intent_id, OrderStatus.AMBIGUOUS)
                raise BrokerLedgerConflict("broker duplicate does not match intent") from exc
        except DefinitiveBrokerError:
            self.ledger.record_status(intent.intent_id, OrderStatus.REJECTED)
            raise
        except BrokerError as exc:
            self.ledger.record_status(intent.intent_id, OrderStatus.AMBIGUOUS)
            raise AmbiguousSubmissionError(
                "broker submission outcome is ambiguous; automatic resubmission blocked"
            ) from exc

        self._synchronize_record(broker_record)
        return self.ledger.result_for(intent.intent_id)

    def synchronize(self, client_order_id: str) -> ExecutionResult:
        """Copy broker-observed records/fills into the ledger without synthesis."""

        broker_record = self.broker.get_order(client_order_id)
        if broker_record is None:
            raise BrokerLedgerConflict(f"broker order not found: {client_order_id}")
        self._synchronize_record(broker_record)
        return self.ledger.result_for(broker_record.intent_id)

    def _synchronize_record(self, broker_record) -> None:
        ledger_record = self.ledger.get_order(broker_record.intent_id)
        if ledger_record is None:
            raise BrokerLedgerConflict("broker order has no persisted ledger intent")
        if ledger_record.broker_order_id is None:
            self.ledger.record_submission(broker_record)
        elif ledger_record.broker_order_id != broker_record.broker_order_id:
            raise BrokerLedgerConflict("broker_order_id differs from ledger")

        for fill in self.broker.list_fills(broker_record.client_order_id):
            try:
                self.ledger.record_fill(fill)
            except DuplicateFill:
                existing = next(
                    (
                        item
                        for item in self.ledger.fills_for(fill.intent_id)
                        if item.fill_id == fill.fill_id
                    ),
                    None,
                )
                if existing != fill:
                    raise BrokerLedgerConflict(
                        "duplicate fill_id has different immutable content"
                    )

        ledger_record = self.ledger.get_order(broker_record.intent_id)
        if ledger_record is None:
            raise BrokerLedgerConflict("ledger order disappeared during synchronization")

        # Fill statuses are always derived from actual fills. Terminal cancel
        # and reject states can be copied directly; FILLED without matching
        # fill rows remains visible as a reconciliation discrepancy.
        if broker_record.status in {OrderStatus.CANCELED, OrderStatus.REJECTED}:
            if ledger_record.status is not broker_record.status:
                self.ledger.record_status(broker_record.intent_id, broker_record.status)
        elif broker_record.status is OrderStatus.FILLED:
            if ledger_record.filled_quantity == ledger_record.requested_quantity:
                if ledger_record.status is not OrderStatus.FILLED:
                    self.ledger.record_status(broker_record.intent_id, OrderStatus.FILLED)
        elif broker_record.status is OrderStatus.PARTIALLY_FILLED:
            if (
                Decimal("0") < ledger_record.filled_quantity < ledger_record.requested_quantity
                and ledger_record.status is not OrderStatus.PARTIALLY_FILLED
            ):
                self.ledger.record_status(
                    broker_record.intent_id, OrderStatus.PARTIALLY_FILLED
                )
