"""Storage ports for the deterministic execution boundary.

The reference in-memory stores are synchronous so replay remains simple and
deterministic.  Durable stores are asynchronous because they perform network
I/O.  Keeping the two protocols explicit prevents an event-loop bridge from
silently blocking a worker.
"""

from __future__ import annotations

from typing import Protocol

from packages.execution.models import (
    ExecutionResult,
    Fill,
    LedgerEvent,
    OrderIntent,
    OrderRecord,
    OrderStatus,
)
from packages.risk import RiskDecision


class DecisionStore(Protocol):
    """Synchronous authorization store used by the replay/paper reference path."""

    def record(self, decision: RiskDecision) -> RiskDecision: ...

    def require_exact_approval(self, decision: RiskDecision) -> RiskDecision: ...

    def get(self, decision_id: str) -> RiskDecision | None: ...


class AsyncDecisionStore(Protocol):
    """Durable authorization store used by async workers."""

    async def record(self, decision: RiskDecision) -> RiskDecision: ...

    async def require_exact_approval(self, decision: RiskDecision) -> RiskDecision: ...

    async def get(self, decision_id: str) -> RiskDecision | None: ...


class ExecutionLedger(Protocol):
    """Synchronous ledger semantics consumed by :class:`ExecutionService`."""

    def persist_intent(self, intent: OrderIntent) -> OrderRecord: ...

    def record_submission(self, broker_record: OrderRecord) -> OrderRecord: ...

    def record_fill(self, fill: Fill) -> OrderRecord: ...

    def record_status(self, intent_id: str, target: OrderStatus) -> OrderRecord: ...

    def get_intent(self, intent_id: str) -> OrderIntent | None: ...

    def get_order(self, intent_id: str) -> OrderRecord | None: ...

    def get_by_client_order_id(self, client_order_id: str) -> OrderRecord | None: ...

    def get_by_broker_order_id(self, broker_order_id: str) -> OrderRecord | None: ...

    def fills_for(self, intent_id: str) -> tuple[Fill, ...]: ...

    def result_for(
        self, intent_id: str, *, duplicate: bool = False
    ) -> ExecutionResult: ...

    def list_orders(self) -> tuple[OrderRecord, ...]: ...

    def list_intents(self) -> tuple[OrderIntent, ...]: ...

    @property
    def fills(self) -> tuple[Fill, ...]: ...

    @property
    def events(self) -> tuple[LedgerEvent, ...]: ...


class AsyncExecutionLedger(Protocol):
    """Network-safe durable ledger semantics for async execution workers."""

    async def persist_intent(self, intent: OrderIntent) -> OrderRecord: ...

    async def record_submission(self, broker_record: OrderRecord) -> OrderRecord: ...

    async def record_fill(self, fill: Fill) -> OrderRecord: ...

    async def record_status(
        self, intent_id: str, target: OrderStatus
    ) -> OrderRecord: ...

    async def get_intent(self, intent_id: str) -> OrderIntent | None: ...

    async def get_order(self, intent_id: str) -> OrderRecord | None: ...

    async def get_by_client_order_id(
        self, client_order_id: str
    ) -> OrderRecord | None: ...

    async def get_by_broker_order_id(
        self, broker_order_id: str
    ) -> OrderRecord | None: ...

    async def fills_for(self, intent_id: str) -> tuple[Fill, ...]: ...

    async def result_for(
        self, intent_id: str, *, duplicate: bool = False
    ) -> ExecutionResult: ...

    async def list_orders(self) -> tuple[OrderRecord, ...]: ...

    async def list_intents(self) -> tuple[OrderIntent, ...]: ...

    async def list_fills(self) -> tuple[Fill, ...]: ...

    async def list_events(self) -> tuple[LedgerEvent, ...]: ...
