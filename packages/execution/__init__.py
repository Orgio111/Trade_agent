"""Canonical, deterministic paper execution core."""

from packages.execution.ledger import (
    DuplicateFill,
    DuplicateLedgerIdentity,
    ExecutionLedgerError,
    InMemoryExecutionLedger,
    LedgerInvariantViolation,
    UnknownIntent,
)
from packages.execution.models import (
    ExecutionResult,
    Fill,
    LedgerEvent,
    MarketSnapshot,
    OrderIntent,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
)
from packages.execution.reconciliation import (
    DiscrepancyKind,
    Reconciler,
    ReconciliationDiscrepancy,
    ReconciliationReport,
)
from packages.execution.service import (
    BrokerLedgerConflict,
    ExecutionService,
    ExecutionServiceError,
    UnapprovedDecisionError,
)
from packages.execution.state_machine import (
    InvalidOrderTransition,
    allowed_transitions,
    assert_transition,
)

__all__ = [
    "BrokerLedgerConflict",
    "DiscrepancyKind",
    "DuplicateFill",
    "DuplicateLedgerIdentity",
    "ExecutionLedgerError",
    "ExecutionResult",
    "ExecutionService",
    "ExecutionServiceError",
    "Fill",
    "InMemoryExecutionLedger",
    "InvalidOrderTransition",
    "LedgerEvent",
    "LedgerInvariantViolation",
    "MarketSnapshot",
    "OrderIntent",
    "OrderRecord",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Reconciler",
    "ReconciliationDiscrepancy",
    "ReconciliationReport",
    "UnapprovedDecisionError",
    "UnknownIntent",
    "allowed_transitions",
    "assert_transition",
]

