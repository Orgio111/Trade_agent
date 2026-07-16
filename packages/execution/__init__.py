"""Canonical, deterministic paper execution core."""

from packages.execution.decision_store import (
    DecisionAuthorizationError,
    InMemoryDecisionStore,
)
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
from packages.execution.ports import (
    AsyncDecisionStore,
    AsyncExecutionLedger,
    DecisionStore,
    ExecutionLedger,
)
from packages.execution.postgres_decision_store import (
    DecisionStoreUnavailable,
    PostgresDecisionStore,
)
from packages.execution.postgres_ledger import (
    LedgerUnavailable,
    PostgresExecutionLedger,
)
from packages.execution.reconciliation import (
    DiscrepancyKind,
    Reconciler,
    ReconciliationDiscrepancy,
    ReconciliationReport,
)
from packages.execution.service import (
    AmbiguousSubmissionError,
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
    "AmbiguousSubmissionError",
    "AsyncDecisionStore",
    "AsyncExecutionLedger",
    "BrokerLedgerConflict",
    "DiscrepancyKind",
    "DecisionAuthorizationError",
    "DecisionStore",
    "DecisionStoreUnavailable",
    "DuplicateFill",
    "DuplicateLedgerIdentity",
    "ExecutionLedgerError",
    "ExecutionLedger",
    "ExecutionResult",
    "ExecutionService",
    "ExecutionServiceError",
    "Fill",
    "InMemoryExecutionLedger",
    "InMemoryDecisionStore",
    "InvalidOrderTransition",
    "LedgerEvent",
    "LedgerInvariantViolation",
    "LedgerUnavailable",
    "MarketSnapshot",
    "OrderIntent",
    "OrderRecord",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PostgresDecisionStore",
    "PostgresExecutionLedger",
    "Reconciler",
    "ReconciliationDiscrepancy",
    "ReconciliationReport",
    "UnapprovedDecisionError",
    "UnknownIntent",
    "allowed_transitions",
    "assert_transition",
]
