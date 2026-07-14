"""Execution-facing broker protocol.

Only the deterministic paper implementation is provided in the MVP core. Live
adapters must satisfy this protocol and pass the same contract/reconciliation
tests before they can be wired to an execution worker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from packages.execution.models import Fill, MarketSnapshot, OrderIntent, OrderRecord


class BrokerError(RuntimeError):
    pass


class InvalidOrderError(BrokerError):
    pass


class DuplicateBrokerIntent(BrokerError):
    pass


class PaperModeRequired(BrokerError):
    pass


@runtime_checkable
class ExecutionBroker(Protocol):
    """Minimal protocol needed by ExecutionService and Reconciler."""

    def submit_order(self, intent: OrderIntent, market: MarketSnapshot) -> OrderRecord:
        ...

    def get_order(self, client_order_id: str) -> OrderRecord | None:
        ...

    def list_orders(self) -> tuple[OrderRecord, ...]:
        ...

    def list_fills(self, client_order_id: str | None = None) -> tuple[Fill, ...]:
        ...
