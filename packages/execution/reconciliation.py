"""Read-only broker/ledger reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from packages.brokers.base import ExecutionBroker
from packages.execution.ledger import InMemoryExecutionLedger


class DiscrepancyKind(str, Enum):
    MISSING_AT_BROKER = "missing_at_broker"
    MISSING_IN_LEDGER = "missing_in_ledger"
    REQUESTED_QUANTITY_MISMATCH = "requested_quantity_mismatch"
    FILLED_QUANTITY_MISMATCH = "filled_quantity_mismatch"
    STATUS_MISMATCH = "status_mismatch"


@dataclass(frozen=True, slots=True)
class ReconciliationDiscrepancy:
    kind: DiscrepancyKind
    client_order_id: str
    ledger_value: str | None
    broker_value: str | None


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    discrepancies: tuple[ReconciliationDiscrepancy, ...]
    ledger_order_count: int
    broker_order_count: int

    @property
    def clean(self) -> bool:
        return not self.discrepancies


class Reconciler:
    """Compare snapshots without changing ledger state or inventing fills."""

    def __init__(self, ledger: InMemoryExecutionLedger, broker: ExecutionBroker) -> None:
        self.ledger = ledger
        self.broker = broker

    def reconcile(self) -> ReconciliationReport:
        ledger_orders = {
            order.client_order_id: order for order in self.ledger.list_orders()
        }
        broker_orders = {order.client_order_id: order for order in self.broker.list_orders()}
        discrepancies: list[ReconciliationDiscrepancy] = []

        for client_order_id in sorted(ledger_orders.keys() - broker_orders.keys()):
            order = ledger_orders[client_order_id]
            discrepancies.append(
                ReconciliationDiscrepancy(
                    kind=DiscrepancyKind.MISSING_AT_BROKER,
                    client_order_id=client_order_id,
                    ledger_value=order.status.value,
                    broker_value=None,
                )
            )

        for client_order_id in sorted(broker_orders.keys() - ledger_orders.keys()):
            order = broker_orders[client_order_id]
            discrepancies.append(
                ReconciliationDiscrepancy(
                    kind=DiscrepancyKind.MISSING_IN_LEDGER,
                    client_order_id=client_order_id,
                    ledger_value=None,
                    broker_value=order.status.value,
                )
            )

        for client_order_id in sorted(ledger_orders.keys() & broker_orders.keys()):
            ledger_order = ledger_orders[client_order_id]
            broker_order = broker_orders[client_order_id]
            comparisons = (
                (
                    DiscrepancyKind.REQUESTED_QUANTITY_MISMATCH,
                    ledger_order.requested_quantity,
                    broker_order.requested_quantity,
                ),
                (
                    DiscrepancyKind.FILLED_QUANTITY_MISMATCH,
                    ledger_order.filled_quantity,
                    broker_order.filled_quantity,
                ),
                (
                    DiscrepancyKind.STATUS_MISMATCH,
                    ledger_order.status.value,
                    broker_order.status.value,
                ),
            )
            for kind, ledger_value, broker_value in comparisons:
                if ledger_value != broker_value:
                    discrepancies.append(
                        ReconciliationDiscrepancy(
                            kind=kind,
                            client_order_id=client_order_id,
                            ledger_value=str(ledger_value),
                            broker_value=str(broker_value),
                        )
                    )

        return ReconciliationReport(
            discrepancies=tuple(discrepancies),
            ledger_order_count=len(ledger_orders),
            broker_order_count=len(broker_orders),
        )

