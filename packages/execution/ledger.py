"""In-memory reference ledger with durable-style uniqueness semantics.

This implementation is intentionally small enough for unit/replay tests while
enforcing the same invariants expected from the PostgreSQL ledger: intent,
client-order, broker-order, and fill IDs are never reused; fills and events are
append-only; derived order state can only move through the state machine.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, localcontext
from threading import RLock
from types import MappingProxyType

from packages.execution.models import (
    ExecutionResult,
    Fill,
    LedgerEvent,
    OrderIntent,
    OrderRecord,
    OrderStatus,
    canonical_decimal,
    utc_now,
)
from packages.execution.state_machine import assert_transition


class ExecutionLedgerError(RuntimeError):
    pass


class DuplicateLedgerIdentity(ExecutionLedgerError):
    pass


class DuplicateFill(ExecutionLedgerError):
    pass


class UnknownIntent(ExecutionLedgerError):
    pass


class LedgerInvariantViolation(ExecutionLedgerError):
    pass


class InMemoryExecutionLedger:
    """Thread-safe execution ledger used by paper/replay mode.

    There is deliberately no delete/reset API. A caller that needs another
    scenario creates another ledger, preserving permanent uniqueness inside a
    ledger's lifetime.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._intents: dict[str, OrderIntent] = {}
        self._orders: dict[str, OrderRecord] = {}
        self._intent_by_client_id: dict[str, str] = {}
        self._intent_by_broker_id: dict[str, str] = {}
        self._seen_intent_ids: set[str] = set()
        self._seen_client_order_ids: set[str] = set()
        self._seen_broker_order_ids: set[str] = set()
        self._seen_fill_ids: set[str] = set()
        self._fills: list[Fill] = []
        self._events: list[LedgerEvent] = []

    def persist_intent(self, intent: OrderIntent) -> OrderRecord:
        """Atomically reserve both deterministic IDs before broker submission."""

        with self._lock:
            if intent.intent_id in self._seen_intent_ids:
                raise DuplicateLedgerIdentity(f"intent_id already used: {intent.intent_id}")
            if intent.client_order_id in self._seen_client_order_ids:
                raise DuplicateLedgerIdentity(
                    f"client_order_id already used: {intent.client_order_id}"
                )
            record = OrderRecord.pending(intent)
            self._seen_intent_ids.add(intent.intent_id)
            self._seen_client_order_ids.add(intent.client_order_id)
            self._intents[intent.intent_id] = intent
            self._orders[intent.intent_id] = record
            self._intent_by_client_id[intent.client_order_id] = intent.intent_id
            self._append_event(
                "intent_persisted",
                intent.intent_id,
                {
                    "client_order_id": intent.client_order_id,
                    "status": record.status.value,
                    "quantity": canonical_decimal(intent.quantity),
                },
                occurred_at=intent.created_at,
            )
            return record

    def record_submission(self, broker_record: OrderRecord) -> OrderRecord:
        """Attach a broker identity after an intent has been persisted.

        Fills are applied separately. Even when the broker returns an immediate
        FILLED snapshot, the ledger records OPEN first and derives FILLED from
        the actual append-only fill rows.
        """

        with self._lock:
            current = self._require_order(broker_record.intent_id)
            self._assert_same_order(current, broker_record)
            if current.broker_order_id is not None:
                if current.broker_order_id == broker_record.broker_order_id:
                    return current
                raise LedgerInvariantViolation("broker_order_id cannot change")

            broker_id = broker_record.broker_order_id
            if not broker_id:
                raise LedgerInvariantViolation("submitted order needs a broker_order_id")
            if broker_id in self._seen_broker_order_ids:
                raise DuplicateLedgerIdentity(f"broker_order_id already used: {broker_id}")

            target = (
                OrderStatus.REJECTED
                if broker_record.status is OrderStatus.REJECTED
                else OrderStatus.OPEN
            )
            assert_transition(current.status, target)
            now = broker_record.updated_at
            updated = replace(
                current,
                broker_order_id=broker_id,
                status=target,
                updated_at=now,
            )
            self._seen_broker_order_ids.add(broker_id)
            self._intent_by_broker_id[broker_id] = current.intent_id
            self._orders[current.intent_id] = updated
            self._append_event(
                "order_submitted",
                current.intent_id,
                {"broker_order_id": broker_id, "status": target.value},
                occurred_at=now,
            )
            return updated

    def record_fill(self, fill: Fill) -> OrderRecord:
        with localcontext() as context:
            context.prec = 50
            return self._record_fill_with_context(fill)

    def _record_fill_with_context(self, fill: Fill) -> OrderRecord:
        with self._lock:
            if fill.fill_id in self._seen_fill_ids:
                raise DuplicateFill(f"fill_id already used: {fill.fill_id}")
            current = self._require_order(fill.intent_id)
            if current.broker_order_id is None:
                raise LedgerInvariantViolation("cannot record a fill before broker submission")
            if fill.client_order_id != current.client_order_id:
                raise LedgerInvariantViolation("fill client_order_id does not match order")
            if fill.broker_order_id != current.broker_order_id:
                raise LedgerInvariantViolation("fill broker_order_id does not match order")
            if fill.instrument != current.instrument or fill.side is not current.side:
                raise LedgerInvariantViolation("fill instrument/side does not match order")

            new_quantity = current.filled_quantity + fill.quantity
            if new_quantity > current.requested_quantity:
                raise LedgerInvariantViolation("fill exceeds remaining order quantity")
            old_notional = (
                current.average_fill_price * current.filled_quantity
                if current.average_fill_price is not None
                else Decimal("0")
            )
            average = (old_notional + fill.notional) / new_quantity
            target = (
                OrderStatus.FILLED
                if new_quantity == current.requested_quantity
                else OrderStatus.PARTIALLY_FILLED
            )
            assert_transition(current.status, target)
            updated_at = max(current.updated_at, fill.filled_at)
            updated = replace(
                current,
                status=target,
                filled_quantity=new_quantity,
                average_fill_price=average,
                cumulative_fee=current.cumulative_fee + fill.fee,
                updated_at=updated_at,
            )
            self._seen_fill_ids.add(fill.fill_id)
            self._fills.append(fill)
            self._orders[current.intent_id] = updated
            self._append_event(
                "fill_recorded",
                current.intent_id,
                {
                    "fill_id": fill.fill_id,
                    "quantity": canonical_decimal(fill.quantity),
                    "price": canonical_decimal(fill.price),
                    "fee": canonical_decimal(fill.fee),
                    "status": target.value,
                },
                occurred_at=fill.filled_at,
            )
            return updated

    def record_status(self, intent_id: str, target: OrderStatus) -> OrderRecord:
        with self._lock:
            current = self._require_order(intent_id)
            target = OrderStatus(target)
            if target is current.status:
                return current
            if target is OrderStatus.FILLED and current.filled_quantity != current.requested_quantity:
                raise LedgerInvariantViolation("FILLED status requires full fill quantity")
            if target is OrderStatus.PARTIALLY_FILLED and not (
                Decimal("0") < current.filled_quantity < current.requested_quantity
            ):
                raise LedgerInvariantViolation("PARTIALLY_FILLED requires a partial fill")
            if target is OrderStatus.REJECTED and current.filled_quantity != 0:
                raise LedgerInvariantViolation("an order with fills cannot be rejected")
            assert_transition(current.status, target)
            now = utc_now()
            updated = replace(current, status=target, updated_at=now)
            self._orders[intent_id] = updated
            self._append_event(
                "status_changed",
                intent_id,
                {"from": current.status.value, "to": target.value},
                occurred_at=now,
            )
            return updated

    def get_intent(self, intent_id: str) -> OrderIntent | None:
        with self._lock:
            return self._intents.get(intent_id)

    def get_order(self, intent_id: str) -> OrderRecord | None:
        with self._lock:
            return self._orders.get(intent_id)

    def get_by_client_order_id(self, client_order_id: str) -> OrderRecord | None:
        with self._lock:
            intent_id = self._intent_by_client_id.get(client_order_id)
            return self._orders.get(intent_id) if intent_id else None

    def get_by_broker_order_id(self, broker_order_id: str) -> OrderRecord | None:
        with self._lock:
            intent_id = self._intent_by_broker_id.get(broker_order_id)
            return self._orders.get(intent_id) if intent_id else None

    def fills_for(self, intent_id: str) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(fill for fill in self._fills if fill.intent_id == intent_id)

    def result_for(self, intent_id: str, *, duplicate: bool = False) -> ExecutionResult:
        with self._lock:
            intent = self._intents.get(intent_id)
            order = self._orders.get(intent_id)
            if intent is None or order is None:
                raise UnknownIntent(intent_id)
            fills = tuple(fill for fill in self._fills if fill.intent_id == intent_id)
            return ExecutionResult(intent=intent, order=order, fills=fills, duplicate=duplicate)

    def list_orders(self) -> tuple[OrderRecord, ...]:
        with self._lock:
            return tuple(self._orders.values())

    def list_intents(self) -> tuple[OrderIntent, ...]:
        with self._lock:
            return tuple(self._intents.values())

    @property
    def fills(self) -> tuple[Fill, ...]:
        with self._lock:
            return tuple(self._fills)

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _require_order(self, intent_id: str) -> OrderRecord:
        try:
            return self._orders[intent_id]
        except KeyError as exc:
            raise UnknownIntent(intent_id) from exc

    @staticmethod
    def _assert_same_order(current: OrderRecord, incoming: OrderRecord) -> None:
        fields = (
            "intent_id",
            "client_order_id",
            "account_id",
            "venue",
            "market_type",
            "instrument",
            "side",
            "order_type",
            "requested_quantity",
            "limit_price",
        )
        changed = [name for name in fields if getattr(current, name) != getattr(incoming, name)]
        if changed:
            raise LedgerInvariantViolation(
                "broker record changed immutable fields: " + ", ".join(changed)
            )

    def _append_event(
        self,
        event_type: str,
        intent_id: str,
        details: dict[str, str | None],
        *,
        occurred_at=None,
    ) -> None:
        self._events.append(
            LedgerEvent(
                sequence=len(self._events) + 1,
                event_type=event_type,
                intent_id=intent_id,
                occurred_at=occurred_at or utc_now(),
                details=MappingProxyType(dict(details)),
            )
        )
