"""Async PostgreSQL execution ledger with append-only identity semantics."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal, localcontext
import hashlib
from types import MappingProxyType
from typing import Any, Mapping

from asyncpg import UniqueViolationError  # type: ignore[import-untyped]

from packages.domain import SourceMode, canonical_json, compute_payload_checksum
from packages.execution._postgres import PostgresConnection, PostgresPool, json_object
from packages.execution.ledger import (
    DuplicateFill,
    DuplicateLedgerIdentity,
    ExecutionLedgerError,
    LedgerInvariantViolation,
    UnknownIntent,
)
from packages.execution.models import (
    ExecutionResult,
    Fill,
    LedgerEvent,
    OrderIntent,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    canonical_decimal,
    utc_now,
)
from packages.execution.state_machine import assert_transition


class LedgerUnavailable(ExecutionLedgerError):
    """The authoritative ledger cannot be verified, so execution must stop."""


_ORDER_SELECT = """
    SELECT
        oi.payload AS intent_payload,
        bo.venue_order_id AS broker_order_id,
        bo.status,
        bo.requested_quantity,
        bo.filled_quantity,
        bo.average_price,
        bo.total_fee,
        bo.created_at AS order_created_at,
        bo.updated_at AS order_updated_at
    FROM broker_orders AS bo
    JOIN order_intents AS oi ON oi.id = bo.order_intent_id
"""

_FILL_SELECT = """
    SELECT
        f.id AS fill_id,
        f.quantity,
        f.price,
        f.fee,
        COALESCE(f.exchange_ts, f.received_ts) AS filled_at,
        bo.venue_order_id AS broker_order_id,
        oi.payload AS intent_payload
    FROM fills AS f
    JOIN broker_orders AS bo ON bo.id = f.order_id
    JOIN order_intents AS oi ON oi.id = bo.order_intent_id
"""


class PostgresExecutionLedger:
    """Durable implementation of the execution-ledger semantic port.

    Each mutation locks one order row and writes its append-only event in the
    same PostgreSQL transaction.  No method submits to a broker or infers an
    approval from an event.
    """

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    async def persist_intent(self, intent: OrderIntent) -> OrderRecord:
        if intent.source_mode not in {SourceMode.REPLAY, SourceMode.PAPER_LIVE}:
            raise LedgerInvariantViolation(
                "durable execution ledger accepts replay and paper_live intents only"
            )
        payload = self._intent_json(intent)
        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    conflict = await connection.fetchrow(
                        """
                        SELECT id
                        FROM order_intents
                        WHERE id = $1
                           OR client_order_id = $2
                           OR risk_decision_id = $3
                        LIMIT 1
                        FOR UPDATE
                        """,
                        intent.intent_id,
                        intent.client_order_id,
                        intent.decision_id,
                    )
                    if conflict is not None:
                        raise DuplicateLedgerIdentity(
                            "intent, client-order, or risk-decision identity already used"
                        )
                    authority = await connection.fetchrow(
                        """
                        SELECT trace_id
                        FROM risk_decisions
                        WHERE id = $1 AND approved = TRUE
                        FOR SHARE
                        """,
                        intent.decision_id,
                    )
                    if authority is None:
                        raise LedgerInvariantViolation(
                            "intent has no persisted approved risk decision"
                        )
                    await connection.execute(
                        """
                        INSERT INTO order_intents (
                            id, client_order_id, risk_decision_id, trace_id,
                            account_id, venue, market_type, instrument_id, side,
                            order_type, quantity, limit_price, source_mode,
                            status, payload, created_at, updated_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, 'persisted', $14::jsonb, $15, $15
                        )
                        """,
                        intent.intent_id,
                        intent.client_order_id,
                        intent.decision_id,
                        str(authority["trace_id"]),
                        intent.account_id,
                        intent.venue,
                        intent.market_type,
                        intent.instrument,
                        intent.side.value,
                        intent.order_type.value,
                        intent.quantity,
                        intent.limit_price,
                        intent.source_mode.value,
                        payload,
                        intent.created_at,
                    )
                    await connection.execute(
                        """
                        INSERT INTO broker_orders (
                            id, order_intent_id, client_order_id, status,
                            requested_quantity, created_at, updated_at
                        )
                        VALUES ($1, $1, $2, 'pending_submit', $3, $4, $4)
                        """,
                        intent.intent_id,
                        intent.client_order_id,
                        intent.quantity,
                        intent.created_at,
                    )
                    await self._append_event(
                        connection,
                        event_type="intent_persisted",
                        intent_id=intent.intent_id,
                        previous_status=None,
                        new_status=OrderStatus.PENDING_SUBMIT,
                        details={
                            "client_order_id": intent.client_order_id,
                            "quantity": canonical_decimal(intent.quantity),
                            "status": OrderStatus.PENDING_SUBMIT.value,
                        },
                        occurred_at=intent.created_at,
                    )
            return OrderRecord.pending(intent)
        except (DuplicateLedgerIdentity, LedgerInvariantViolation, UnknownIntent):
            raise
        except UniqueViolationError as exc:
            raise DuplicateLedgerIdentity(
                "intent, client-order, or risk-decision identity already used"
            ) from exc
        except Exception as exc:
            raise LedgerUnavailable("intent persistence failed closed") from exc

    async def record_submission(self, broker_record: OrderRecord) -> OrderRecord:
        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    current = await self._require_order(
                        connection, broker_record.intent_id, lock=True
                    )
                    self._assert_same_order(current, broker_record)
                    if current.broker_order_id is not None:
                        if current.broker_order_id == broker_record.broker_order_id:
                            return current
                        raise LedgerInvariantViolation("broker_order_id cannot change")
                    broker_id = broker_record.broker_order_id
                    if not broker_id:
                        raise LedgerInvariantViolation(
                            "submitted order needs a broker_order_id"
                        )
                    duplicate = await connection.fetchrow(
                        "SELECT id FROM broker_orders WHERE venue_order_id = $1",
                        broker_id,
                    )
                    if duplicate is not None:
                        raise DuplicateLedgerIdentity(
                            f"broker_order_id already used: {broker_id}"
                        )
                    target = (
                        OrderStatus.REJECTED
                        if broker_record.status is OrderStatus.REJECTED
                        else OrderStatus.OPEN
                    )
                    assert_transition(current.status, target)
                    await connection.execute(
                        """
                        UPDATE broker_orders
                        SET venue_order_id = $2, status = $3, updated_at = $4
                        WHERE id = $1 AND venue_order_id IS NULL
                        """,
                        current.intent_id,
                        broker_id,
                        target.value,
                        broker_record.updated_at,
                    )
                    await connection.execute(
                        """
                        UPDATE order_intents
                        SET status = $2, updated_at = $3
                        WHERE id = $1
                        """,
                        current.intent_id,
                        "terminal" if target.terminal else "submitted",
                        broker_record.updated_at,
                    )
                    await self._append_event(
                        connection,
                        event_type="order_submitted",
                        intent_id=current.intent_id,
                        previous_status=current.status,
                        new_status=target,
                        details={
                            "broker_order_id": broker_id,
                            "status": target.value,
                        },
                        occurred_at=broker_record.updated_at,
                    )
                    return replace(
                        current,
                        broker_order_id=broker_id,
                        status=target,
                        updated_at=broker_record.updated_at,
                    )
        except (DuplicateLedgerIdentity, LedgerInvariantViolation, UnknownIntent):
            raise
        except UniqueViolationError as exc:
            raise DuplicateLedgerIdentity("broker_order_id already used") from exc
        except Exception as exc:
            raise LedgerUnavailable(
                "broker submission persistence failed closed"
            ) from exc

    async def record_fill(self, fill: Fill) -> OrderRecord:
        try:
            with localcontext() as context:
                context.prec = 50
                return await self._record_fill(fill)
        except (DuplicateFill, LedgerInvariantViolation, UnknownIntent):
            raise
        except UniqueViolationError as exc:
            raise DuplicateFill(f"fill_id already used: {fill.fill_id}") from exc
        except Exception as exc:
            raise LedgerUnavailable("fill persistence failed closed") from exc

    async def _record_fill(self, fill: Fill) -> OrderRecord:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                duplicate = await connection.fetchrow(
                    "SELECT id FROM fills WHERE id = $1", fill.fill_id
                )
                if duplicate is not None:
                    raise DuplicateFill(f"fill_id already used: {fill.fill_id}")
                current = await self._require_order(
                    connection, fill.intent_id, lock=True
                )
                if current.broker_order_id is None:
                    raise LedgerInvariantViolation(
                        "cannot record a fill before broker submission"
                    )
                if fill.client_order_id != current.client_order_id:
                    raise LedgerInvariantViolation(
                        "fill client_order_id does not match order"
                    )
                if fill.broker_order_id != current.broker_order_id:
                    raise LedgerInvariantViolation(
                        "fill broker_order_id does not match order"
                    )
                if (
                    fill.instrument != current.instrument
                    or fill.side is not current.side
                ):
                    raise LedgerInvariantViolation(
                        "fill instrument/side does not match order"
                    )

                new_quantity = current.filled_quantity + fill.quantity
                if new_quantity > current.requested_quantity:
                    raise LedgerInvariantViolation(
                        "fill exceeds remaining order quantity"
                    )
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
                total_fee = current.cumulative_fee + fill.fee
                await connection.execute(
                    """
                    INSERT INTO fills (
                        id, order_id, venue_fill_id, quantity, price, fee,
                        liquidity_role, exchange_ts, received_ts, payload
                    )
                    VALUES (
                        $1, $2, $1, $3, $4, $5, 'paper', $6, $6, $7::jsonb
                    )
                    """,
                    fill.fill_id,
                    fill.intent_id,
                    fill.quantity,
                    fill.price,
                    fill.fee,
                    fill.filled_at,
                    self._fill_json(fill),
                )
                await connection.execute(
                    """
                    UPDATE broker_orders
                    SET status = $2, filled_quantity = $3, average_price = $4,
                        total_fee = $5, updated_at = $6
                    WHERE id = $1
                    """,
                    current.intent_id,
                    target.value,
                    new_quantity,
                    average,
                    total_fee,
                    updated_at,
                )
                if target.terminal:
                    await connection.execute(
                        """
                        UPDATE order_intents
                        SET status = 'terminal', updated_at = $2
                        WHERE id = $1
                        """,
                        current.intent_id,
                        updated_at,
                    )
                await self._append_event(
                    connection,
                    event_type="fill_recorded",
                    intent_id=current.intent_id,
                    previous_status=current.status,
                    new_status=target,
                    details={
                        "fill_id": fill.fill_id,
                        "quantity": canonical_decimal(fill.quantity),
                        "price": canonical_decimal(fill.price),
                        "fee": canonical_decimal(fill.fee),
                        "status": target.value,
                    },
                    occurred_at=fill.filled_at,
                )
                return replace(
                    current,
                    status=target,
                    filled_quantity=new_quantity,
                    average_fill_price=average,
                    cumulative_fee=total_fee,
                    updated_at=updated_at,
                )

    async def record_status(self, intent_id: str, target: OrderStatus) -> OrderRecord:
        resolved = OrderStatus(target)
        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    current = await self._require_order(
                        connection, intent_id, lock=True
                    )
                    if resolved is current.status:
                        return current
                    if (
                        resolved is OrderStatus.FILLED
                        and current.filled_quantity != current.requested_quantity
                    ):
                        raise LedgerInvariantViolation(
                            "FILLED status requires full fill quantity"
                        )
                    if resolved is OrderStatus.PARTIALLY_FILLED and not (
                        Decimal("0")
                        < current.filled_quantity
                        < current.requested_quantity
                    ):
                        raise LedgerInvariantViolation(
                            "PARTIALLY_FILLED requires a partial fill"
                        )
                    if (
                        resolved is OrderStatus.REJECTED
                        and current.filled_quantity != 0
                    ):
                        raise LedgerInvariantViolation(
                            "an order with fills cannot be rejected"
                        )
                    assert_transition(current.status, resolved)
                    now = utc_now()
                    await connection.execute(
                        """
                        UPDATE broker_orders
                        SET status = $2, updated_at = $3
                        WHERE id = $1
                        """,
                        intent_id,
                        resolved.value,
                        now,
                    )
                    await connection.execute(
                        """
                        UPDATE order_intents
                        SET status = $2, updated_at = $3
                        WHERE id = $1
                        """,
                        intent_id,
                        (
                            "terminal"
                            if resolved.terminal
                            else "ambiguous"
                            if resolved is OrderStatus.AMBIGUOUS
                            else "submitted"
                        ),
                        now,
                    )
                    await self._append_event(
                        connection,
                        event_type="status_changed",
                        intent_id=intent_id,
                        previous_status=current.status,
                        new_status=resolved,
                        details={
                            "from": current.status.value,
                            "to": resolved.value,
                        },
                        occurred_at=now,
                    )
                    return replace(current, status=resolved, updated_at=now)
        except (LedgerInvariantViolation, UnknownIntent):
            raise
        except Exception as exc:
            raise LedgerUnavailable("order status persistence failed closed") from exc

    async def get_intent(self, intent_id: str) -> OrderIntent | None:
        try:
            async with self._pool.acquire() as connection:
                row = await connection.fetchrow(
                    "SELECT payload FROM order_intents WHERE id = $1", intent_id
                )
            return None if row is None else self._intent_from_payload(row["payload"])
        except Exception as exc:
            raise LedgerUnavailable("intent lookup failed closed") from exc

    async def get_order(self, intent_id: str) -> OrderRecord | None:
        return await self._get_order("bo.id = $1", intent_id)

    async def get_by_client_order_id(self, client_order_id: str) -> OrderRecord | None:
        return await self._get_order("bo.client_order_id = $1", client_order_id)

    async def get_by_broker_order_id(self, broker_order_id: str) -> OrderRecord | None:
        return await self._get_order("bo.venue_order_id = $1", broker_order_id)

    async def _get_order(self, clause: str, identity: str) -> OrderRecord | None:
        try:
            async with self._pool.acquire() as connection:
                row = await connection.fetchrow(
                    _ORDER_SELECT + f" WHERE {clause}", identity
                )
            return None if row is None else self._order_from_row(row)
        except Exception as exc:
            raise LedgerUnavailable("order lookup failed closed") from exc

    async def fills_for(self, intent_id: str) -> tuple[Fill, ...]:
        return await self._get_fills("f.order_id = $1", intent_id)

    async def list_fills(self) -> tuple[Fill, ...]:
        return await self._get_fills(None, None)

    async def _get_fills(
        self, clause: str | None, identity: str | None
    ) -> tuple[Fill, ...]:
        query = _FILL_SELECT
        args: tuple[str, ...] = ()
        if clause is not None and identity is not None:
            query += f" WHERE {clause}"
            args = (identity,)
        query += " ORDER BY f.received_ts, f.id"
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(query, *args)
            return tuple(self._fill_from_row(row) for row in rows)
        except Exception as exc:
            raise LedgerUnavailable("fill lookup failed closed") from exc

    async def result_for(
        self, intent_id: str, *, duplicate: bool = False
    ) -> ExecutionResult:
        intent = await self.get_intent(intent_id)
        order = await self.get_order(intent_id)
        if intent is None or order is None:
            raise UnknownIntent(intent_id)
        fills = await self.fills_for(intent_id)
        return ExecutionResult(
            intent=intent,
            order=order,
            fills=fills,
            duplicate=duplicate,
        )

    async def list_orders(self) -> tuple[OrderRecord, ...]:
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(
                    _ORDER_SELECT + " ORDER BY bo.created_at, bo.id"
                )
            return tuple(self._order_from_row(row) for row in rows)
        except Exception as exc:
            raise LedgerUnavailable("order listing failed closed") from exc

    async def list_intents(self) -> tuple[OrderIntent, ...]:
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(
                    "SELECT payload FROM order_intents ORDER BY created_at, id"
                )
            return tuple(self._intent_from_payload(row["payload"]) for row in rows)
        except Exception as exc:
            raise LedgerUnavailable("intent listing failed closed") from exc

    async def list_events(self) -> tuple[LedgerEvent, ...]:
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT sequence_id, event_type, order_id, received_ts, payload
                    FROM order_events
                    ORDER BY sequence_id
                    """
                )
            events: list[LedgerEvent] = []
            for row in rows:
                payload = json_object(row["payload"], field_name="order_event.payload")
                details = json_object(
                    payload.get("details", {}), field_name="order_event.details"
                )
                events.append(
                    LedgerEvent(
                        sequence=int(row["sequence_id"]),
                        event_type=str(row["event_type"]),
                        intent_id=str(row["order_id"]),
                        occurred_at=row["received_ts"],
                        details=MappingProxyType(
                            {
                                str(key): None if value is None else str(value)
                                for key, value in details.items()
                            }
                        ),
                    )
                )
            return tuple(events)
        except Exception as exc:
            raise LedgerUnavailable("ledger-event listing failed closed") from exc

    async def _require_order(
        self,
        connection: PostgresConnection,
        intent_id: str,
        *,
        lock: bool,
    ) -> OrderRecord:
        query = _ORDER_SELECT + " WHERE bo.id = $1"
        if lock:
            query += " FOR UPDATE OF bo"
        row = await connection.fetchrow(query, intent_id)
        if row is None:
            raise UnknownIntent(intent_id)
        return self._order_from_row(row)

    async def _append_event(
        self,
        connection: PostgresConnection,
        *,
        event_type: str,
        intent_id: str,
        previous_status: OrderStatus | None,
        new_status: OrderStatus,
        details: Mapping[str, str | None],
        occurred_at: datetime,
    ) -> None:
        payload = {
            "schema": "quantex.ledger-event.v1",
            "event_type": event_type,
            "intent_id": intent_id,
            "previous_status": (
                None if previous_status is None else previous_status.value
            ),
            "new_status": new_status.value,
            "occurred_at": occurred_at,
            "details": dict(details),
        }
        event_id = (
            "evt_"
            + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]
        )
        payload_json = canonical_json(payload)
        await connection.execute(
            """
            INSERT INTO order_events (
                order_id, event_id, event_type, previous_status, new_status,
                exchange_ts, received_ts, payload, payload_checksum
            )
            VALUES ($1, $2, $3, $4, $5, $6, $6, $7::jsonb, $8)
            """,
            intent_id,
            event_id,
            event_type,
            None if previous_status is None else previous_status.value,
            new_status.value,
            occurred_at,
            payload_json,
            compute_payload_checksum(payload),
        )

    @staticmethod
    def _intent_json(intent: OrderIntent) -> str:
        return canonical_json(
            {
                "schema": "quantex.order-intent.v1",
                "intent_id": intent.intent_id,
                "client_order_id": intent.client_order_id,
                "account_id": intent.account_id,
                "decision_id": intent.decision_id,
                "signal_id": intent.signal_id,
                "venue": intent.venue,
                "market_type": intent.market_type,
                "instrument": intent.instrument,
                "side": intent.side.value,
                "order_type": intent.order_type.value,
                "quantity": intent.quantity,
                "source_mode": intent.source_mode.value,
                "limit_price": intent.limit_price,
                "reference_price": intent.reference_price,
                "created_at": intent.created_at,
            }
        )

    @staticmethod
    def _intent_from_payload(payload: object) -> OrderIntent:
        data = json_object(payload, field_name="order_intent.payload")
        if data.get("schema") != "quantex.order-intent.v1":
            raise ValueError("unsupported order-intent payload schema")
        return OrderIntent(
            intent_id=str(data["intent_id"]),
            client_order_id=str(data["client_order_id"]),
            account_id=str(data["account_id"]),
            decision_id=str(data["decision_id"]),
            signal_id=str(data["signal_id"]),
            venue=str(data["venue"]),
            market_type=str(data["market_type"]),
            instrument=str(data["instrument"]),
            side=OrderSide(str(data["side"])),
            order_type=OrderType(str(data["order_type"])),
            quantity=Decimal(str(data["quantity"])),
            source_mode=SourceMode(str(data["source_mode"])),
            limit_price=(
                None
                if data.get("limit_price") is None
                else Decimal(str(data["limit_price"]))
            ),
            reference_price=(
                None
                if data.get("reference_price") is None
                else Decimal(str(data["reference_price"]))
            ),
            created_at=PostgresExecutionLedger._datetime(data["created_at"]),
        )

    @staticmethod
    def _order_from_row(row: Mapping[str, Any]) -> OrderRecord:
        intent = PostgresExecutionLedger._intent_from_payload(row["intent_payload"])
        return OrderRecord(
            intent_id=intent.intent_id,
            client_order_id=intent.client_order_id,
            account_id=intent.account_id,
            venue=intent.venue,
            market_type=intent.market_type,
            instrument=intent.instrument,
            side=intent.side,
            order_type=intent.order_type,
            requested_quantity=row["requested_quantity"],
            status=OrderStatus(str(row["status"])),
            broker_order_id=(
                None if row["broker_order_id"] is None else str(row["broker_order_id"])
            ),
            limit_price=intent.limit_price,
            filled_quantity=row["filled_quantity"],
            average_fill_price=row["average_price"],
            cumulative_fee=row["total_fee"],
            created_at=row["order_created_at"],
            updated_at=row["order_updated_at"],
        )

    @staticmethod
    def _fill_json(fill: Fill) -> str:
        return canonical_json(
            {
                "schema": "quantex.fill.v1",
                "fill_id": fill.fill_id,
                "intent_id": fill.intent_id,
                "client_order_id": fill.client_order_id,
                "broker_order_id": fill.broker_order_id,
                "instrument": fill.instrument,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "price": fill.price,
                "fee": fill.fee,
                "filled_at": fill.filled_at,
            }
        )

    @staticmethod
    def _fill_from_row(row: Mapping[str, Any]) -> Fill:
        intent = PostgresExecutionLedger._intent_from_payload(row["intent_payload"])
        broker_order_id = row["broker_order_id"]
        if broker_order_id is None:
            raise ValueError("stored fill has no broker order identity")
        return Fill(
            fill_id=str(row["fill_id"]),
            intent_id=intent.intent_id,
            client_order_id=intent.client_order_id,
            broker_order_id=str(broker_order_id),
            instrument=intent.instrument,
            side=intent.side,
            quantity=row["quantity"],
            price=row["price"],
            fee=row["fee"],
            filled_at=row["filled_at"],
        )

    @staticmethod
    def _datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise ValueError("timestamp must be an ISO-8601 string")
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

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
        changed = [
            name for name in fields if getattr(current, name) != getattr(incoming, name)
        ]
        if changed:
            raise LedgerInvariantViolation(
                "broker record changed immutable fields: " + ", ".join(changed)
            )
