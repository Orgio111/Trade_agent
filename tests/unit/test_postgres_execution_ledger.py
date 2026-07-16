"""Stateful fake-PostgreSQL tests for durable execution-ledger semantics."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from decimal import Decimal
import json

import pytest

from packages.execution import (
    DuplicateFill,
    Fill,
    LedgerInvariantViolation,
    LedgerUnavailable,
    OrderSide,
    OrderStatus,
    PostgresExecutionLedger,
)
from tests.fakes.durable import NOW, intent


class FakeLedgerConnection:
    def __init__(self) -> None:
        self.approvals = {"risk_0123456789abcdef01234567": "trace-durable-001"}
        self.intents: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.fills: dict[str, dict] = {}
        self.events: list[tuple] = []
        self.fail = False

    @asynccontextmanager
    async def transaction(self):
        yield self

    async def fetchrow(self, query, *args):
        if self.fail:
            raise ConnectionError("database unavailable")
        sql = " ".join(query.split()).lower()
        if "from order_intents" in sql and "or client_order_id" in sql:
            candidate = next(
                (
                    row
                    for row in self.intents.values()
                    if row["id"] == args[0]
                    or row["client_order_id"] == args[1]
                    or row["decision_id"] == args[2]
                ),
                None,
            )
            return None if candidate is None else {"id": candidate["id"]}
        if "from risk_decisions" in sql:
            trace_id = self.approvals.get(args[0])
            return None if trace_id is None else {"trace_id": trace_id}
        if "from broker_orders as bo" in sql:
            order = self._find_order(sql, args[0])
            return None if order is None else self._order_row(order)
        if "where venue_order_id = $1" in sql:
            order = next(
                (
                    row
                    for row in self.orders.values()
                    if row["broker_order_id"] == args[0]
                ),
                None,
            )
            return None if order is None else {"id": order["id"]}
        if "select id from fills" in sql:
            return None if args[0] not in self.fills else {"id": args[0]}
        if "select payload from order_intents" in sql:
            row = self.intents.get(args[0])
            return None if row is None else {"payload": row["payload"]}
        raise AssertionError(f"unexpected fetchrow SQL: {sql}")

    async def execute(self, query, *args):
        if self.fail:
            raise ConnectionError("database unavailable")
        sql = " ".join(query.split()).lower()
        if sql.startswith("insert into order_intents"):
            self.intents[args[0]] = {
                "id": args[0],
                "client_order_id": args[1],
                "decision_id": args[2],
                "payload": args[13],
            }
        elif sql.startswith("insert into broker_orders"):
            self.orders[args[0]] = {
                "id": args[0],
                "client_order_id": args[1],
                "broker_order_id": None,
                "status": "pending_submit",
                "requested_quantity": args[2],
                "filled_quantity": Decimal("0"),
                "average_price": None,
                "total_fee": Decimal("0"),
                "created_at": args[3],
                "updated_at": args[3],
            }
        elif sql.startswith("insert into order_events"):
            self.events.append(args)
        elif sql.startswith("insert into fills"):
            self.fills[args[0]] = {
                "fill_id": args[0],
                "intent_id": args[1],
                "quantity": args[2],
                "price": args[3],
                "fee": args[4],
                "filled_at": args[5],
            }
        elif sql.startswith("update broker_orders") and "venue_order_id" in sql:
            order = self.orders[args[0]]
            order.update(
                broker_order_id=args[1],
                status=args[2],
                updated_at=args[3],
            )
        elif sql.startswith("update broker_orders") and "filled_quantity" in sql:
            order = self.orders[args[0]]
            order.update(
                status=args[1],
                filled_quantity=args[2],
                average_price=args[3],
                total_fee=args[4],
                updated_at=args[5],
            )
        elif sql.startswith("update broker_orders"):
            self.orders[args[0]].update(status=args[1], updated_at=args[2])
        elif sql.startswith("update order_intents"):
            pass
        else:
            raise AssertionError(f"unexpected execute SQL: {sql}")
        return "OK"

    async def fetch(self, query, *args):
        if self.fail:
            raise ConnectionError("database unavailable")
        sql = " ".join(query.split()).lower()
        if "from fills as f" in sql:
            rows = self.fills.values()
            if args:
                rows = [row for row in rows if row["intent_id"] == args[0]]
            return [self._fill_row(row) for row in rows]
        if "from broker_orders as bo" in sql:
            return [self._order_row(row) for row in self.orders.values()]
        if "select payload from order_intents" in sql:
            return [{"payload": row["payload"]} for row in self.intents.values()]
        if "from order_events" in sql:
            return []
        raise AssertionError(f"unexpected fetch SQL: {sql}")

    def _find_order(self, sql: str, identity: str):
        if "where bo.client_order_id" in sql:
            return next(
                (
                    row
                    for row in self.orders.values()
                    if row["client_order_id"] == identity
                ),
                None,
            )
        if "where bo.venue_order_id" in sql:
            return next(
                (
                    row
                    for row in self.orders.values()
                    if row["broker_order_id"] == identity
                ),
                None,
            )
        return self.orders.get(identity)

    def _order_row(self, order):
        return {
            "intent_payload": self.intents[order["id"]]["payload"],
            **order,
            "order_created_at": order["created_at"],
            "order_updated_at": order["updated_at"],
        }

    def _fill_row(self, fill):
        order = self.orders[fill["intent_id"]]
        return {
            **fill,
            "broker_order_id": order["broker_order_id"],
            "intent_payload": self.intents[fill["intent_id"]]["payload"],
        }


class FakePool:
    def __init__(self, connection: FakeLedgerConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.mark.asyncio
async def test_persist_submit_fill_round_trip_is_durable_and_exact() -> None:
    connection = FakeLedgerConnection()
    ledger = PostgresExecutionLedger(FakePool(connection))
    order_intent = intent()

    pending = await ledger.persist_intent(order_intent)
    open_order = replace(
        pending,
        broker_order_id="paper-order-1",
        status=OrderStatus.OPEN,
        updated_at=NOW,
    )
    await ledger.record_submission(open_order)
    fill = Fill(
        fill_id="paper-fill-1",
        intent_id=order_intent.intent_id,
        client_order_id=order_intent.client_order_id,
        broker_order_id="paper-order-1",
        instrument=order_intent.instrument,
        side=OrderSide.BUY,
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0.1"),
        filled_at=NOW,
    )
    filled = await ledger.record_fill(fill)
    result = await ledger.result_for(order_intent.intent_id)

    assert pending.status is OrderStatus.PENDING_SUBMIT
    assert filled.status is OrderStatus.FILLED
    assert result.intent == order_intent
    assert result.order == filled
    assert result.fills == (fill,)
    assert len(connection.events) == 3

    with pytest.raises(DuplicateFill, match="already used"):
        await ledger.record_fill(fill)


@pytest.mark.asyncio
async def test_intent_without_persisted_approved_decision_fails_closed() -> None:
    connection = FakeLedgerConnection()
    connection.approvals.clear()
    ledger = PostgresExecutionLedger(FakePool(connection))

    with pytest.raises(LedgerInvariantViolation, match="approved risk decision"):
        await ledger.persist_intent(intent())

    assert connection.intents == {}
    assert connection.orders == {}


@pytest.mark.asyncio
async def test_database_failure_never_looks_like_an_empty_ledger() -> None:
    connection = FakeLedgerConnection()
    connection.fail = True
    ledger = PostgresExecutionLedger(FakePool(connection))

    with pytest.raises(LedgerUnavailable, match="failed closed"):
        await ledger.get_order("int_missing")


def test_intent_payload_is_canonical_json_not_pickle() -> None:
    payload = PostgresExecutionLedger._intent_json(intent())

    parsed = json.loads(payload)
    assert parsed["schema"] == "quantex.order-intent.v1"
    assert parsed["source_mode"] == "paper_live"
