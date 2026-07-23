"""Transactional inbox/outbox behavior at the PostgreSQL adapter boundary."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from packages.event_bus import CoreSubject, JetStreamTransportError
from workers.durability import PostgresInboxOutbox


class FakeConnection:
    def __init__(self) -> None:
        self.inbox: set[str] = set()
        self.outbox: dict[str, dict[str, object]] = {}
        self.dead_letters: list[tuple[object, ...]] = []

    def transaction(self):
        @asynccontextmanager
        async def transaction():
            yield

        return transaction()

    async def fetchval(self, query: str, *args: object):
        if "INSERT INTO event_inbox" in query:
            event_id = str(args[2])
            if event_id in self.inbox:
                return None
            self.inbox.add(event_id)
            return event_id
        raise AssertionError(query)

    async def execute(self, query: str, *args: object) -> str:
        if "INSERT INTO event_outbox" in query:
            self.outbox[str(args[0])] = {
                "event_id": args[0],
                "subject": args[3],
                "payload": args[4],
                "headers": args[6],
                "status": "pending",
                "publish_attempts": 0,
            }
        elif "status = 'published'" in query:
            self.outbox[str(args[0])]["status"] = "published"
        elif "INSERT INTO dead_letter_events" in query:
            self.dead_letters.append(args)
        elif "status = 'dead_letter'" in query:
            self.outbox[str(args[0])]["status"] = "dead_letter"
        elif "next_attempt_at = NOW()" in query:
            self.outbox[str(args[0])]["status"] = "pending"
            self.outbox[str(args[0])]["retry_delay_seconds"] = args[2]
        elif "UPDATE event_inbox" not in query and "UPDATE event_outbox" not in query:
            raise AssertionError(query)
        return "UPDATE 1"

    async def fetchrow(self, query: str, *args: object):
        if "FOR UPDATE SKIP LOCKED" in query:
            event_id = str(args[0]) if args and args[0] is not None else next(
                (
                    key
                    for key, value in self.outbox.items()
                    if value["status"] in {"pending", "publishing"}
                ),
                "",
            )
            row = self.outbox.get(event_id)
            if row is None or row["status"] == "published":
                return None
            selected = dict(row)
            row["status"] = "publishing"
            row["publish_attempts"] = int(row["publish_attempts"]) + 1
            return selected
        raise AssertionError(query)


class FakePool:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def acquire(self):
        @asynccontextmanager
        async def acquire():
            yield self.connection

        return acquire()


class FakeBus:
    def __init__(self) -> None:
        self.publications = 0
        self.fail = False

    async def publish(self, *_args: object, **_kwargs: object):
        self.publications += 1
        if self.fail:
            raise JetStreamTransportError("offline")
        return SimpleNamespace()


def route_event() -> dict[str, object]:
    return {
        "source_event_id": "source-1",
        "source_subject": CoreSubject.MARKET_RAW,
        "source_payload": {"event_id": "source-1"},
        "payload_checksum": "a" * 64,
        "output_event_id": "output-1",
        "aggregate_type": "market",
        "aggregate_id": "source-1",
        "subject": CoreSubject.MARKET_VALIDATED,
        "payload": b'{"event_id":"source-1"}',
        "message_id": "market-validated:source-1",
    }


@pytest.mark.asyncio
async def test_duplicate_inbox_returns_without_second_publish() -> None:
    pool, bus = FakePool(), FakeBus()
    router = PostgresInboxOutbox(
        pool, bus, stream_name="QUANTEX_CORE", durable_name="market-v1"
    )
    event = route_event()

    assert await router.route(**event) is True
    assert await router.route(**event) is False
    assert bus.publications == 1


@pytest.mark.asyncio
async def test_one_input_routes_two_outputs_atomically_and_only_once() -> None:
    pool, bus = FakePool(), FakeBus()
    router = PostgresInboxOutbox(
        pool, bus, stream_name="QUANTEX_CORE", durable_name="execution-v1"
    )
    source = route_event()
    outputs = (
        {
            "output_event_id": "intent-1",
            "aggregate_type": "order_intent",
            "aggregate_id": "order-1",
            "subject": CoreSubject.ORDER_INTENT,
            "payload": b'{"event_id":"intent-1"}',
            "message_id": "intent:intent-1",
        },
        {
            "output_event_id": "order-1",
            "aggregate_type": "order",
            "aggregate_id": "order-1",
            "subject": CoreSubject.ORDER_UPDATED,
            "payload": b'{"event_id":"order-1"}',
            "message_id": "order:order-1",
        },
    )

    assert await router.route_many(
        source_event_id=str(source["source_event_id"]),
        source_subject=CoreSubject.RISK_APPROVED,
        source_payload=source["source_payload"],
        payload_checksum=str(source["payload_checksum"]),
        outputs=outputs,
    ) is True
    assert await router.route_many(
        source_event_id=str(source["source_event_id"]),
        source_subject=CoreSubject.RISK_APPROVED,
        source_payload=source["source_payload"],
        payload_checksum=str(source["payload_checksum"]),
        outputs=outputs,
    ) is False
    assert set(pool.connection.outbox) == {"intent-1", "order-1"}
    assert bus.publications == 2


@pytest.mark.asyncio
async def test_publish_failure_leaves_outbox_retryable() -> None:
    pool, bus = FakePool(), FakeBus()
    bus.fail = True
    router = PostgresInboxOutbox(
        pool, bus, stream_name="QUANTEX_CORE", durable_name="market-v1"
    )

    with pytest.raises(JetStreamTransportError):
        await router.route(**route_event())

    assert pool.connection.outbox["output-1"]["status"] == "pending"
    assert pool.connection.outbox["output-1"]["retry_delay_seconds"] == 2

    bus.fail = False
    assert await router.dispatch_next() is True
    assert pool.connection.outbox["output-1"]["status"] == "published"
    assert bus.publications == 2


@pytest.mark.asyncio
async def test_terminal_publish_failure_moves_event_to_dlq() -> None:
    pool, bus = FakePool(), FakeBus()
    bus.fail = True
    router = PostgresInboxOutbox(
        pool,
        bus,
        stream_name="QUANTEX_CORE",
        durable_name="market-v1",
        max_publish_attempts=1,
    )

    with pytest.raises(JetStreamTransportError):
        await router.route(**route_event())

    assert pool.connection.outbox["output-1"]["status"] == "dead_letter"
    assert len(pool.connection.dead_letters) == 1
    assert "offline" not in str(pool.connection.dead_letters[0])
