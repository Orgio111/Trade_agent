"""Boundary tests for the canonical paper execution worker."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest
from pydantic import SecretStr

from packages.brokers import PaperBroker
from packages.domain import SourceMode
from packages.event_bus import CoreSubject, JetStreamEventBus
from packages.execution import ExecutionResult, OrderIntent
from tests.unit.test_async_execution_service import NOW, approved_event
from workers.config import WorkerSettings
from workers.contracts import OrderIntentEvent, OrderUpdatedEvent, RiskDecisionEvent
from workers.execution.runtime import (
    ExecutionRuntime,
    ExecutionRuntimeError,
    RuntimeModeMismatch,
)


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[CoreSubject, bytes, str, str | None]] = []
        self.subscription: tuple[CoreSubject, object, object] | None = None

    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> object:
        self.published.append((subject, payload, message_id, stream))
        return object()

    async def subscribe(
        self,
        subject: CoreSubject,
        handler: object,
        *,
        settings: object,
        on_failure: object | None = None,
    ) -> object:
        del on_failure
        self.subscription = (subject, handler, settings)
        return object()


class FakeExecutionService:
    def __init__(self) -> None:
        self.calls: list[tuple[RiskDecisionEvent, datetime]] = []

    async def submit(
        self,
        event: RiskDecisionEvent,
        *,
        execution_at: datetime,
    ) -> ExecutionResult:
        self.calls.append((event, execution_at))
        intent = OrderIntent.from_approved_decision(
            event.decision,
            event.candidate,
            account_id="paper-main",
            created_at=execution_at,
        )
        broker = PaperBroker()
        order = broker.submit_order(intent, event.market)
        return ExecutionResult(
            intent=intent,
            order=order,
            fills=broker.list_fills(intent.client_order_id),
        )


class FakeDurableRouter:
    def __init__(self) -> None:
        self.routed: list[dict[str, object]] = []

    async def route_many(self, **event: object) -> bool:
        self.routed.append(event)
        return True


def settings(
    *,
    mode: SourceMode = SourceMode.PAPER_LIVE,
    account_id: str = "paper-main",
) -> WorkerSettings:
    return WorkerSettings(
        service_name="execution-worker",
        mode=mode,
        nats_url="nats://127.0.0.1:4222",
        database_url=SecretStr("postgresql://quantex:secret@127.0.0.1/quantex"),
        account_id=account_id,
        durable_name="execution-paper-v1",
    )


@pytest.mark.asyncio
async def test_runtime_publishes_deterministic_intent_before_order_update() -> None:
    bus = FakeBus()
    service = FakeExecutionService()
    runtime = ExecutionRuntime(
        settings=settings(),
        bus=cast(JetStreamEventBus, bus),
        service=cast(Any, service),
        clock=lambda: NOW,
    )

    event = approved_event()
    await runtime.handle(event.canonical_json().encode("utf-8"))

    assert service.calls == [(event, NOW)]
    assert [published[0] for published in bus.published] == [
        CoreSubject.ORDER_INTENT,
        CoreSubject.ORDER_UPDATED,
    ]
    intent_event = OrderIntentEvent.model_validate_json(bus.published[0][1])
    updated_event = OrderUpdatedEvent.model_validate_json(bus.published[1][1])
    assert updated_event.intent_event_id == intent_event.event_id
    assert bus.published[0][2] == f"intent:{intent_event.event_id}"
    assert bus.published[1][2] == f"order:{updated_event.event_id}"
    assert all(item[3] == "QUANTEX_CORE" for item in bus.published)


@pytest.mark.asyncio
async def test_runtime_routes_intent_and_update_as_one_durable_output_set() -> None:
    bus = FakeBus()
    service = FakeExecutionService()
    router = FakeDurableRouter()
    runtime = ExecutionRuntime(
        settings=settings(),
        bus=cast(JetStreamEventBus, bus),
        service=cast(Any, service),
        durable_router=router,  # type: ignore[arg-type]
        clock=lambda: NOW,
    )
    event = approved_event()

    await runtime.handle(event.canonical_json().encode("utf-8"))

    assert bus.published == []
    assert len(router.routed) == 1
    routed = router.routed[0]
    assert routed["source_event_id"] == str(event.event_id)
    assert routed["payload_checksum"] == event.content_sha256
    outputs = routed["outputs"]
    assert isinstance(outputs, tuple)
    assert [output["subject"] for output in outputs] == [
        CoreSubject.ORDER_INTENT,
        CoreSubject.ORDER_UPDATED,
    ]


@pytest.mark.asyncio
async def test_runtime_starts_only_the_approved_subject_consumer() -> None:
    bus = FakeBus()
    runtime = ExecutionRuntime(
        settings=settings(),
        bus=cast(JetStreamEventBus, bus),
        service=cast(Any, FakeExecutionService()),
    )

    await runtime.start()

    assert bus.subscription is not None
    assert bus.subscription[0] is CoreSubject.RISK_APPROVED


@pytest.mark.asyncio
async def test_runtime_fails_closed_on_mode_or_account_mismatch() -> None:
    event = approved_event()
    for runtime_settings, error in (
        (settings(mode=SourceMode.REPLAY), RuntimeModeMismatch),
        (settings(account_id="paper-other"), ExecutionRuntimeError),
    ):
        bus = FakeBus()
        service = FakeExecutionService()
        runtime = ExecutionRuntime(
            settings=runtime_settings,
            bus=cast(JetStreamEventBus, bus),
            service=cast(Any, service),
            clock=lambda: datetime(2026, 7, 16, 7, 0, tzinfo=UTC),
        )

        with pytest.raises(error):
            await runtime.handle(event.canonical_json().encode("utf-8"))
        assert service.calls == []
        assert bus.published == []
