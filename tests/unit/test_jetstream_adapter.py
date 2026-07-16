"""Unit contracts for bounded manual-ack JetStream delivery."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from packages.event_bus import (
    CoreSubject,
    DurableConsumerSettings,
    JetStreamConfigurationError,
    JetStreamEventBus,
    JetStreamTransportError,
)


@dataclass
class FakeAck:
    stream: str = "CORE"
    seq: int = 7
    duplicate: bool = False


class FakeContext:
    def __init__(self) -> None:
        self.publish_call = None
        self.subscribe_call = None
        self.callback = None
        self.fail_publish = False

    async def publish(self, subject, payload=b"", **kwargs):
        if self.fail_publish:
            raise ConnectionError("offline")
        self.publish_call = (subject, payload, kwargs)
        return FakeAck()

    async def subscribe(self, subject, **kwargs):
        self.subscribe_call = (subject, kwargs)
        self.callback = kwargs["cb"]
        return "subscription"


class FakeMessage:
    def __init__(self, *, delivery: int = 1, data: bytes = b"{}") -> None:
        self.subject = CoreSubject.RISK_APPROVED.value
        self.data = data
        self.metadata = SimpleNamespace(num_delivered=delivery)
        self.acked = 0
        self.naks: list[float | None] = []
        self.terminated = 0

    async def ack(self) -> None:
        self.acked += 1

    async def nak(self, delay=None) -> None:
        self.naks.append(delay)

    async def term(self) -> None:
        self.terminated += 1


def settings(**overrides) -> DurableConsumerSettings:
    values = {
        "stream": "CORE",
        "durable_name": "execution-worker-v1",
        "max_deliver": 3,
        "max_ack_pending": 8,
        "retry_delay_seconds": 0.5,
        "max_payload_bytes": 64,
    }
    values.update(overrides)
    return DurableConsumerSettings(**values)


@pytest.mark.asyncio
async def test_publish_requires_explicit_deduplication_identity() -> None:
    context = FakeContext()
    bus = JetStreamEventBus(context)

    receipt = await bus.publish(
        CoreSubject.RISK_APPROVED,
        b'{"decision_id":"risk-1"}',
        message_id="risk-1",
        stream="CORE",
    )

    assert receipt.sequence == 7
    assert context.publish_call[2]["headers"] == {"Nats-Msg-Id": "risk-1"}
    with pytest.raises(JetStreamConfigurationError, match="message_id"):
        await bus.publish(CoreSubject.RISK_APPROVED, b"{}", message_id="bad id")


@pytest.mark.asyncio
async def test_publish_transport_failure_is_fail_closed() -> None:
    context = FakeContext()
    context.fail_publish = True

    with pytest.raises(JetStreamTransportError, match="publish failed"):
        await JetStreamEventBus(context).publish(
            CoreSubject.ORDER_INTENT, b"{}", message_id="intent-1"
        )


@pytest.mark.asyncio
async def test_successful_handler_acknowledges_once_with_explicit_consumer() -> None:
    context = FakeContext()
    received = []

    async def handler(payload: bytes) -> None:
        received.append(payload)

    result = await JetStreamEventBus(context).subscribe(
        CoreSubject.RISK_APPROVED,
        handler,
        settings=settings(),
    )
    message = FakeMessage()
    await context.callback(message)

    assert result == "subscription"
    assert received == [b"{}"]
    assert message.acked == 1
    assert message.naks == []
    kwargs = context.subscribe_call[1]
    assert kwargs["manual_ack"] is True
    assert kwargs["durable"] == "execution-worker-v1"
    assert kwargs["config"].max_deliver == 3
    assert kwargs["config"].max_ack_pending == 8


@pytest.mark.asyncio
async def test_handler_failure_is_naked_then_terminated_at_delivery_bound() -> None:
    context = FakeContext()
    failures = []

    async def handler(_payload: bytes) -> None:
        raise ValueError("poison")

    async def on_failure(failure) -> None:
        failures.append(failure)

    await JetStreamEventBus(context).subscribe(
        CoreSubject.RISK_APPROVED,
        handler,
        settings=settings(),
        on_failure=on_failure,
    )
    first = FakeMessage(delivery=1)
    final = FakeMessage(delivery=3)
    await context.callback(first)
    await context.callback(final)

    assert first.naks == [0.5]
    assert first.terminated == 0
    assert final.naks == []
    assert final.terminated == 1
    assert [failure.terminal for failure in failures] == [False, True]
    assert all(failure.error_type == "ValueError" for failure in failures)


@pytest.mark.asyncio
async def test_invalid_payload_size_is_terminal_without_calling_handler() -> None:
    context = FakeContext()
    called = False

    async def handler(_payload: bytes) -> None:
        nonlocal called
        called = True

    await JetStreamEventBus(context).subscribe(
        CoreSubject.RISK_APPROVED,
        handler,
        settings=settings(max_payload_bytes=2),
    )
    message = FakeMessage(data=b"oversized")
    await context.callback(message)

    assert called is False
    assert message.terminated == 1
