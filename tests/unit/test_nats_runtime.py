"""Focused tests for the shared canonical NATS lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from nats.js.api import StreamConfig
from nats.js.errors import NotFoundError
from pydantic import SecretStr

from packages.domain import SourceMode
from packages.event_bus import CoreSubject
from workers.config import WorkerSettings
from workers.nats_runtime import (
    NatsLifecycleState,
    NatsRuntimeError,
    NatsStreamDriftError,
    connect_nats_runtime,
    consumer_settings,
)


class FakeJetStream:
    def __init__(
        self,
        *,
        subjects: list[str] | None = None,
        existing_config: StreamConfig | None = None,
    ) -> None:
        self.subjects = subjects
        self.existing_config = existing_config
        self.added_config = None

    async def stream_info(self, name: str):
        if self.existing_config is not None:
            return SimpleNamespace(config=self.existing_config)
        if self.subjects is None:
            raise NotFoundError(stream=name)
        return SimpleNamespace(
            config=SimpleNamespace(name=name, subjects=self.subjects)
        )

    async def add_stream(self, *, config):
        self.added_config = config
        return SimpleNamespace(config=config)

    async def publish(self, *_args, **_kwargs):  # pragma: no cover - adapter owns it
        raise AssertionError("unexpected publish")

    async def subscribe(self, *_args, **_kwargs):  # pragma: no cover - adapter owns it
        raise AssertionError("unexpected subscribe")

    async def consumer_info(self, _stream: str, _durable: str):
        return SimpleNamespace(num_pending=0, num_ack_pending=0)


class FakeConnection:
    def __init__(self, context: FakeJetStream) -> None:
        self.context = context
        self.is_closed = False
        self.drained = 0
        self.closed = 0

    def jetstream(self) -> FakeJetStream:
        return self.context

    async def drain(self) -> None:
        self.drained += 1
        self.is_closed = True

    async def close(self) -> None:
        self.closed += 1
        self.is_closed = True


class FakeLeaseStore:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def heartbeat(self, **record: object) -> None:
        self.records.append(record)

    async def close(self) -> None:
        return None


def settings() -> WorkerSettings:
    return WorkerSettings(
        service_name="market-data-worker",
        mode=SourceMode.PAPER_LIVE,
        nats_url="nats://127.0.0.1:4222",
        database_url=SecretStr("postgresql://127.0.0.1/quantex"),
        durable_name="market-data-worker-v1",
    )


@pytest.mark.asyncio
async def test_connect_creates_exact_bounded_core_stream(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)

    async def fake_connect(**_kwargs):
        return connection

    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    runtime = await connect_nats_runtime(settings())

    assert set(context.added_config.subjects) == {
        subject.value for subject in CoreSubject
    }
    assert context.added_config.max_bytes == 512 * 1024 * 1024
    assert context.added_config.max_msg_size == 1024 * 1024
    assert context.added_config.num_replicas == 1
    await runtime.close()
    assert connection.drained == 1
    assert connection.closed == 0
    assert runtime._started_at is not None


@pytest.mark.asyncio
async def test_temporary_disconnect_marks_unready_then_recovers(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)
    callbacks = {}

    async def fake_connect(**kwargs):
        callbacks.update(kwargs)
        return connection

    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    runtime = await connect_nats_runtime(settings())

    await callbacks["disconnected_cb"]()
    assert runtime.state is NatsLifecycleState.DISCONNECTED
    assert runtime.ready is False

    await callbacks["reconnected_cb"]()
    assert runtime.state is NatsLifecycleState.CONNECTED
    assert runtime.ready is True


@pytest.mark.asyncio
async def test_unexpected_terminal_close_fails_worker_wait(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)
    callbacks = {}

    async def fake_connect(**kwargs):
        callbacks.update(kwargs)
        return connection

    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    runtime = await connect_nats_runtime(settings())

    await callbacks["closed_cb"]()

    with pytest.raises(NatsRuntimeError, match="terminally closed"):
        await runtime.wait()
    assert runtime.state is NatsLifecycleState.FATAL


@pytest.mark.asyncio
async def test_worker_lease_expires_when_transport_disconnects(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)
    callbacks = {}
    store = FakeLeaseStore()

    async def fake_connect(**kwargs):
        callbacks.update(kwargs)
        return connection

    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    runtime = await connect_nats_runtime(settings(), lease_store=store)
    await runtime.heartbeat_once()
    await callbacks["disconnected_cb"]()
    await runtime.heartbeat_once()

    assert store.records[0]["status"] == "ready"
    assert store.records[0]["lease_seconds"] > 0
    assert store.records[1]["status"] == "disconnected"
    assert store.records[1]["lease_seconds"] == 0


@pytest.mark.asyncio
async def test_consumer_lag_above_limit_marks_runtime_unready(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)

    async def fake_connect(**_kwargs):
        return connection

    async def consumer_info(_stream: str, _durable: str):
        return SimpleNamespace(num_pending=65, num_ack_pending=0)

    context.consumer_info = consumer_info
    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    configured = settings().model_copy(update={"max_consumer_lag": 64})
    runtime = await connect_nats_runtime(configured)

    await runtime.heartbeat_once()

    assert runtime.consumer_lag == 65
    assert runtime.ready is False


@pytest.mark.asyncio
async def test_deleted_consumer_marks_runtime_unready(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)

    async def fake_connect(**_kwargs):
        return connection

    async def consumer_info(_stream: str, _durable: str):
        raise NotFoundError(stream="QUANTEX_CORE")

    context.consumer_info = consumer_info
    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    runtime = await connect_nats_runtime(settings())

    await runtime.heartbeat_once()

    assert runtime.ready is False
    assert runtime.last_error == "consumer monitor failed: NotFoundError"


@pytest.mark.asyncio
async def test_producer_runtime_does_not_require_consumer(monkeypatch) -> None:
    context = FakeJetStream()
    connection = FakeConnection(context)

    async def fake_connect(**_kwargs):
        return connection

    async def consumer_info(_stream: str, _durable: str):
        raise AssertionError("producer must not query consumer state")

    context.consumer_info = consumer_info
    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)
    runtime = await connect_nats_runtime(settings(), monitor_consumer=False)

    await runtime.heartbeat_once()

    assert runtime.ready is True
    assert runtime.consumer_lag is None


@pytest.mark.asyncio
async def test_connect_fails_closed_on_existing_subject_drift(monkeypatch) -> None:
    context = FakeJetStream(subjects=[CoreSubject.MARKET_RAW.value])
    connection = FakeConnection(context)

    async def fake_connect(**_kwargs):
        return connection

    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)

    with pytest.raises(NatsStreamDriftError, match="subject authority drift"):
        await connect_nats_runtime(settings())

    assert connection.closed == 1
    assert connection.drained == 0


@pytest.mark.asyncio
async def test_connect_fails_closed_on_unbounded_existing_stream(monkeypatch) -> None:
    context = FakeJetStream(
        existing_config=StreamConfig(
            name="QUANTEX_CORE",
            subjects=[subject.value for subject in CoreSubject],
        )
    )
    connection = FakeConnection(context)

    async def fake_connect(**_kwargs):
        return connection

    monkeypatch.setattr("workers.nats_runtime.nats.connect", fake_connect)

    with pytest.raises(NatsStreamDriftError, match="durability bounds drift"):
        await connect_nats_runtime(settings())

    assert connection.closed == 1


def test_consumer_contract_is_explicit_bounded_and_settings_owned() -> None:
    configured = consumer_settings(settings())

    assert configured.stream == "QUANTEX_CORE"
    assert configured.durable_name == "market-data-worker-v1"
    assert configured.max_ack_pending == 64
    assert configured.max_payload_bytes == 1024 * 1024
    assert configured.max_deliver == 5
