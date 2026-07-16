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
