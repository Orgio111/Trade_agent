"""Shared local NATS JetStream lifecycle for canonical worker processes.

The runtime creates one bounded file-backed stream when it is absent and
refuses to start when an existing stream's exact subject authority has drifted.
It never reads credentials from dotenv files and transports canonical bytes
only; domain authorization remains in the worker services.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import signal
from typing import cast

import nats
from nats.aio.client import Client as NatsClient
from nats.js.api import RetentionPolicy, StorageType, StreamConfig, StreamInfo
from nats.js.client import JetStreamContext
from nats.js.errors import NotFoundError

from packages.event_bus import DurableConsumerSettings, JetStreamEventBus
from packages.event_bus.jetstream import JetStreamContext as EventBusJetStreamContext
from packages.event_bus.subjects import CoreSubject
from workers.config import WorkerSettings


_CORE_STREAM_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
_CORE_STREAM_MAX_BYTES = 512 * 1024 * 1024
_CORE_MESSAGE_MAX_BYTES = 1024 * 1024
_CORE_DUPLICATE_WINDOW_SECONDS = 120.0
_DRAIN_TIMEOUT_SECONDS = 5.0


class NatsRuntimeError(RuntimeError):
    """The bounded local NATS runtime could not be started or stopped safely."""


class NatsStreamDriftError(NatsRuntimeError):
    """The existing core stream does not own the exact canonical subjects."""


def _core_subjects() -> tuple[str, ...]:
    return tuple(sorted(subject.value for subject in CoreSubject))


def _stream_config(name: str) -> StreamConfig:
    return StreamConfig(
        name=name,
        description="Canonical local paper/replay event stream",
        subjects=list(_core_subjects()),
        retention=RetentionPolicy.LIMITS,
        max_consumers=32,
        max_bytes=_CORE_STREAM_MAX_BYTES,
        max_age=_CORE_STREAM_MAX_AGE_SECONDS,
        max_msg_size=_CORE_MESSAGE_MAX_BYTES,
        storage=StorageType.FILE,
        num_replicas=1,
        duplicate_window=_CORE_DUPLICATE_WINDOW_SECONDS,
        no_ack=False,
        allow_direct=False,
    )


def _assert_subject_authority(info: StreamInfo, *, expected_name: str) -> None:
    actual_name = info.config.name
    if actual_name != expected_name:
        raise NatsStreamDriftError("core stream identity drift detected")
    actual = tuple(info.config.subjects or ())
    expected = _core_subjects()
    if len(actual) != len(expected) or frozenset(actual) != frozenset(expected):
        raise NatsStreamDriftError("core stream subject authority drift detected")
    config = info.config
    bounded = (
        config.retention == RetentionPolicy.LIMITS
        and config.storage == StorageType.FILE
        and config.max_consumers == 32
        and config.max_bytes == _CORE_STREAM_MAX_BYTES
        and config.max_age == _CORE_STREAM_MAX_AGE_SECONDS
        and config.max_msg_size == _CORE_MESSAGE_MAX_BYTES
        and config.num_replicas == 1
        and config.duplicate_window == _CORE_DUPLICATE_WINDOW_SECONDS
        and config.no_ack is False
        and config.allow_direct is False
    )
    if not bounded:
        raise NatsStreamDriftError("core stream durability bounds drift detected")


async def _ensure_core_stream(
    context: JetStreamContext,
    *,
    stream_name: str,
) -> None:
    try:
        info = await context.stream_info(stream_name)
    except NotFoundError:
        try:
            info = await context.add_stream(config=_stream_config(stream_name))
        except Exception as exc:
            raise NatsRuntimeError("canonical core stream creation failed") from exc
    except Exception as exc:
        raise NatsRuntimeError("canonical core stream inspection failed") from exc
    _assert_subject_authority(info, expected_name=stream_name)


def consumer_settings(settings: WorkerSettings) -> DurableConsumerSettings:
    """Build the shared explicit-ack, bounded-redelivery consumer contract."""

    return DurableConsumerSettings(
        stream=settings.stream_name,
        durable_name=settings.durable_name,
        ack_wait_seconds=settings.ack_wait_seconds,
        max_deliver=settings.max_deliver,
        max_ack_pending=64,
        retry_delay_seconds=0.25,
        max_payload_bytes=_CORE_MESSAGE_MAX_BYTES,
    )


@dataclass(slots=True)
class NatsRuntime:
    """A connected canonical event bus with owned graceful shutdown."""

    connection: NatsClient
    bus: JetStreamEventBus

    async def close(self) -> None:
        """Drain outstanding publications/subscriptions, then force-close on error."""

        if self.connection.is_closed:
            return
        try:
            await asyncio.wait_for(
                self.connection.drain(),
                timeout=_DRAIN_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            try:
                await self.connection.close()
            except Exception:
                pass
            raise NatsRuntimeError("local NATS shutdown failed") from exc


async def connect_nats_runtime(settings: WorkerSettings) -> NatsRuntime:
    """Connect to the admitted local NATS endpoint and verify stream authority."""

    try:
        connection = await nats.connect(
            servers=[settings.nats_url],
            name=settings.service_name,
            allow_reconnect=True,
            connect_timeout=2,
            reconnect_time_wait=0.25,
            max_reconnect_attempts=10,
        )
    except Exception as exc:
        raise NatsRuntimeError("local NATS connection failed") from exc

    context = connection.jetstream()
    try:
        await _ensure_core_stream(context, stream_name=settings.stream_name)
    except Exception:
        try:
            await connection.close()
        except Exception:
            pass
        raise
    event_context = cast(EventBusJetStreamContext, context)
    return NatsRuntime(connection=connection, bus=JetStreamEventBus(event_context))


async def wait_for_shutdown() -> None:
    """Wait for SIGINT/SIGTERM without blocking the worker event loop."""

    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    registered: list[signal.Signals] = []
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, event.set)
        except (NotImplementedError, RuntimeError):
            # Windows' default event loop does not implement signal handlers;
            # asyncio.run still turns Ctrl+C into task cancellation.
            continue
        registered.append(stop_signal)
    try:
        await event.wait()
    finally:
        for stop_signal in registered:
            loop.remove_signal_handler(stop_signal)
