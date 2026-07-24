"""Shared local NATS JetStream lifecycle for canonical worker processes.

The runtime creates one bounded file-backed stream when it is absent and
refuses to start when an existing stream's exact subject authority has drifted.
It never reads credentials from dotenv files and transports canonical bytes
only; domain authorization remains in the worker services.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import signal
from typing import Any, Protocol, cast

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
_CONSUMER_MONITOR_TIMEOUT_SECONDS = 2.0


class NatsRuntimeError(RuntimeError):
    """The bounded local NATS runtime could not be started or stopped safely."""


class NatsStreamDriftError(NatsRuntimeError):
    """The existing core stream does not own the exact canonical subjects."""


class NatsLifecycleState(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FATAL = "fatal"
    STOPPING = "stopping"
    STOPPED = "stopped"


class WorkerLeaseStore(Protocol):
    async def heartbeat(self, **record: object) -> None: ...

    async def close(self) -> None: ...


class PostgresWorkerLeaseStore:
    """Small pool-backed lease projection; transport health owns its expiry."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def heartbeat(self, **record: object) -> None:
        await self._pool.execute(
            """
            INSERT INTO worker_leases (
                worker_id, service_name, instance_id, started_at, last_heartbeat,
                lease_expires_at, status, last_error, build_version,
                consumer_name, consumer_lag
            ) VALUES ($1, $2, $3, $4, NOW(), NOW() + $5 * INTERVAL '1 second',
                      $6, $7, $8, $9, $10)
            ON CONFLICT (worker_id) DO UPDATE SET
                last_heartbeat = NOW(),
                lease_expires_at = NOW() + $5 * INTERVAL '1 second',
                status = EXCLUDED.status,
                last_error = EXCLUDED.last_error,
                consumer_lag = EXCLUDED.consumer_lag
            """,
            record["worker_id"], record["service_name"], record["instance_id"],
            record["started_at"], record["lease_seconds"], record["status"],
            record.get("last_error"), record["build_version"],
            record["consumer_name"], record.get("consumer_lag"),
        )

    async def close(self) -> None:
        return None


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
    settings: WorkerSettings
    context: JetStreamContext
    lease_store: WorkerLeaseStore | None = None
    state: NatsLifecycleState = NatsLifecycleState.CONNECTED
    consumer_lag: int | None = None
    consumer_healthy: bool = True
    monitor_consumer: bool = True
    last_error: str | None = None
    _fatal_event: asyncio.Event | None = None
    _closing: bool = False
    _started_at: datetime | None = None

    @property
    def ready(self) -> bool:
        lag_ok = self.consumer_lag is None or self.consumer_lag <= self.settings.max_consumer_lag
        return (
            self.state is NatsLifecycleState.CONNECTED
            and self.consumer_healthy
            and lag_ok
        )

    async def wait(self) -> None:
        """Wait for operator shutdown or fail when NATS closes terminally."""

        fatal = self._fatal_event or asyncio.Event()
        self._fatal_event = fatal
        shutdown = asyncio.create_task(wait_for_shutdown())
        failed = asyncio.create_task(fatal.wait())
        heartbeat = asyncio.create_task(self._heartbeat_loop())
        try:
            done, _ = await asyncio.wait(
                {shutdown, failed}, return_when=asyncio.FIRST_COMPLETED
            )
            if failed in done:
                raise NatsRuntimeError("local NATS connection terminally closed")
        finally:
            shutdown.cancel()
            failed.cancel()
            heartbeat.cancel()

    async def heartbeat_once(self) -> None:
        if self.monitor_consumer:
            try:
                info = await asyncio.wait_for(
                    self.context.consumer_info(
                        self.settings.stream_name, self.settings.durable_name
                    ),
                    timeout=_CONSUMER_MONITOR_TIMEOUT_SECONDS,
                )
                self.consumer_lag = int(info.num_pending) + int(info.num_ack_pending)
                self.consumer_healthy = True
                if (
                    self.last_error is not None
                    and self.last_error.startswith("consumer monitor failed:")
                ):
                    self.last_error = None
            except Exception as exc:
                self.consumer_lag = None
                self.consumer_healthy = False
                if self.state is NatsLifecycleState.CONNECTED:
                    self.last_error = f"consumer monitor failed: {type(exc).__name__}"
        if self.lease_store is None:
            return
        status = "ready" if self.ready else self.state.value
        await self.lease_store.heartbeat(
            worker_id=f"{self.settings.service_name}:{self.settings.durable_name}",
            service_name=self.settings.service_name,
            instance_id=self.settings.durable_name,
            started_at=self._started_at or datetime.now(UTC),
            lease_seconds=self.settings.lease_ttl_seconds if self.ready else 0,
            status=status,
            last_error=self.last_error,
            build_version="unknown",
            consumer_name=self.settings.durable_name,
            consumer_lag=self.consumer_lag,
        )

    async def _heartbeat_loop(self) -> None:
        while True:
            if (
                self.state is NatsLifecycleState.CONNECTED
                and self.last_error is not None
                and self.last_error.startswith("lease heartbeat failed:")
            ):
                # A failed database write expires the persisted lease. Let the
                # next successful heartbeat restore it instead of keeping the
                # in-memory worker unhealthy forever.
                self.consumer_healthy = True
                self.last_error = None
            try:
                await self.heartbeat_once()
            except Exception as exc:
                self.consumer_healthy = False
                self.last_error = f"lease heartbeat failed: {type(exc).__name__}"
            await asyncio.sleep(self.settings.heartbeat_interval_seconds)

    async def close(self) -> None:
        """Drain outstanding publications/subscriptions, then force-close on error."""

        if self.connection.is_closed:
            return
        self._closing = True
        self.state = NatsLifecycleState.STOPPING
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
        finally:
            self.state = NatsLifecycleState.STOPPED
            if self.lease_store is not None:
                await self.lease_store.close()


async def connect_nats_runtime(
    settings: WorkerSettings,
    *,
    lease_store: WorkerLeaseStore | None = None,
    monitor_consumer: bool = True,
) -> NatsRuntime:
    """Connect to the admitted local NATS endpoint and verify stream authority."""

    holder: dict[str, NatsRuntime] = {}

    async def disconnected() -> None:
        runtime = holder.get("runtime")
        if runtime is not None and not runtime._closing:
            runtime.state = NatsLifecycleState.DISCONNECTED
            runtime.last_error = "NATS disconnected"

    async def reconnected() -> None:
        runtime = holder.get("runtime")
        if runtime is not None:
            runtime.state = NatsLifecycleState.CONNECTED
            runtime.last_error = None

    async def closed() -> None:
        runtime = holder.get("runtime")
        if runtime is not None and not runtime._closing:
            runtime.state = NatsLifecycleState.FATAL
            runtime.last_error = "NATS terminally closed"
            if runtime._fatal_event is None:
                runtime._fatal_event = asyncio.Event()
            runtime._fatal_event.set()

    try:
        connection = await nats.connect(
            servers=[settings.nats_url],
            name=settings.service_name,
            allow_reconnect=True,
            connect_timeout=2,
            reconnect_time_wait=0.25,
            max_reconnect_attempts=-1,
            disconnected_cb=disconnected,
            reconnected_cb=reconnected,
            closed_cb=closed,
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
    runtime = NatsRuntime(
        connection=connection,
        bus=JetStreamEventBus(event_context),
        settings=settings,
        context=context,
        lease_store=lease_store,
        monitor_consumer=monitor_consumer,
        _fatal_event=asyncio.Event(),
        _started_at=datetime.now(UTC),
    )
    holder["runtime"] = runtime
    return runtime


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
