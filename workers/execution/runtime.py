"""JetStream boundary for deterministic, paper-only execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

import asyncpg

from packages.event_bus import CoreSubject, JetStreamEventBus
from packages.execution import PostgresDecisionStore, PostgresExecutionLedger
from workers.config import WorkerSettings
from workers.contracts import (
    OrderIntentEvent,
    OrderUpdatedEvent,
    RiskDecisionEvent,
)
from workers.nats_runtime import (
    connect_nats_runtime,
    consumer_settings,
    wait_for_shutdown,
)

from .async_service import AsyncPaperExecutionService


class ExecutionRuntimeError(RuntimeError):
    """The execution worker rejected an event or lost a required dependency."""


class RuntimeModeMismatch(ExecutionRuntimeError):
    """An approved event attempted to cross its configured source-mode boundary."""


class ExecutionService(Protocol):
    async def submit(
        self,
        event: RiskDecisionEvent,
        *,
        execution_at: datetime,
    ) -> object: ...


class ExecutionRuntime:
    """Consume exact approvals and publish idempotent paper ledger snapshots."""

    def __init__(
        self,
        *,
        settings: WorkerSettings,
        bus: JetStreamEventBus,
        service: AsyncPaperExecutionService,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._service = service
        self._clock = clock or (lambda: datetime.now(UTC))

    async def start(self) -> object:
        """Start one bounded, manual-ack durable approval consumer."""

        return await self._bus.subscribe(
            CoreSubject.RISK_APPROVED,
            self.handle,
            settings=consumer_settings(self._settings),
        )

    async def handle(self, payload: bytes) -> None:
        """Validate, execute on PaperBroker, then publish deterministic outputs."""

        try:
            event = RiskDecisionEvent.model_validate_json(payload)
        except Exception as exc:
            raise ExecutionRuntimeError("risk approval payload is invalid") from exc
        if event.candidate.source_mode is not self._settings.mode:
            raise RuntimeModeMismatch(
                "risk approval source mode differs from execution runtime mode"
            )
        if event.decision.account_id != self._settings.account_id:
            raise ExecutionRuntimeError(
                "risk approval account differs from execution runtime account"
            )
        if not event.decision.approved:
            raise ExecutionRuntimeError(
                "approved subject cannot carry a rejected risk decision"
            )

        result = await self._service.submit(event, execution_at=self._clock())
        intent_event = OrderIntentEvent.create(event, result.intent)
        updated_event = OrderUpdatedEvent.create(
            event,
            intent_event,
            result,
            created_at=result.order.updated_at,
        )
        await self._bus.publish(
            CoreSubject.ORDER_INTENT,
            intent_event.canonical_json().encode("utf-8"),
            message_id=f"intent:{intent_event.event_id}",
            stream=self._settings.stream_name,
        )
        await self._bus.publish(
            CoreSubject.ORDER_UPDATED,
            updated_event.canonical_json().encode("utf-8"),
            message_id=f"order:{updated_event.event_id}",
            stream=self._settings.stream_name,
        )


async def run() -> None:
    """Build local durable adapters and run until the process is stopped."""

    settings = WorkerSettings.from_env(
        service_name="execution-worker",
        durable_name="execution-paper-v1",
    )
    pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(),
        min_size=1,
        max_size=4,
        command_timeout=5,
    )
    if pool is None:  # pragma: no cover - defensive asyncpg typing boundary
        raise ExecutionRuntimeError("PostgreSQL pool creation returned no pool")
    nats_runtime = None
    try:
        nats_runtime = await connect_nats_runtime(settings)
        worker = ExecutionRuntime(
            settings=settings,
            bus=nats_runtime.bus,
            service=AsyncPaperExecutionService(
                PostgresDecisionStore(pool),
                PostgresExecutionLedger(pool),
                account_id=settings.account_id,
            ),
        )
        await worker.start()
        await wait_for_shutdown()
    finally:
        if nats_runtime is not None:
            await nats_runtime.close()
        await pool.close()


def main() -> None:
    """Synchronous process entry point."""

    asyncio.run(run())


__all__ = [
    "ExecutionRuntime",
    "ExecutionRuntimeError",
    "RuntimeModeMismatch",
    "main",
    "run",
]
