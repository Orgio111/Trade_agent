"""NATS runtime wiring for the canonical deterministic decision worker."""

from __future__ import annotations

from typing import Any, Protocol

import asyncpg  # type: ignore[import-untyped]

from packages.event_bus import (
    CoreSubject,
    DurableConsumerSettings,
    PublishReceipt,
)
from packages.execution import PostgresDecisionStore
from packages.risk import PostgresRiskInputRepository
from workers.config import WorkerSettings
from workers.contracts import CandidateForRiskEvent, RiskDecisionEvent
from workers.nats_runtime import (
    NatsRuntime,
    connect_nats_runtime,
    consumer_settings,
    wait_for_shutdown,
)

from .async_service import AsyncRiskDecisionService


class RuntimeModeMismatch(RuntimeError):
    """A candidate came from a mode outside this worker's isolated runtime."""


class DecisionEventBus(Protocol):
    """Narrow JetStream surface used by the decision worker."""

    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> PublishReceipt: ...

    async def subscribe(
        self,
        subject: CoreSubject,
        handler: Any,
        *,
        settings: DurableConsumerSettings,
        on_failure: Any = None,
    ) -> Any: ...


class DecisionRuntime:
    """Consume candidates and publish only their persisted hard-rule verdicts."""

    def __init__(
        self,
        *,
        settings: WorkerSettings,
        service: AsyncRiskDecisionService,
        bus: DecisionEventBus,
    ) -> None:
        self._settings = settings
        self._service = service
        self._bus = bus

    async def handle(self, payload: bytes) -> RiskDecisionEvent:
        """Validate, authorize, persist, and durably publish one candidate."""

        candidate_event = CandidateForRiskEvent.model_validate_json(payload)
        if candidate_event.candidate.source_mode is not self._settings.mode:
            raise RuntimeModeMismatch(
                "candidate source mode does not match isolated worker mode"
            )
        verdict = await self._service.evaluate(candidate_event)
        subject = (
            CoreSubject.RISK_APPROVED
            if verdict.decision.approved
            else CoreSubject.RISK_REJECTED
        )
        await self._bus.publish(
            subject,
            verdict.canonical_json().encode("utf-8"),
            message_id=str(verdict.event_id),
            stream=self._settings.stream_name,
        )
        return verdict

    async def start(self) -> Any:
        """Install the single explicit-ack durable candidate consumer."""

        return await self._bus.subscribe(
            CoreSubject.SIGNAL_CANDIDATE,
            self.handle,
            settings=consumer_settings(self._settings),
        )


async def run_decision_worker(settings: WorkerSettings | None = None) -> None:
    """Run the local worker until shutdown and close every owned resource."""

    configured = settings or WorkerSettings.from_env(
        service_name="decision-worker",
        durable_name="decision-worker-v1",
    )
    pool = await asyncpg.create_pool(
        dsn=configured.database_url.get_secret_value(),
        min_size=1,
        max_size=4,
        command_timeout=5,
    )
    nats_runtime: NatsRuntime | None = None
    try:
        nats_runtime = await connect_nats_runtime(configured)
        service = AsyncRiskDecisionService(
            risk_inputs=PostgresRiskInputRepository(pool),
            decision_authority=PostgresDecisionStore(pool),
            account_id=configured.account_id,
        )
        worker = DecisionRuntime(
            settings=configured,
            service=service,
            bus=nats_runtime.bus,
        )
        await worker.start()
        await wait_for_shutdown()
    finally:
        try:
            if nats_runtime is not None:
                await nats_runtime.close()
        finally:
            await pool.close()


__all__ = [
    "DecisionRuntime",
    "RuntimeModeMismatch",
    "run_decision_worker",
]
