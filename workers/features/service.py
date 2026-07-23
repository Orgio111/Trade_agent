"""Durable validated-market to feature-snapshot worker."""

from __future__ import annotations

from typing import Any, Protocol

from packages.domain import MarketEvent
from packages.event_bus import CoreSubject
from workers.config import WorkerSettings
from workers.nats_runtime import consumer_settings

from .runtime import FeatureSnapshot


class FeatureRepository(Protocol):
    async def process(self, event: MarketEvent) -> FeatureSnapshot: ...


class FeatureBus(Protocol):
    async def subscribe(
        self,
        subject: CoreSubject,
        handler: Any,
        *,
        settings: Any,
        on_failure: Any = None,
    ) -> Any: ...


class FeatureRuntime:
    def __init__(
        self,
        settings: WorkerSettings,
        bus: FeatureBus,
        repository: FeatureRepository,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._repository = repository

    async def handle(self, payload: bytes) -> FeatureSnapshot:
        event = MarketEvent.model_validate_json(payload)
        if event.source_mode is not self._settings.mode:
            raise ValueError("market event mode does not match feature worker mode")
        return await self._repository.process(event)

    async def start(self) -> Any:
        return await self._bus.subscribe(
            CoreSubject.MARKET_VALIDATED,
            self.handle,
            settings=consumer_settings(self._settings),
        )
