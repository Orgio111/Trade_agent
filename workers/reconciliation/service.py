"""Fail-closed periodic paper portfolio reconciliation."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import logging
from typing import Awaitable, Callable, Protocol, Sequence

from packages.risk import PortfolioState
from workers.reconciliation.projection import FillRow, PaperPortfolioProjector


logger = logging.getLogger(__name__)


class ReconciliationRepository(Protocol):
    async def load_fills(self, account_id: str) -> Sequence[FillRow]: ...

    async def is_manual_kill_switch_active(self, account_id: str) -> bool: ...

    async def persist_success(self, state: PortfolioState) -> None: ...

    async def persist_failure(self, *, account_id: str, reason: str) -> None: ...


class QuoteSource(Protocol):
    async def midpoint(self, symbol: str) -> Decimal: ...


class ReconciliationService:
    def __init__(
        self,
        *,
        repository: ReconciliationRepository,
        quotes: QuoteSource,
        initial_equity: Decimal,
    ) -> None:
        self._repository = repository
        self._quotes = quotes
        self._projector = PaperPortfolioProjector(initial_equity)

    async def run_once(
        self, *, account_id: str, reconciled_at: datetime
    ) -> PortfolioState:
        try:
            fills = tuple(await self._repository.load_fills(account_id))
            symbols = sorted({fill.instrument.upper() for fill in fills})
            marks = {
                symbol: await self._quotes.midpoint(symbol) for symbol in symbols
            }
            state = self._projector.project(
                account_id=account_id,
                fills=fills,
                marks=marks,
                reconciled_at=reconciled_at,
            )
            if await self._repository.is_manual_kill_switch_active(account_id):
                state = state.model_copy(update={"kill_switch_active": True})
            await self._repository.persist_success(state)
            return state
        except Exception as exc:
            fingerprint = hashlib.sha256(type(exc).__name__.encode()).hexdigest()[:16]
            await self._repository.persist_failure(
                account_id=account_id,
                reason=f"{type(exc).__name__}:{fingerprint}",
            )
            raise


async def run_periodic(
    service: ReconciliationService,
    *,
    account_id: str,
    runtime_wait: Awaitable[None],
    interval_seconds: float = 5,
    on_cycle_failure: Callable[[Exception], None] | None = None,
    on_cycle_success: Callable[[], None] | None = None,
) -> None:
    """Retry reconciliation cycles while propagating terminal transport exits."""

    async def reconcile() -> None:
        while True:
            try:
                await service.run_once(
                    account_id=account_id,
                    reconciled_at=datetime.now(UTC),
                )
            except Exception as exc:
                if on_cycle_failure is not None:
                    on_cycle_failure(exc)
                logger.error(
                    "reconciliation cycle failed: %s",
                    type(exc).__name__,
                )
            else:
                if on_cycle_success is not None:
                    on_cycle_success()
            await asyncio.sleep(interval_seconds)

    reconcile_task = asyncio.create_task(reconcile())
    runtime_task = asyncio.ensure_future(runtime_wait)
    try:
        done, _ = await asyncio.wait(
            {reconcile_task, runtime_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for completed in done:
            completed.result()
    finally:
        for task in (reconcile_task, runtime_task):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
