"""Fail-closed paper portfolio reconciliation service."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from workers.reconciliation.projection import FillRow
from workers.reconciliation.service import ReconciliationService
from workers.reconciliation.service import run_periodic


NOW = datetime(2026, 7, 22, 8, 0, tzinfo=UTC)


class FakeRepository:
    def __init__(
        self,
        fills: tuple[FillRow, ...] = (),
        *,
        manual_switch_active: bool = False,
    ) -> None:
        self.fills = fills
        self.manual_switch_active = manual_switch_active
        self.persisted = []
        self.failures: list[str] = []

    async def load_fills(self, account_id: str) -> tuple[FillRow, ...]:
        del account_id
        return self.fills

    async def is_manual_kill_switch_active(self, account_id: str) -> bool:
        del account_id
        return self.manual_switch_active

    async def persist_success(self, state) -> None:
        self.persisted.append(state)

    async def persist_failure(self, *, account_id: str, reason: str) -> None:
        del account_id
        self.failures.append(reason)


class FakeQuotes:
    def __init__(self, marks: dict[str, Decimal] | None = None) -> None:
        self.marks = marks or {}

    async def midpoint(self, symbol: str) -> Decimal:
        return self.marks[symbol]


@pytest.mark.asyncio
async def test_clean_empty_ledger_persists_fresh_switch_off_snapshot() -> None:
    repository = FakeRepository()
    service = ReconciliationService(
        repository=repository,
        quotes=FakeQuotes(),
        initial_equity=Decimal("10000"),
    )

    state = await service.run_once(account_id="paper-main", reconciled_at=NOW)

    assert repository.persisted == [state]
    assert repository.failures == []
    assert state.kill_switch_active is False


@pytest.mark.asyncio
async def test_clean_reconciliation_never_clears_manual_kill_switch() -> None:
    repository = FakeRepository(manual_switch_active=True)
    service = ReconciliationService(
        repository=repository,
        quotes=FakeQuotes(),
        initial_equity=Decimal("10000"),
    )

    state = await service.run_once(account_id="paper-main", reconciled_at=NOW)

    assert state.kill_switch_active is True
    assert repository.persisted == [state]


@pytest.mark.asyncio
async def test_projection_failure_activates_switch_with_sanitized_reason() -> None:
    fill = FillRow(
        fill_id="fill-1",
        instrument="BTCUSDT",
        side="buy",
        quantity=Decimal("1"),
        price=Decimal("100"),
        fee=Decimal("0"),
        occurred_at=NOW,
    )
    repository = FakeRepository((fill,))
    service = ReconciliationService(
        repository=repository,
        quotes=FakeQuotes(),
        initial_equity=Decimal("10000"),
    )

    with pytest.raises(KeyError):
        await service.run_once(account_id="paper-main", reconciled_at=NOW)

    assert repository.persisted == []
    assert len(repository.failures) == 1
    assert repository.failures[0].startswith("KeyError:")
    assert "BTCUSDT" not in repository.failures[0]


@pytest.mark.asyncio
async def test_periodic_runner_propagates_terminal_runtime_failure() -> None:
    class TerminalRuntimeError(RuntimeError):
        pass

    class CountingService:
        calls = 0

        async def run_once(self, *, account_id: str, reconciled_at: datetime) -> None:
            del account_id, reconciled_at
            self.calls += 1

    async def runtime_wait() -> None:
        await asyncio.sleep(0)
        raise TerminalRuntimeError("terminal transport failure")

    service = CountingService()
    with pytest.raises(TerminalRuntimeError):
        await run_periodic(
            service,
            account_id="paper-main",
            runtime_wait=runtime_wait(),
            interval_seconds=0,
        )

    assert service.calls >= 1


@pytest.mark.asyncio
async def test_periodic_runner_retries_failed_cycle_and_reports_health() -> None:
    class TerminalRuntimeError(RuntimeError):
        pass

    class FlakyService:
        calls = 0

        async def run_once(self, *, account_id: str, reconciled_at: datetime) -> None:
            del account_id, reconciled_at
            self.calls += 1
            if self.calls == 1:
                raise ValueError("upstream quote unavailable")

    service = FlakyService()
    failures: list[str] = []
    successes: list[bool] = []

    async def runtime_wait() -> None:
        while service.calls < 2:
            await asyncio.sleep(0)
        raise TerminalRuntimeError("terminal transport failure")

    with pytest.raises(TerminalRuntimeError):
        await run_periodic(
            service,
            account_id="paper-main",
            runtime_wait=runtime_wait(),
            interval_seconds=0,
            on_cycle_failure=lambda exc: failures.append(type(exc).__name__),
            on_cycle_success=lambda: successes.append(True),
        )

    assert service.calls >= 2
    assert failures == ["ValueError"]
    assert successes
