"""Periodic fail-closed paper portfolio reconciliation process."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from decimal import Decimal

import asyncpg  # type: ignore[import-untyped]

from workers.candidate.market import BinanceBookTickerClient
from workers.config import WorkerSettings
from workers.nats_runtime import PostgresWorkerLeaseStore, connect_nats_runtime
from workers.reconciliation.authority import RiskAuthorityBootstrap
from workers.reconciliation.postgres import PostgresReconciliationRepository
from workers.reconciliation.service import ReconciliationService, run_periodic


class BinanceMidpointSource:
    def __init__(self, client: BinanceBookTickerClient) -> None:
        self._client = client

    async def midpoint(self, symbol: str) -> Decimal:
        quote = await self._client.fetch(symbol)
        return (quote.bid + quote.ask) / Decimal("2")


async def run() -> None:
    settings = WorkerSettings.from_env(
        service_name="reconciliation-worker",
        durable_name="reconciliation-worker-v1",
    )
    pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(), min_size=1, max_size=2
    )
    if pool is None:
        raise RuntimeError("PostgreSQL pool creation returned no pool")
    quote_client = BinanceBookTickerClient()
    authority = RiskAuthorityBootstrap(pool)
    nats_runtime = None
    try:
        nats_runtime = await connect_nats_runtime(
            settings,
            lease_store=PostgresWorkerLeaseStore(pool),
            monitor_consumer=False,
        )
        await authority.ensure()
        service = ReconciliationService(
            repository=PostgresReconciliationRepository(pool),
            quotes=BinanceMidpointSource(quote_client),
            # ponytail: one canonical paper account; add persisted funding ledger
            # before supporting deposits, withdrawals, or multiple account balances.
            initial_equity=Decimal("10000"),
        )

        def mark_cycle_failed(exc: Exception) -> None:
            nats_runtime.consumer_healthy = False
            nats_runtime.last_error = (
                f"reconciliation cycle failed: {type(exc).__name__}"
            )

        def mark_cycle_healthy() -> None:
            nats_runtime.consumer_healthy = True
            nats_runtime.last_error = None

        await run_periodic(
            service,
            account_id=settings.account_id,
            runtime_wait=nats_runtime.wait(),
            on_cycle_failure=mark_cycle_failed,
            on_cycle_success=mark_cycle_healthy,
        )
    finally:
        if nats_runtime is not None:
            await nats_runtime.close()
        await quote_client.aclose()
        await authority.aclose()
        await pool.close()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(run())


if __name__ == "__main__":
    main()
