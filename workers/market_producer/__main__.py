"""Process entry point for Binance public market producer."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging

import asyncpg  # type: ignore[import-untyped]

from workers.config import WorkerSettings
from workers.market_producer.runtime import BinanceMarketProducer
from workers.nats_runtime import PostgresWorkerLeaseStore, connect_nats_runtime


async def run() -> None:
    settings = WorkerSettings.from_env(
        service_name="market-producer",
        durable_name="market-producer-v1",
    )
    pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(),
        min_size=1,
        max_size=2,
        command_timeout=5,
    )
    nats_runtime = await connect_nats_runtime(
        settings,
        lease_store=PostgresWorkerLeaseStore(pool),
        monitor_consumer=False,
    )
    producer_task: asyncio.Task[None] | None = None
    runtime_wait: asyncio.Task[None] | None = None
    try:
        producer = BinanceMarketProducer(
            nats_runtime.bus,
            stream_name=settings.stream_name,
        )
        producer_task = asyncio.create_task(producer.run_forever())
        runtime_wait = asyncio.create_task(nats_runtime.wait())
        done, _ = await asyncio.wait(
            {producer_task, runtime_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        for completed in done:
            completed.result()
    finally:
        for owned_task in (producer_task, runtime_wait):
            if owned_task is not None:
                owned_task.cancel()
                with suppress(asyncio.CancelledError):
                    await owned_task
        try:
            await nats_runtime.close()
        finally:
            await pool.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
