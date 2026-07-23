"""Process entry point for durable feature worker."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import asyncpg  # type: ignore[import-untyped]

from workers.config import WorkerSettings
from workers.durability import PostgresInboxOutbox
from workers.nats_runtime import PostgresWorkerLeaseStore, connect_nats_runtime

from .postgres import PostgresFeatureRepository
from .service import FeatureRuntime


async def run() -> None:
    settings = WorkerSettings.from_env(
        service_name="feature-worker", durable_name="feature-worker-v1"
    )
    pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(), min_size=1, max_size=3, command_timeout=5
    )
    nats_runtime = await connect_nats_runtime(
        settings, lease_store=PostgresWorkerLeaseStore(pool)
    )
    dispatcher: asyncio.Task[None] | None = None
    runtime_wait: asyncio.Task[None] | None = None
    try:
        repository = PostgresFeatureRepository(
            pool, stream_name=settings.stream_name, durable_name=settings.durable_name
        )
        worker = FeatureRuntime(settings, nats_runtime.bus, repository)
        await worker.start()
        outbox = PostgresInboxOutbox(
            pool,
            nats_runtime.bus,
            stream_name=settings.stream_name,
            durable_name=settings.durable_name,
        )
        dispatcher = asyncio.create_task(outbox.run_dispatcher())
        runtime_wait = asyncio.create_task(nats_runtime.wait())
        done, _ = await asyncio.wait(
            {dispatcher, runtime_wait}, return_when=asyncio.FIRST_COMPLETED
        )
        for completed in done:
            completed.result()
    finally:
        for task in (dispatcher, runtime_wait):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        try:
            await nats_runtime.close()
        finally:
            await pool.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
