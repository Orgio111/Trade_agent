"""Process entry point for local typed-candidate worker."""

from __future__ import annotations

import asyncio
from contextlib import suppress

import asyncpg  # type: ignore[import-untyped]

from workers.config import WorkerSettings
from workers.durability import PostgresInboxOutbox
from workers.nats_runtime import PostgresWorkerLeaseStore, connect_nats_runtime

from .market import BinanceBookTickerClient
from .ollama import OllamaCandidateClient
from .runtime import TypedCandidateProducer
from .service import CandidateRuntime


async def run() -> None:
    settings = WorkerSettings.from_env(
        service_name="candidate-worker", durable_name="candidate-worker-v1"
    )
    pool = await asyncpg.create_pool(
        dsn=settings.database_url.get_secret_value(), min_size=1, max_size=3, command_timeout=5
    )
    nats_runtime = await connect_nats_runtime(
        settings, lease_store=PostgresWorkerLeaseStore(pool)
    )
    ollama = OllamaCandidateClient(base_url="http://host.docker.internal:11434")
    market = BinanceBookTickerClient()
    dispatcher: asyncio.Task[None] | None = None
    runtime_wait: asyncio.Task[None] | None = None
    try:
        durability = PostgresInboxOutbox(
            pool,
            nats_runtime.bus,
            stream_name=settings.stream_name,
            durable_name=settings.durable_name,
        )
        producer = TypedCandidateProducer(
            ollama, model_digest=await ollama.model_digest()
        )
        worker = CandidateRuntime(
            settings, nats_runtime.bus, producer, market, durability
        )
        await worker.start()
        dispatcher = asyncio.create_task(durability.run_dispatcher())
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
        await ollama.aclose()
        await market.aclose()
        try:
            await nats_runtime.close()
        finally:
            await pool.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
