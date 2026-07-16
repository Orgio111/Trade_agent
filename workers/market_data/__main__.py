"""Process entry point for the canonical local market-data worker."""

from __future__ import annotations

import asyncio
import logging

from workers.config import WorkerSettings
from workers.market_data.runtime import MarketDataRuntime
from workers.nats_runtime import connect_nats_runtime, wait_for_shutdown


_LOGGER = logging.getLogger(__name__)


async def run() -> None:
    settings = WorkerSettings.from_env(
        service_name="market-data-worker",
        durable_name="market-data-worker-v1",
    )
    nats_runtime = await connect_nats_runtime(settings)
    try:
        worker = MarketDataRuntime(settings, nats_runtime.bus)
        await worker.start()
        _LOGGER.info("canonical worker ready: %s", settings.public_summary())
        await wait_for_shutdown()
    finally:
        await nats_runtime.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
