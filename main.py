"""
Entry point — boots uvloop, starts the market data feed, supervisor loop,
and the web monitoring dashboard.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading

try:
    import uvloop  # type: ignore[import]
    _HAVE_UVLOOP = True
except ImportError:
    _HAVE_UVLOOP = False

from prometheus_client import start_http_server

from agents.supervisor import PortfolioSupervisor
from core.config import get_settings
from core.market_data import MarketDataFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")


async def _run_supervisor_loop(
    supervisor: PortfolioSupervisor,
    feed: MarketDataFeed,
    cfg,
) -> None:
    logger.info("Supervisor loop started for symbols: %s", cfg.symbols)
    while True:
        for symbol in cfg.symbols:
            tick = feed.latest_tick(symbol)
            if tick is None:
                continue
            closes = feed.get_closes(symbol).tolist()
            highs = feed.get_highs(symbol).tolist()
            lows = feed.get_lows(symbol).tolist()
            volumes = feed.get_volumes(symbol).tolist()
            if len(closes) < 50:
                continue  # not enough warmup data
            try:
                await supervisor.run_cycle(symbol, closes, highs, lows, volumes)
            except Exception as exc:
                logger.error("Supervisor cycle error for %s: %s", symbol, exc, exc_info=True)
        await asyncio.sleep(60)  # 1-minute cadence


# ── Web dashboard server thread ──────────────────────────────────────────────
_dashboard_thread: threading.Thread | None = None


def _start_dashboard() -> None:
    """Start the FastAPI dashboard server in a daemon thread."""
    try:
        import uvicorn  # type: ignore[import]
        logger.info("Starting web dashboard on http://0.0.0.0:3000")
        uvicorn.run(
            "web.server:app",
            host="0.0.0.0",
            port=3000,
            log_level="info",
            reload=False,
        )
    except ImportError:
        logger.warning("uvicorn not installed — dashboard disabled. pip install uvicorn fastapi")
    except Exception as exc:
        logger.error("Dashboard server error: %s", exc)


async def main() -> None:
    cfg = get_settings()

    # Start Prometheus metrics server
    start_http_server(cfg.prometheus_port)
    logger.info("Prometheus metrics on :%d/metrics", cfg.prometheus_port)

    # Start web dashboard in a daemon thread
    global _dashboard_thread
    _dashboard_thread = threading.Thread(target=_start_dashboard, daemon=True)
    _dashboard_thread.start()

    feed = MarketDataFeed()
    supervisor = PortfolioSupervisor(initial_equity=cfg.initial_capital)
    await supervisor.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(sig: signal.Signals) -> None:
        logger.info("Received %s — shutting down", sig.name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    feed_task = asyncio.create_task(feed.start(), name="market_feed")
    supervisor_task = asyncio.create_task(
        _run_supervisor_loop(supervisor, feed, cfg), name="supervisor"
    )

    await stop_event.wait()
    feed_task.cancel()
    supervisor_task.cancel()
    await feed.stop()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    if _HAVE_UVLOOP:
        uvloop.install()  # type: ignore[union-attr]
    else:
        logger.info("uvloop not available — using asyncio default event loop")
    asyncio.run(main())
