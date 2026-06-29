"""QUANTEX Trinity Architecture — Python Layer A Entry Point.

Launches all 11 AI Brains in parallel, each publishing signals to
NATS JetStream subject `signals.raw` for the Go Orchestrator (Layer B)
to aggregate and forward to the Rust Execution Engine (Layer C).

Usage:
    python main.py                      # Run with default .env config
    python main.py --once               # Single cycle then exit
    python main.py --interval 60         # Custom cycle interval (seconds)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

import nats
import orjson
from dotenv import load_dotenv

from notifications.telegram_notifier import TelegramNotifier

# ── Load .env before anything else ──────────────────────────────

ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    logging.warning("No .env file found at %s", ENV_PATH)

# ── Logging Setup ──────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
log = logging.getLogger("quantex.brains")

# ── Configuration ──────────────────────────────────────────────

NATS_URL = os.getenv("NATS_URL", "nats://localhost:4222")
NATS_SUBJECT_RAW = os.getenv("NATS_SUBJECT_RAW", "signals.raw")
TRADE_SYMBOLS = [s.strip() for s in os.getenv("TRADE_SYMBOLS", "BTC/USDT").split(",") if s.strip()]
CYCLE_INTERVAL = int(os.getenv("BRAIN_CYCLE_INTERVAL", "60"))


# ── NATS JetStream Stream Setup ────────────────────────────────

async def ensure_jetstream_stream(nc: nats.NATS) -> None:
    """Create the `signals` JetStream stream if it doesn't exist.

    This is called once at startup so the Go Orchestrator and
    Rust Execution Engine can subscribe immediately.
    """
    js = nc.jetstream()

    try:
        await js.add_stream(
            name="signals",
            subjects=["signals.raw", "signals.aggregated", "signals.executed"],
            retention="limits",
            max_msgs=10000,
            max_age=86400,  # 24h
        )
        log.info("JetStream stream 'signals' created with 3 subjects")
    except nats.js.errors.BadRequestError:
        log.info("JetStream stream 'signals' already exists")
    except Exception as e:
        log.warning("JetStream stream creation failed (may already exist): %s", e)


# ── Brain Registry ─────────────────────────────────────────────

def create_all_brains() -> list:
    """Instantiate all 11 AI brains from the brains package."""
    from orchestrator.brains import (
        TimesFMBrain,
        FreqAIBrain,
        LLMRegimeBrain,
        MicrostructureBrain,
        FinBERTBrain,
        FinRLBrain,
        OnChainBrain,
        StatArbBrain,
        OrderFlowNautilusBrain,
        PolymarketBrain,
        CustomNNBrain,
    )

    return [
        TimesFMBrain(),
        FreqAIBrain(),
        LLMRegimeBrain(),
        MicrostructureBrain(),
        FinBERTBrain(),
        FinRLBrain(),
        OnChainBrain(),
        StatArbBrain(),
        OrderFlowNautilusBrain(),
        PolymarketBrain(),
        CustomNNBrain(),
    ]


# ── Main Loop ──────────────────────────────────────────────────

async def run_brains_once(nc: nats.NATS, brains: list, tg: TelegramNotifier) -> None:
    """Run all 11 brains once in parallel and publish signals to NATS."""

    log.info("Brain cycle started — running %d brains for %d symbols", len(brains), len(TRADE_SYMBOLS))

    for symbol in TRADE_SYMBOLS:
        # Run all brains concurrently for this symbol
        tasks = [brain.compute_score(symbol) for brain in brains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for brain, result in zip(brains, results):
            if isinstance(result, Exception):
                log.error("Brain %s failed for %s: %s", brain.brain_id, symbol, result)
                continue

            # Publish raw signal to NATS
            signal_data = {
                "brain_name": result.brain_id,
                "symbol": result.symbol,
                "score": result.score,
                "confidence": result.confidence,
                "weight": result.weight,
                "direction": result.direction,
                "metadata": result.metadata,
                "timestamp_ms": int(asyncio.get_event_loop().time() * 1000),
            }

            payload = orjson.dumps(signal_data)

            try:
                await nc.publish(
                    NATS_SUBJECT_RAW,
                    payload,
                )
                log.debug(
                    "Published: %s %s score=%.4f dir=%s",
                    result.brain_id, result.symbol, result.score, result.direction,
                )
                # Notify Telegram for strong signals (|score| > 0.6)
                if abs(result.score) > 0.6:
                    direction_str = "BUY" if result.direction > 0 else "SELL"
                    if direction_str == "BUY":
                        tg.signal_buy(
                            symbol=result.symbol,
                            entry_price=0.0,
                            score=result.score,
                            confidence=result.confidence,
                            stop_loss=0.0,
                            take_profit=0.0,
                        )
                    else:
                        tg.signal_sell(
                            symbol=result.symbol,
                            entry_price=0.0,
                            score=result.score,
                            confidence=result.confidence,
                            stop_loss=0.0,
                            take_profit=0.0,
                        )
            except Exception as e:
                log.error("NATS publish failed for %s: %s", result.brain_id, e)

    log.info("Brain cycle complete — %d symbols processed", len(TRADE_SYMBOLS))


async def main_loop(run_once: bool = False, interval: int = CYCLE_INTERVAL) -> None:
    """Main event loop — connect to NATS, run brain cycles."""

    log.info("Connecting to NATS at %s", NATS_URL)

    try:
        nc = await nats.connect(NATS_URL)
    except Exception as e:
        log.error("Failed to connect to NATS: %s", e)
        log.error("Make sure NATS server is running (docker start quantex-nats)")
        sys.exit(1)

    log.info("Connected to NATS successfully")

    # Ensure JetStream stream exists
    await ensure_jetstream_stream(nc)

    # Start Telegram notifier
    tg = TelegramNotifier()
    await tg.start()
    tg.system_status(f"QUANTEX Brain Layer started — {len(TRADE_SYMBOLS)} symbol(s)")

    # Create all brains
    brains = create_all_brains()
    log.info(
        "Initialized %d brains: %s",
        len(brains),
        ", ".join(b.brain_id for b in brains),
    )

    # Shutdown event
    shutdown_event = asyncio.Event()

    def _signal_handler():
        log.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Main cycle loop
    cycle_count = 0
    while not shutdown_event.is_set():
        cycle_count += 1
        log.info("=== Brain Cycle #%d ===", cycle_count)

        try:
            await run_brains_once(nc, brains, tg)
        except Exception as e:
            log.error("Brain cycle error: %s", e)

        if run_once:
            break

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass  # Normal — next cycle

    # Cleanup
    log.info("Draining NATS connection...")
    tg.system_status("QUANTEX Brain Layer shutting down")
    await tg.stop()
    await nc.drain()
    log.info("QUANTEX Brain Layer shutdown complete")


# ── CLI ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QUANTEX Trinity — Brain Layer (Python)")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--interval", type=int, default=CYCLE_INTERVAL,
                        help="Cycle interval in seconds (default: %(default)s)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("QUANTEX TRINITY — Python Brain Layer (Layer A)")
    log.info("Symbols: %s", TRADE_SYMBOLS)
    log.info("NATS: %s / %s", NATS_URL, NATS_SUBJECT_RAW)
    log.info("Mode: %s", "ONCE" if args.once else f"CONTINUOUS (interval={args.interval}s)")
    log.info("=" * 60)

    try:
        asyncio.run(main_loop(run_once=args.once, interval=args.interval))
    except KeyboardInterrupt:
        log.info("Interrupted by user")
