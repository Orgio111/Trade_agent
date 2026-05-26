"""
Entry point — boots uvloop, starts the market data feed, supervisor loop,
and the web monitoring dashboard.
"""
from __future__ import annotations

import argparse
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

from agents.features import get_feature_extractor
from agents.supervisor import PortfolioSupervisor
from backtest import BacktestConfig, BacktestEngine
from core.config import get_settings
from core.market_data import MarketDataFeed
from memory.pipeline import get_memory_pipeline as get_semantic_pipeline

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

    def _shutdown() -> None:
        logger.info("Shutdown signal received — stopping")
        stop_event.set()

    # Register signal handlers (Unix only — Windows falls back to Ctrl+C)
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)
    else:
        logger.info("Signal handlers unavailable on Windows — press Ctrl+C to stop gracefully")

    feed_task = asyncio.create_task(feed.start(), name="market_feed")
    supervisor_task = asyncio.create_task(
        _run_supervisor_loop(supervisor, feed, cfg), name="supervisor"
    )

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("Main task cancelled — shutting down")
    finally:
        feed_task.cancel()
        supervisor_task.cancel()
        await feed.stop()

        # Stop feature extractor
        try:
            extractor = get_feature_extractor()
            await extractor.stop()
        except Exception:
            pass

        # Save memory index on shutdown
        try:
            pipe = get_semantic_pipeline()
            await pipe.stop()
        except Exception:
            pass

        logger.info("Shutdown complete.")


# ── Backtest CLI ──────────────────────────────────────────────────────────────

async def _run_backtest(args: argparse.Namespace) -> None:
    """Entry point for the backtest sub-command."""
    cfg = BacktestConfig(
        symbols=[s.strip() for s in args.symbols.split(",")] if args.symbols
                else get_settings().symbols,
        start=args.start or get_settings().backtest_start,
        end=args.end or get_settings().backtest_end,
        initial_capital=args.capital or get_settings().initial_capital,
        exchange_id=args.exchange or get_settings().exchange,
        timeframe=args.timeframe or get_settings().backtest_timeframe,
        slippage_bps=args.slippage or get_settings().backtest_slippage_bps,
        features_enabled=args.features,
        max_cycles=args.max_cycles or get_settings().backtest_max_cycles,
    )

    engine = BacktestEngine(cfg)
    result = await engine.run()

    # Print summary to stdout
    summary = result.to_dict()
    print("\n" + "═" * 60)
    print("  BACKTEST RESULTS")
    print("═" * 60)
    print(f"  Symbols:        {cfg.symbols}")
    print(f"  Period:         {cfg.start} → {cfg.end}")
    print(f"  Timeframe:      {cfg.timeframe}")
    print(f"  Initial capital: ${cfg.initial_capital:,.2f}")
    print("─" * 60)
    s = summary["summary"]
    print(f"  Final equity:    ${s['final_equity']:,.2f}")
    print(f"  Total return:    {s['total_return_pct']:+.2f}%")
    print(f"  Sharpe ratio:    {s['sharpe_ratio']:.3f}")
    print(f"  Sortino ratio:   {s['sortino_ratio']:.3f}")
    print(f"  Max drawdown:    {s['max_drawdown_pct']:.2f}%")
    print(f"  Win rate:        {s['win_rate']:.1%}")
    print(f"  Profit factor:   {s['profit_factor']:.3f}")
    print(f"  Total trades:    {s['total_trades']}")
    print(f"  Calmar ratio:    {s['calmar_ratio']:.3f}")
    print("═" * 60)

    if result.errors:
        print(f"\n⚠  {len(result.errors)} error(s):")
        for e in result.errors[:5]:
            print(f"   • {e}")

    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-agent AI trading system — live or backtest mode.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode (replay historical data)",
    )
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols for backtest")
    parser.add_argument("--start", type=str, help="Backtest start date (ISO)")
    parser.add_argument("--end", type=str, help="Backtest end date (ISO)")
    parser.add_argument("--capital", type=float, help="Initial capital for backtest")
    parser.add_argument("--exchange", type=str, help="Exchange ID for backtest data")
    parser.add_argument("--timeframe", type=str, help="Candle interval (1h, 1m, etc.)")
    parser.add_argument("--slippage", type=float, help="Slippage in bps")
    parser.add_argument("--features", action="store_true", help="Enable feature extraction in backtest")
    parser.add_argument("--max-cycles", type=int, help="Limit bars processed")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.backtest:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
            stream=sys.stdout,
        )
        asyncio.run(_run_backtest(args))
    else:
        if _HAVE_UVLOOP:
            uvloop.install()  # type: ignore[union-attr]
        else:
            logger.info("uvloop not available — using asyncio default event loop")
        asyncio.run(main())
