"""
Entry point — boots uvloop, launches concurrent per-symbol supervisor tasks,
starts the market data feed, web dashboard, and optional API gateway.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

try:
    import uvloop  # type: ignore[import]
    _HAVE_UVLOOP = True
except ImportError:
    _HAVE_UVLOOP = False

from prometheus_client import start_http_server

from agents.features import get_feature_extractor
from agents.paper_account import PaperAccount
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


# ── Concurrent per-symbol supervisor loops ─────────────────────────────────

async def _run_symbol_loop(
    supervisor: PortfolioSupervisor,
    feed: MarketDataFeed,
    symbol: str,
    cfg,
) -> None:
    """Independent supervisor cycle for a single symbol.

    Each symbol runs in its own ``while True`` loop with its own cadence,
    so N symbols never block one another.
    """
    logger.info("Supervisor loop started for %s (cadence=%.1fs)", symbol, cfg.supervisor_cadence_s)
    while True:
        tick = feed.latest_tick(symbol)
        if tick is None:
            await asyncio.sleep(1)
            continue
        closes = feed.get_closes(symbol).tolist()
        highs = feed.get_highs(symbol).tolist()
        lows = feed.get_lows(symbol).tolist()
        volumes = feed.get_volumes(symbol).tolist()
        if len(closes) < 50:
            await asyncio.sleep(10)
            continue  # not enough warmup data
        try:
            await supervisor.run_cycle(symbol, closes, highs, lows, volumes)
        except Exception as exc:
            logger.error("Supervisor cycle error for %s: %s", symbol, exc, exc_info=True)
        await asyncio.sleep(cfg.supervisor_cadence_s)


# ── Web dashboard server (async uvicorn) ──────────────────────────────────
_dashboard_task: asyncio.Task | None = None


async def _start_dashboard() -> None:
    """Start the FastAPI dashboard server as an async uvicorn task."""
    try:
        import uvicorn  # type: ignore[import]

        config = uvicorn.Config(
            "web.server:app",
            host="0.0.0.0",
            port=3000,
            log_level="info",
            reload=False,
        )
        server = uvicorn.Server(config)
        logger.info("Starting web dashboard on http://0.0.0.0:3000")
        await server.serve()
    except ImportError:
        logger.warning("uvicorn not installed — dashboard disabled. pip install uvicorn fastapi")
    except Exception as exc:
        logger.error("Dashboard server error: %s", exc)


# ── API Gateway server (optional, async uvicorn) ──────────────────────────
_gateway_task: asyncio.Task | None = None


async def _start_gateway() -> None:
    """Start the API Gateway with JWT auth & rate limiting (optional)."""
    cfg = get_settings()
    if not cfg.gateway_enabled:
        logger.info("API Gateway disabled (set GATEWAY_ENABLED=True to enable)")
        return
    try:
        import uvicorn  # type: ignore[import]

        config = uvicorn.Config(
            "gateway.app:app",
            host="0.0.0.0",
            port=cfg.gateway_port,
            log_level="info",
            reload=False,
        )
        server = uvicorn.Server(config)
        logger.info("Starting API Gateway on http://0.0.0.0:%d", cfg.gateway_port)
        await server.serve()
    except ImportError:
        logger.warning("uvicorn not installed — gateway disabled")
    except Exception as exc:
        logger.error("Gateway server error: %s", exc)


async def main() -> None:
    cfg = get_settings()

    # Start Prometheus metrics HTTP server (runs in a background daemon thread
    # internally — pragmatic choice; prometheus_client lacks full async support)
    start_http_server(cfg.prometheus_port)
    logger.info("Prometheus metrics on :%d/metrics", cfg.prometheus_port)

    # Start web dashboard as an async task instead of a daemon thread
    global _dashboard_task
    _dashboard_task = asyncio.create_task(_start_dashboard(), name="dashboard")

    # Start API Gateway (if enabled)
    global _gateway_task
    if cfg.gateway_enabled:
        _gateway_task = asyncio.create_task(_start_gateway(), name="gateway")

    # ── Paper trading account (enabled via --paper or PAPER_TRADING=True) ──
    paper_account: PaperAccount | None = None
    if cfg.paper_trading:
        paper_account = PaperAccount(initial_capital=cfg.initial_capital)
        logger.info(
            "Paper trading enabled — account=%.2f  initial=%.2f",
            paper_account.equity, paper_account.initial_capital,
        )

    feed = MarketDataFeed()
    supervisor = PortfolioSupervisor(
        initial_equity=cfg.initial_capital,
        paper_account=paper_account,
    )
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

    # ── Launch one concurrent supervisor task per symbol ─────────────────
    supervisor_tasks = [
        asyncio.create_task(
            _run_symbol_loop(supervisor, feed, sym, cfg),
            name=f"supervisor-{sym}",
        )
        for sym in cfg.symbols
    ]

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("Main task cancelled — shutting down")
    finally:
        feed_task.cancel()
        for st in supervisor_tasks:
            st.cancel()

        # Cancel dashboard task
        if _dashboard_task is not None:
            _dashboard_task.cancel()
            try:
                await _dashboard_task
            except asyncio.CancelledError:
                pass

        # Cancel gateway task
        if _gateway_task is not None:
            _gateway_task.cancel()
            try:
                await _gateway_task
            except asyncio.CancelledError:
                pass

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


# ── Backtest CLI ──────────────────────────────────────────────────────────

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
    print(f"  Period:         {cfg.start} \u2192 {cfg.end}")
    print(f"  Timeframe:      {cfg.timeframe}")
    print(f"  Initial capital: ${cfg.initial_capital:,.2f}")
    print("\u2500" * 60)
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
        print(f"\n\u26a0  {len(result.errors)} error(s):")
        for e in result.errors[:5]:
            print(f"   \u2022 {e}")

    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-agent AI trading system \u2014 live or backtest mode.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest mode (replay historical data)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading with position/P&L tracking (overrides PAPER_TRADING env)",
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
        # Override PAPER_TRADING env if --paper flag is passed
        if args.paper:
            import os as _os
            _os.environ["PAPER_TRADING"] = "True"
        if _HAVE_UVLOOP:
            uvloop.install()  # type: ignore[union-attr]
        else:
            logger.info("uvloop not available \u2014 using asyncio default event loop")
        asyncio.run(main())
