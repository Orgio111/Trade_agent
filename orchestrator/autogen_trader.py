"""AutoGen Trader CLI — command-line entry point.

Modes:
  single   — one trading cycle and exit
  daemon   — continuous trading loop (configurable interval)
  backtest — replay historical candles through autogen pipeline
  status   — show system status

Usage:
  python -m orchestrator.autogen_trader single --paper
  python -m orchestrator.autogen_trader daemon --paper --interval 60
  python -m orchestrator.autogen_trader backtest --data data/btc_1m.csv
  python -m orchestrator.autogen_trader status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# ── Logging ─────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"


def _setup_logging(level: str = "INFO", log_file: str | None = None):
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT, handlers=handlers)


def _load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Demo candle data ────────────────────────────────────────

DEMO_CANDLE: dict[str, Any] = {
    "symbol": "BTCUSDT",
    "price": 61250.0,
    "ohlcv": {
        "open": 61000.0,
        "high": 61380.0,
        "low": 60900.0,
        "close": 61250.0,
        "volume": 1523.5,
    },
    "indicators": {
        "rsi_14": 62.3,
        "atr_14": 580.0,
        "ema_9": 61150.0,
        "ema_21": 60800.0,
        "bb_width": 0.032,
    },
    "regime": "trending",
}

DEMO_VLM: dict[str, Any] = {
    "trend": "bullish",
    "pattern": "ascending_triangle",
    "support": 60500.0,
    "resistance": 62000.0,
    "confidence": 0.78,
}


# ── Candle fetcher (Binance public API) ─────────────────────

def _fetch_latest_candle(symbol: str = "BTCUSDT", interval: str = "1m") -> dict:
    """Fetch latest candle from Binance public API (no auth needed)."""
    import requests

    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 2}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()
        if len(klines) < 2:
            raise ValueError("No kline data returned")

        k = klines[-2]  # last completed candle
        return {
            "symbol": symbol,
            "price": float(k[4]),  # close
            "ohlcv": {
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            },
            "indicators": {},  # indicators filled by feature pipeline
            "regime": "unknown",
        }
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to fetch candle: {e}. Using demo data.")
        return DEMO_CANDLE


# ── Commands ────────────────────────────────────────────────


def cmd_single(args):
    """Execute one trading cycle."""
    from orchestrator.autogen_executor import AutoGenSignalExecutor

    _setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("autogen_trader")

    logger.info("=== AutoGen Trader — Single Cycle Mode ===")
    logger.info(f"Paper mode: {args.paper}")

    executor = AutoGenSignalExecutor(
        paper_mode=args.paper,
        config_path=args.config,
        initial_balance=args.balance,
        max_round=args.max_round,
        model_warmup=not args.no_warmup,
    )

    # Get candle data
    if args.live_data:
        candle = _fetch_latest_candle(args.symbol, args.interval)
        logger.info(f"Fetched live candle: {candle['symbol']} @ {candle['price']}")
    else:
        candle = DEMO_CANDLE
        logger.info("Using demo candle data")

    # Run cycle
    result = executor.execute_cycle(candle_data=candle, vlm_output=DEMO_VLM)
    _print_result(result)

    # Save result
    if args.output:
        _save_result(result, args.output)


def cmd_daemon(args):
    """Continuous trading loop."""
    from orchestrator.autogen_executor import AutoGenSignalExecutor

    _setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("autogen_trader")

    logger.info("=== AutoGen Trader — Daemon Mode ===")
    logger.info(f"Paper mode: {args.paper}, Interval: {args.interval}s")

    executor = AutoGenSignalExecutor(
        paper_mode=args.paper,
        config_path=args.config,
        initial_balance=args.balance,
        max_round=args.max_round,
        model_warmup=not args.no_warmup,
    )

    cycle = 0
    try:
        while True:
            cycle += 1
            logger.info(f"\n{'='*50} Cycle #{cycle} {'='*50}")

            if args.live_data:
                candle = _fetch_latest_candle(args.symbol, args.interval)
            else:
                candle = DEMO_CANDLE

            result = executor.execute_cycle(candle_data=candle, vlm_output=DEMO_VLM)
            _print_result(result, verbose=False)

            # Check if we should stop
            if result.status == "error" and args.stop_on_error:
                logger.error("Stopping on error")
                break

            # Status summary every 10 cycles
            if cycle % 10 == 0:
                status = executor.get_status()
                logger.info(f"Account: equity={status['account']['equity']:.2f}, "
                          f"positions={status['account']['open_positions']}, "
                          f"streak={status['account']['loss_streak']}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("\nDaemon stopped by user")
        status = executor.get_status()
        logger.info(f"Final status: {json.dumps(status, indent=2, default=str)}")


def cmd_backtest(args):
    """Replay historical data through autogen pipeline."""
    from orchestrator.autogen_executor import AutoGenSignalExecutor
    from orchestrator.backtest_engine import BacktestEngine

    _setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("autogen_trader")

    logger.info("=== AutoGen Trader — Backtest Mode ===")

    if not args.data:
        logger.error("--data flag required for backtest (CSV file path)")
        sys.exit(1)

    executor = AutoGenSignalExecutor(
        paper_mode=True,  # always paper in backtest
        config_path=args.config,
        initial_balance=args.balance,
        max_round=args.max_round,
        model_warmup=False,  # skip warmup in backtest
    )

    engine = BacktestEngine(
        executor=executor,
        data_path=args.data,
        sample_rate=args.sample_rate,
        max_cycles=args.max_cycles,
    )

    report = engine.run()
    _print_backtest_report(report)

    if args.output:
        _save_result(report, args.output)


def cmd_status(args):
    """Show system status."""
    _setup_logging("WARNING")
    import requests as req

    # Check Ollama
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        ollama_status = f"running ({len(models)} models)"
    except Exception:
        models = []
        ollama_status = "NOT running"

    # Check config
    config = _load_config(args.config)

    print(f"""
╔══════════════════════════════════════════════════╗
║        AutoGen Trader — System Status            ║
╠══════════════════════════════════════════════════╣
║  Ollama:      {ollama_status:<35s}║
║  Models:      {', '.join(models[:4]) + ('...' if len(models)>4 else ''):<35s}║
║  Config:      {args.config:<35s}║
║  Paper mode:  {config.get('system', {}).get('paper_mode', True):<35s}║
║  Symbol:      {config.get('system', {}).get('symbol', 'BTCUSDT'):<35s}║
║  Timeframe:   {config.get('system', {}).get('timeframe', '1m'):<35s}║
╚══════════════════════════════════════════════════╝
""")


# ── Output helpers ──────────────────────────────────────────


def _print_result(result, verbose: bool = True):
    """Print CycleResult to console."""
    status_emoji = {"completed": "✅", "rejected": "🛑", "error": "❌", "pending": "⏳"}.get(
        result.status, "?"
    )

    signal = result.langgraph_signal
    action = signal.get("action", "?")
    conf = signal.get("confidence", 0.0)

    print(f"\n  {status_emoji} [{result.cycle_id}] {result.status.upper()}")
    print(f"  Action:    {action}")
    print(f"  Confidence: {conf:.2f}")
    print(f"  Latency:   autogen={result.autogen_latency_ms:.0f}ms  "
          f"risk={result.risk_latency_ms:.0f}ms  "
          f"exec={result.exec_latency_ms:.0f}ms  "
          f"total={result.total_latency_ms:.0f}ms")

    if result.risk_decision:
        print(f"  Risk:      {'APPROVED' if result.risk_decision.get('allow') else 'BLOCKED'}"
              f" — {result.risk_decision.get('reason', '')[:60]}")

    if verbose:
        if result.execution_result:
            print(f"  Execution: {json.dumps(result.execution_result, indent=4, default=str)}")
        if result.error:
            print(f"  Error:     {result.error}")


def _print_backtest_report(report: dict):
    """Print backtest summary."""
    print(f"""
╔══════════════════════════════════════════════════╗
║           Backtest Report                         ║
╠══════════════════════════════════════════════════╣
║  Total cycles:     {report.get('total_cycles', 0):>6}                          ║
║  Completed:        {report.get('completed', 0):>6}                          ║
║  Rejected:         {report.get('rejected', 0):>6}                          ║
║  Errors:           {report.get('errors', 0):>6}                          ║
║  Final equity:     ${report.get('final_equity', 0):>10,.2f}                  ║
║  Total PnL:        ${report.get('total_pnl', 0):>10,.2f}                  ║
║  Win rate:         {report.get('win_rate', 0):>6.1%}                          ║
║  Avg latency:      {report.get('avg_latency_ms', 0):>6.0f}ms                        ║
╚══════════════════════════════════════════════════╝
""")


def _save_result(data, path: str):
    """Save result dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Result saved to {p}")


# ── Argument parser ─────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autogen_trader",
        description="AutoGen Multi-Agent Trading System — CLI entry point",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--log-file", default=None, help="Log file path")
    parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance (paper)")
    parser.add_argument("--max-round", type=int, default=6, help="AutoGen GroupChat max_round")
    parser.add_argument("--no-warmup", action="store_true", help="Skip model pre-warming")
    parser.add_argument("--output", "-o", default=None, help="Save result JSON to file")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── single ──
    p_single = sub.add_parser("single", help="One trading cycle")
    p_single.add_argument("--paper", action="store_true", default=True, help="Paper mode (default)")
    p_single.add_argument("--live", dest="paper", action="store_false", help="Live mode (DANGER!)")
    p_single.add_argument("--live-data", action="store_true", help="Fetch real candle from Binance")
    p_single.add_argument("--symbol", default="BTCUSDT")
    p_single.add_argument("--interval", default="1m")

    # ── daemon ──
    p_daemon = sub.add_parser("daemon", help="Continuous trading loop")
    p_daemon.add_argument("--paper", action="store_true", default=True)
    p_daemon.add_argument("--live", dest="paper", action="store_false")
    p_daemon.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    p_daemon.add_argument("--live-data", action="store_true")
    p_daemon.add_argument("--symbol", default="BTCUSDT")
    p_daemon.add_argument("--stop-on-error", action="store_true")

    # ── backtest ──
    p_backtest = sub.add_parser("backtest", help="Replay historical data")
    p_backtest.add_argument("--data", required=False, help="Path to historical CSV")
    p_backtest.add_argument("--sample-rate", type=int, default=1, help="Use every Nth candle")
    p_backtest.add_argument("--max-cycles", type=int, default=0, help="Max cycles (0=unlimited)")

    # ── status ──
    sub.add_parser("status", help="Show system status")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "single": cmd_single,
        "daemon": cmd_daemon,
        "backtest": cmd_backtest,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
