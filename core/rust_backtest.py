"""Rust backtest engine Python bridge.

Launches the ``sentinel-risk`` binary in ``--backtest`` CLI mode as a
subprocess, feeds it OHLCV data as JSON via stdin, and parses the JSON
result back into a Python dataclass.

Provides both a high-level ``rust_backtest()`` function and an
async-compatible helper for integration with the main event loop.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class RustBacktestResult:
    """Backtest result returned by the Rust engine."""

    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    returns: list[float] = field(default_factory=list)


# ── Binary path resolution ────────────────────────────────────────────────────


def _rust_binary_path() -> Path | None:
    """Resolve the path to the ``sentinel-risk`` binary.

    Checks, in order:
    1. ``SENTINELX_RUST_BINARY`` env var
    2. Default release build location
    """
    env_path = os.environ.get("SENTINELX_RUST_BINARY")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p.resolve()
        logger.warning("SENTINELX_RUST_BINARY set but not found: %s", env_path)

    # Default: relative to project root
    project_root = Path(__file__).resolve().parent.parent
    binary_name = "sentinel-risk.exe" if sys.platform == "win32" else "sentinel-risk"
    candidates = [
        project_root / "sentinel-x" / "rust" / "target" / "release" / binary_name,
        project_root / "sentinel-x" / "rust" / "target" / "debug" / binary_name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


# ── Sync runner ────────────────────────────────────────────────────────────────


def run_rust_backtest(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    *,
    initial_equity: float = 100_000.0,
    kelly_fraction: float = 0.25,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
    binary_path: Path | None = None,
) -> RustBacktestResult | None:
    """Run the Rust backtest engine as a subprocess.

    Parameters
    ----------
    closes, highs, lows, volumes:
        OHLCV price series (must all be the same length).
    initial_equity:
        Starting capital.
    kelly_fraction:
        Fractional Kelly for position sizing.
    atr_multiplier:
        ATR multiplier for stop-loss distance.
    atr_period:
        Number of bars for ATR calculation.
    binary_path:
        Override the Rust binary path.  If ``None``, auto-detected.

    Returns
    -------
    RustBacktestResult | None
        The backtest result, or ``None`` if the binary is not found or the
        subprocess fails.
    """
    binary = binary_path or _rust_binary_path()
    if binary is None:
        logger.warning(
            "sentinel-risk binary not found. "
            "Build it with: cd sentinel-x/rust && cargo build --release"
        )
        return None

    input_data = {
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "volumes": volumes,
        "initial_equity": initial_equity,
        "kelly_fraction": kelly_fraction,
        "atr_multiplier": atr_multiplier,
        "atr_period": atr_period,
    }

    try:
        proc = subprocess.run(
            [str(binary), "--backtest"],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except FileNotFoundError:
        logger.error("Rust binary not found at %s", binary)
        return None
    except subprocess.TimeoutExpired:
        logger.error("Rust backtest timed out after 30s")
        return None
    except Exception as exc:
        logger.error("Rust backtest subprocess error: %s", exc)
        return None

    if proc.returncode != 0:
        logger.error(
            "Rust backtest failed (exit=%d): %s",
            proc.returncode,
            proc.stderr.strip() or "(no stderr)",
        )
        return None

    try:
        data: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Rust backtest JSON output: %s", exc)
        return None

    return RustBacktestResult(
        total_trades=int(data.get("total_trades", 0)),
        winning_trades=int(data.get("winning_trades", 0)),
        total_pnl=float(data.get("total_pnl", 0.0)),
        sharpe_ratio=float(data.get("sharpe_ratio", 0.0)),
        max_drawdown=float(data.get("max_drawdown", 0.0)),
        win_rate=float(data.get("win_rate", 0.0)),
        returns=[float(r) for r in data.get("returns", [])],
    )


# ── Async wrapper ─────────────────────────────────────────────────────────────


async def run_rust_backtest_async(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    volumes: list[float],
    *,
    initial_equity: float = 100_000.0,
    kelly_fraction: float = 0.25,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> RustBacktestResult | None:
    """Async wrapper around :func:`run_rust_backtest`.

    Runs the subprocess in a thread executor to avoid blocking the event loop.
    Accepts the same parameters as :func:`run_rust_backtest`.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_rust_backtest(
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
            initial_equity=initial_equity,
            kelly_fraction=kelly_fraction,
            atr_multiplier=atr_multiplier,
            atr_period=atr_period,
        ),
    )
