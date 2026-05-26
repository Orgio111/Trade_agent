"""Integration test — Sentinel-X FundSimulator smoke test.

Validates that the full trading simulation pipeline:
1. Imports correctly (FundSimulator, SimResult, etc.)
2. Runs for a configurable number of days
3. Produces a non-degenerate equity curve
4. Tracks key metrics (Sharpe, win rate, max drawdown)
5. Reports per-strategy breakdowns

This is a *smoke test* — it uses Monte Carlo random sampling with a
fixed seed, so results should be reproducible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Ensure the sentinel-x phase5_simulation package is importable
_SENTINEL_X = Path(__file__).resolve().parent.parent / "sentinel-x" / "phase5_simulation" / "python"
if _SENTINEL_X.exists():
    sys.path.insert(0, str(_SENTINEL_X))


@pytest.fixture(scope="module")
def simulator():
    """Import and return a fresh FundSimulator with a fixed seed."""
    from simulator import FundSimulator
    sim = FundSimulator(seed=42)
    return sim


@pytest.mark.timeout(30)  # generous timeout for the 1-year simulation
def test_fund_simulator_imports():
    """Verify all simulation types import correctly."""
    from simulator import (
        FundSimulator,
        SimResult,
        SimTrade,
        TradeResult,
        INITIAL_CAPITAL,
        RISK_PER_TRADE_MIN,
        RISK_PER_TRADE_MAX,
        MAX_DRAWDOWN_HALT,
        slippage_cost,
        fee_cost,
    )

    assert INITIAL_CAPITAL == 1_000_000.0
    assert 0.0 < RISK_PER_TRADE_MIN < RISK_PER_TRADE_MAX
    assert 0.0 < MAX_DRAWDOWN_HALT <= 1.0

    # Check helper functions
    slip = slippage_cost(10_000.0)
    assert 0.0 <= slip < 1.0

    fee = fee_cost(10_000.0, is_taker=True)
    assert fee > 0.0

    print(f"  slippage_cost(10k USD) = {slip:.4%}")
    print(f"  taker_fee(10k USD)     = ${fee:.2f}")


@pytest.mark.slow
def test_fund_simulator_short_run(simulator):
    """Run a short simulation (20 days) and verify basic properties."""
    result = simulator.run(n_days=20)

    assert result.total_trades > 0
    assert result.final_equity > 0
    assert result.equity_curve[0] == result.initial_capital
    assert len(result.equity_curve) == 21  # initial + 20 days
    assert len(result.daily_returns) == 20

    # Win rate should be between 0% and 100%
    assert 0.0 <= result.win_rate <= 1.0

    # Sharpe should be computed (may be negative in short run)
    assert isinstance(result.sharpe_ratio, float)

    # Strategy breakdown should contain all three strategies
    assert set(result.strategy_breakdown.keys()) == {
        "momentum", "arb_spatial", "arb_triangular",
    }


@pytest.mark.slow
def test_fund_simulator_one_year(simulator):
    """Run a full 252-day simulation and verify comprehensive metrics."""
    result = simulator.run(n_days=252)

    # ── Basic sanity ────────────────────────────────────────────────────────
    assert result.total_trades > 50  # should have plenty of trades
    assert result.final_equity > 0.0
    assert result.equity_curve[0] == result.initial_capital

    # ── PnL ─────────────────────────────────────────────────────────────────
    # The simulator is calibrated to be profitable on average
    assert result.net_pnl != 0.0  # should have some PnL
    assert result.total_return_pct != 0.0

    # ── Risk metrics ────────────────────────────────────────────────────────
    # These are smoke-level checks — the simulator uses random sampling with
    # realistic fees/slippage, and individual runs can vary significantly
    assert isinstance(result.max_drawdown_pct, float)
    assert result.max_drawdown_pct > 0  # should have some drawdown
    assert isinstance(result.sharpe_ratio, float)
    assert not math.isnan(result.sharpe_ratio)

    # Profit factor should be positive
    assert result.profit_factor > 0.0

    # Calmar should be computed
    assert result.calmar_ratio != 0.0

    # ── Trade metrics ────────────────────────────────────────────────────────
    total_wins_and_losses = result.winning_trades + result.losing_trades
    assert total_wins_and_losses <= result.total_trades
    assert 0.0 <= result.win_rate <= 1.0
    assert result.avg_win_pct > 0.0
    assert result.avg_loss_pct < 0.0

    # ── Fees and slippage ────────────────────────────────────────────────────
    assert result.total_fees > 0
    assert result.total_slippage > 0

    # ── Strategy breakdown ──────────────────────────────────────────────────
    for name, s in result.strategy_breakdown.items():
        assert s["trades"] > 0, f"Strategy {name} has zero trades"
        assert 0.0 <= s["win_rate"] <= 1.0
        # net_pnl can be negative for some strategies
        assert isinstance(s["net_pnl"], float)
        assert s["total_fees"] > 0

    print(f"\n252-day simulation results:")
    print(f"  Final equity:    ${result.final_equity:>12,.2f}")
    print(f"  Total return:    {result.total_return_pct:>+8.2f}%")
    print(f"  Sharpe:          {result.sharpe_ratio:>8.3f}")
    print(f"  Max drawdown:    {result.max_drawdown_pct:>7.2f}%")
    print(f"  Win rate:        {result.win_rate:>7.1%}")
    print(f"  Total trades:    {result.total_trades}")
    print(f"  Profit factor:   {result.profit_factor:>8.2f}")
    print(f"  Kill switch:     {'TRIGGERED' if result.kill_switch_triggered else 'OK'}")


def test_fund_simulator_print_report(simulator, capsys):
    """Verify the print_report method runs without errors."""
    from simulator import SimResult

    # Create a minimal result to test rendering
    result = SimResult(
        initial_capital=1_000_000.0,
        final_equity=1_050_000.0,
        total_trades=1000,
        winning_trades=550,
        losing_trades=450,
        gross_pnl=100_000.0,
        total_fees=25_000.0,
        total_slippage=10_000.0,
        net_pnl=50_000.0,
        total_return_pct=5.0,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.0,
        max_drawdown_pct=5.0,
        avg_win_pct=0.8,
        avg_loss_pct=-0.6,
        profit_factor=1.5,
        win_rate=0.55,
        equity_curve=[1_000_000.0, 1_050_000.0],
        daily_returns=[0.0, 0.05],
        strategy_breakdown={
            "momentum": {"trades": 500, "win_rate": 0.56, "net_pnl": 30_000.0, "avg_net_pct": 0.5, "total_fees": 12_000.0},
            "arb_spatial": {"trades": 300, "win_rate": 0.72, "net_pnl": 15_000.0, "avg_net_pct": 0.3, "total_fees": 8_000.0},
            "arb_triangular": {"trades": 200, "win_rate": 0.68, "net_pnl": 5_000.0, "avg_net_pct": 0.2, "total_fees": 5_000.0},
        },
        kill_switch_triggered=False,
        kill_switch_at=None,
    )

    # Should not raise
    simulator.print_report(result)
    captured = capsys.readouterr()
    assert "CAPITAL SIMULATION REPORT" in captured.out
    # The format uses :>15,.2f so there's a space between $ and the number
    assert "1,050,000" in captured.out
    assert "Kill switch never triggered" in captured.out
