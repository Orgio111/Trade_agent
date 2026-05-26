"""Tests for PaperAccount — P&L tracking, positions, equity curve."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agents.paper_account import PaperAccount, PaperPosition, ClosedTrade
from core.models import Side


class TestPaperAccountInit:
    def test_default_initial_capital(self) -> None:
        acc = PaperAccount()
        assert acc.initial_capital == 100_000.0
        assert acc.cash == 100_000.0
        assert acc.equity == pytest.approx(100_000.0, rel=1e-3)
        assert acc.total_trades == 0
        assert acc.win_rate == 0.0
        assert acc.sharpe_ratio == 0.0

    def test_custom_initial_capital(self) -> None:
        acc = PaperAccount(initial_capital=50_000.0)
        assert acc.initial_capital == 50_000.0
        assert acc.cash == 50_000.0

    def test_snapshot_fields(self) -> None:
        acc = PaperAccount(100_000.0)
        snap = acc.snapshot()
        assert snap["initial_capital"] == 100_000.0
        assert snap["cash"] == 100_000.0
        assert snap["total_trades"] == 0
        assert snap["mode"] == "paper"
        assert len(snap["equity_history"]) == 1
        assert len(snap["open_positions"]) == 0


class TestPaperAccountOpenClose:
    def test_open_position_deducts_cash(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        assert acc.cash == pytest.approx(50_000.0, rel=1e-3)
        assert "BTC/USDT" in acc.open_positions
        pos = acc.open_positions["BTC/USDT"]
        assert pos.quantity == 1.0
        assert pos.entry_price == 50_000.0
        assert pos.side == Side.BUY

    def test_close_position_credits_cash_and_records_trade(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        trade = acc.close_position("BTC/USDT", 55_000.0)
        assert trade is not None
        assert trade.pnl == pytest.approx(5_000.0, rel=1e-3)
        assert trade.win is True
        assert acc.cash == pytest.approx(105_000.0, rel=1e-3)
        assert acc.total_trades == 1
        assert acc.win_rate == 1.0
        assert "BTC/USDT" not in acc.open_positions

    def test_losing_trade(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("ETH/USDT", Side.BUY, 10.0, 3_000.0)
        trade = acc.close_position("ETH/USDT", 2_500.0)
        assert trade is not None
        assert trade.pnl == pytest.approx(-5_000.0, rel=1e-3)
        assert trade.win is False
        assert acc.win_rate == 0.0

    def test_short_position(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("SOL/USDT", Side.SELL, 100.0, 150.0)
        assert acc.cash == pytest.approx(85_000.0, rel=1e-3)  # 100_000 - (100 * 150)

        # Price drops — short wins
        trade = acc.close_position("SOL/USDT", 120.0)
        assert trade is not None
        assert trade.pnl == pytest.approx(3_000.0, rel=1e-3)  # (150 - 120) * 100
        assert trade.win is True

    def test_close_nonexistent_position_returns_none(self) -> None:
        acc = PaperAccount(100_000.0)
        trade = acc.close_position("NONEXIST", 100.0)
        assert trade is None

    def test_insufficient_cash_skips_open(self) -> None:
        acc = PaperAccount(initial_capital=1_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)  # far exceeds cash
        assert "BTC/USDT" not in acc.open_positions
        assert acc.cash == pytest.approx(1_000.0, rel=1e-3)

    def test_cannot_double_open(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 0.5, 50_000.0)  # already open
        # Quantity should remain unchanged
        assert acc.open_positions["BTC/USDT"].quantity == 1.0


class TestPaperAccountEquity:
    def test_equity_reflects_unrealised_pnl(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.mark_to_market("BTC/USDT", 55_000.0)
        # Cash = 50k, position worth 55k, equity = 105k
        assert acc.equity == pytest.approx(105_000.0, rel=1e-3)
        assert acc.unrealised_pnl == pytest.approx(5_000.0, rel=1e-3)

    def test_equity_correct_after_close(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.mark_to_market("BTC/USDT", 60_000.0)
        # equity = 110k
        assert acc.equity == pytest.approx(110_000.0, rel=1e-3)
        trade = acc.close_position("BTC/USDT", 60_000.0)
        assert trade is not None
        # After close: cash = 50k + 60k = 110k, no positions, equity = 110k
        assert acc.equity == pytest.approx(110_000.0, rel=1e-3)

    def test_drawdown_tracking(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.mark_to_market("BTC/USDT", 65_000.0)  # equity = 115k, peak = 115k
        assert acc.peak_equity == pytest.approx(115_000.0, rel=1e-3)
        acc.mark_to_market("BTC/USDT", 55_000.0)  # equity = 105k
        dd = acc.current_drawdown_pct
        expected_dd = (115_000 - 105_000) / 115_000
        assert dd == pytest.approx(expected_dd, rel=1e-3)

    def test_win_rate_multiple_trades(self) -> None:
        acc = PaperAccount(100_000.0)
        # Trade 1: win
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.close_position("BTC/USDT", 55_000.0)
        # Trade 2: loss
        acc.open_position("ETH/USDT", Side.BUY, 10.0, 3_000.0)
        acc.close_position("ETH/USDT", 2_800.0)
        # Trade 3: win
        acc.open_position("SOL/USDT", Side.BUY, 100.0, 100.0)
        acc.close_position("SOL/USDT", 120.0)
        assert acc.total_trades == 3
        assert acc.win_rate == pytest.approx(2 / 3, rel=1e-3)

    def test_sharpe_zero_with_few_trades(self) -> None:
        acc = PaperAccount(100_000.0)
        assert acc.sharpe_ratio == 0.0
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.close_position("BTC/USDT", 50_500.0)
        # Only 1 trade — std dev undefined → sharpe = 0
        assert acc.sharpe_ratio == 0.0

    def test_equity_history(self) -> None:
        acc = PaperAccount(100_000.0, max_history=5)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.mark_to_market("BTC/USDT", 55_000.0)
        acc.mark_to_market("BTC/USDT", 60_000.0)
        hist = acc.snapshot()["equity_history"]
        assert len(hist) >= 3  # initial + 2 MTM updates
        assert hist[0]["v"] == pytest.approx(100_000.0, rel=1e-3)
        assert hist[-1]["v"] == pytest.approx(110_000.0, rel=1e-3)

    def test_reset(self) -> None:
        acc = PaperAccount(100_000.0)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0)
        acc.close_position("BTC/USDT", 55_000.0)
        assert acc.total_trades == 1
        acc.reset()
        assert acc.cash == 100_000.0
        assert acc.total_trades == 0
        assert len(acc.open_positions) == 0

    def test_average_holding_period(self) -> None:
        acc = PaperAccount(100_000.0)
        t0 = datetime.utcnow() - timedelta(hours=2)
        acc.open_position("BTC/USDT", Side.BUY, 1.0, 50_000.0, timestamp=t0)
        acc.close_position("BTC/USDT", 55_000.0)
        avg = acc.average_holding_period_s
        # Should be ~7200 seconds (2 hours)
        assert avg == pytest.approx(7200.0, rel=0.1)
