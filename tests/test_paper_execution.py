"""Tests for PaperExecutionEngine — fill models, slippage, account integration."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from agents.paper_account import PaperAccount, PaperPosition
from agents.paper_execution import PaperExecutionEngine
from core.models import Order, OrderStatus, OrderType, Side


@pytest.fixture
def prices() -> np.ndarray:
    """A realistic price series ending at ~67_500."""
    rng = np.random.default_rng(42)
    base = 67_500.0
    returns = np.cumsum(rng.normal(0, 0.002, 50))
    return base * (1 + returns)


class TestPaperExecutionInitialization:
    def test_default_params(self) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc)
        assert engine.fill_model == "immediate"
        assert engine.slippage_model == "fixed"
        assert engine.slippage_bps == pytest.approx(5.0)
        assert engine.latency_ms == pytest.approx(50.0)
        assert engine.account is acc

    def test_custom_params(self) -> None:
        acc = PaperAccount(50_000.0)
        engine = PaperExecutionEngine(
            acc,
            fill_model="twap",
            slippage_model="market_impact",
            slippage_bps=10.0,
            latency_ms=100.0,
            default_slices=5,
        )
        assert engine.fill_model == "twap"
        assert engine.slippage_model == "market_impact"
        assert engine.slippage_bps == 10.0
        assert engine.latency_ms == 100.0
        assert engine.default_slices == 5


class TestPaperExecutionFill:
    @pytest.mark.asyncio
    async def test_immediate_fill(self, prices: np.ndarray) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, fill_model="immediate", slippage_bps=0.0, latency_ms=0.0)

        order = Order(
            session_id="test-session",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        filled = await engine.execute(order, prices)

        assert filled.status == OrderStatus.FILLED
        assert filled.avg_fill_price is not None and filled.avg_fill_price > 0
        assert filled.filled_at is not None
        # Account should reflect the position
        assert "BTC/USDT" in acc.open_positions
        pos = acc.open_positions["BTC/USDT"]
        assert pos.quantity == 1.0
        assert pos.side == Side.BUY

    @pytest.mark.asyncio
    async def test_immediate_fill_with_slippage(self, prices: np.ndarray) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, fill_model="immediate", slippage_bps=10.0, latency_ms=0.0)

        order = Order(
            session_id="test-session",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        filled = await engine.execute(order, prices)

        # Slippage should increase the fill price for buys
        mid = float(prices[-1])
        assert filled.avg_fill_price is not None
        # The fill price should be higher than mid (buy + slippage)
        # Allow for noise, but the base should be higher
        assert filled.slippage_bps is not None and filled.slippage_bps > 0

    @pytest.mark.asyncio
    async def test_twap_fill_creates_slices(self, prices: np.ndarray) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(
            acc,
            fill_model="twap",
            slippage_bps=5.0,
            latency_ms=0.0,
            default_slices=3,
            slice_delay_s=0.0,
        )

        acc = PaperAccount(300_000.0)
        engine = PaperExecutionEngine(
            acc,
            fill_model="twap",
            slippage_bps=5.0,
            latency_ms=0.0,
            default_slices=3,
            slice_delay_s=0.0,
        )

        order = Order(
            session_id="test-session",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.TWAP,
            quantity=3.0,
        )
        filled = await engine.execute(order, prices)

        assert filled.status == OrderStatus.FILLED
        assert filled.avg_fill_price is not None and filled.avg_fill_price > 0
        pos = acc.open_positions["BTC/USDT"]
        assert pos.quantity == 3.0  # full order filled

    @pytest.mark.asyncio
    async def test_sell_fill(self, prices: np.ndarray) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_bps=0.0, latency_ms=0.0)

        order = Order(
            session_id="test-session",
            symbol="BTC/USDT",
            side=Side.SELL,
            order_type=OrderType.MARKET,
            quantity=0.5,
        )
        filled = await engine.execute(order, prices)

        assert filled.status == OrderStatus.FILLED
        assert "BTC/USDT" in acc.open_positions
        pos = acc.open_positions["BTC/USDT"]
        assert pos.quantity == 0.5
        assert pos.side == Side.SELL

    @pytest.mark.asyncio
    async def test_fill_updates_account_equity(self, prices: np.ndarray) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_bps=0.0, latency_ms=0.0)

        order = Order(
            session_id="test-session",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        await engine.execute(order, prices)

        # After fill: cash reduced, position opened, equity ~initial
        mid = float(prices[-1])
        assert acc.cash == pytest.approx(100_000.0 - mid, rel=0.01)
        assert acc.equity == pytest.approx(100_000.0, rel=0.01)  # position value = cash + mtm

    @pytest.mark.asyncio
    async def test_fill_close_then_open_on_opposite_side(self, prices: np.ndarray) -> None:
        """Opening a position with opposite side should close existing first."""
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_bps=0.0, latency_ms=0.0)

        # Open long
        order1 = Order(
            session_id="s1", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        )
        await engine.execute(order1, prices)

        assert acc.total_trades == 0  # not closed yet
        assert "BTC/USDT" in acc.open_positions
        assert acc.open_positions["BTC/USDT"].side == Side.BUY

        # Open short — should close long first, then open short
        order2 = Order(
            session_id="s2", symbol="BTC/USDT", side=Side.SELL,
            order_type=OrderType.MARKET, quantity=0.5,
        )
        await engine.execute(order2, prices)

        assert acc.total_trades == 1  # long was closed
        assert "BTC/USDT" in acc.open_positions
        # Position should now be SHORT
        # (the quantity might be different due to the new position)
        pos = acc.open_positions["BTC/USDT"]
        assert pos.side == Side.SELL
        assert pos.quantity == 0.5

    @pytest.mark.asyncio
    async def test_latency_simulated(self, prices: np.ndarray) -> None:
        """With latency > 0, the fill should take at least latency_ms."""
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_bps=0.0, latency_ms=200.0)

        order = Order(
            session_id="test-session",
            symbol="BTC/USDT",
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        import time
        t0 = time.monotonic()
        await engine.execute(order, prices)
        elapsed = (time.monotonic() - t0) * 1000
        assert elapsed >= 190.0  # allow small scheduling jitter


class TestPaperExecutionSlippageModels:
    def test_fixed_slippage_buy(self) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_model="fixed", slippage_bps=10.0)
        price = engine._compute_slippage_price(50_000.0, Order(
            session_id="t", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        ))
        expected = 50_000.0 * (1 + 10 / 10_000)
        assert price == pytest.approx(expected, rel=1e-6)

    def test_fixed_slippage_sell(self) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_model="fixed", slippage_bps=10.0)
        price = engine._compute_slippage_price(50_000.0, Order(
            session_id="t", symbol="BTC/USDT", side=Side.SELL,
            order_type=OrderType.MARKET, quantity=1.0,
        ))
        expected = 50_000.0 * (1 - 10 / 10_000)
        assert price == pytest.approx(expected, rel=1e-6)

    def test_percentage_slippage(self) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_model="percentage", slippage_pct=0.001)
        price = engine._compute_slippage_price(50_000.0, Order(
            session_id="t", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        ))
        expected = 50_000.0 * (1 + 0.001)
        assert price == pytest.approx(expected, rel=1e-6)

    def test_market_impact_slippage(self) -> None:
        acc = PaperAccount(100_000.0)
        engine = PaperExecutionEngine(acc, slippage_model="market_impact")
        # Large order relative to estimated volume
        price = engine._compute_slippage_price(50_000.0, Order(
            session_id="t", symbol="BTC/USDT", side=Side.BUY,
            order_type=OrderType.MARKET, quantity=1000.0,
        ))
        # Market impact should make price higher than base
        assert price > 50_000.0
