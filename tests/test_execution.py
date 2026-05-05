"""Unit tests for execution.py — paper trading path."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agents.execution import ExecutionAgent, _build_obs
from core.models import (
    CouncilDecision,
    DebateArgument,
    OrderStatus,
    OrderType,
    RiskReport,
    Side,
    TechnicalSignal,
)


@pytest.fixture
def sample_prices():
    rng = np.random.default_rng(0)
    return 65000.0 + np.cumsum(rng.normal(0, 100, 200))


@pytest.fixture
def sample_decision():
    tech = TechnicalSignal(
        symbol="BTC/USDT",
        timestamp=datetime.utcnow(),
        rsi_14=52.0,
        ema_fast=65500.0,
        ema_slow=65000.0,
        atr_14=400.0,
        macd=100.0,
        macd_signal=80.0,
        bb_upper=66500.0,
        bb_lower=64000.0,
        volume_ratio=1.3,
        trend=Side.BUY,
        confidence=0.72,
    )
    return CouncilDecision(
        symbol="BTC/USDT",
        timestamp=datetime.utcnow(),
        session_id="test-session",
        bull_score=0.70,
        bear_score=0.30,
        consensus_score=0.40,
        final_side=Side.BUY,
        debate_log=[],
        rationale="Strong momentum.",
        technical=tech,
    )


@pytest.fixture
def sample_risk():
    return RiskReport(
        symbol="BTC/USDT",
        timestamp=datetime.utcnow(),
        session_id="test-session",
        var_95=0.02,
        var_99=0.035,
        cvar_99=0.045,
        kelly_raw=0.4,
        kelly_fractional=0.10,
        position_size_usd=10_000.0,
        position_size_units=0.154,
        stop_loss_price=64200.0,
        take_profit_price=66800.0,
        max_drawdown_pct=0.035,
        approved=True,
    )


def test_build_obs_correct_shape(sample_decision, sample_risk, sample_prices):
    obs = _build_obs(sample_decision, sample_risk, sample_prices)
    assert obs.shape == (40,)
    assert obs.dtype == np.float32


def test_build_obs_thin_prices(sample_decision, sample_risk):
    thin = np.array([65000.0, 65100.0])
    obs = _build_obs(sample_decision, sample_risk, thin)
    assert obs.shape == (40,)


@pytest.mark.asyncio
async def test_paper_trade_buy_fills(sample_decision, sample_risk, sample_prices):
    with patch("agents.execution.get_bus") as mock_bus:
        mock_bus.return_value = AsyncMock()
        mock_bus.return_value.publish = AsyncMock()

        agent = ExecutionAgent()
        agent._ppo = None  # force TWAP fallback

        order = await agent.execute(sample_decision, sample_risk, sample_prices)

    assert order.status == OrderStatus.FILLED
    assert order.avg_fill_price is not None
    assert order.avg_fill_price > 0
    assert order.slippage_bps is not None
    assert order.slippage_bps >= 0
    assert order.side == Side.BUY
    assert order.quantity == pytest.approx(sample_risk.position_size_units)


@pytest.mark.asyncio
async def test_paper_trade_small_order_uses_market(sample_decision, sample_risk, sample_prices):
    sample_risk.position_size_usd = 500.0  # small → market order
    sample_risk.position_size_units = 0.0077

    with patch("agents.execution.get_bus") as mock_bus:
        mock_bus.return_value = AsyncMock()
        mock_bus.return_value.publish = AsyncMock()

        agent = ExecutionAgent()
        agent._ppo = None

        order = await agent.execute(sample_decision, sample_risk, sample_prices)

    assert order.status == OrderStatus.FILLED
    assert order.order_type == OrderType.MARKET


@pytest.mark.asyncio
async def test_paper_trade_large_order_uses_twap(sample_decision, sample_risk, sample_prices):
    sample_risk.position_size_usd = 50_000.0
    sample_risk.position_size_units = 0.77

    with patch("agents.execution.get_bus") as mock_bus:
        mock_bus.return_value = AsyncMock()
        mock_bus.return_value.publish = AsyncMock()

        agent = ExecutionAgent()
        agent._ppo = None

        order = await agent.execute(sample_decision, sample_risk, sample_prices)

    assert order.order_type == OrderType.TWAP
