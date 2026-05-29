"""Unit tests for STL Protocol logic in council.py (mock NIM calls)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.council import CouncilAgent, _build_market_context
from agents.stl_protocol import STLResult
from core.models import Side, TechnicalSignal
from datetime import datetime


@pytest.fixture
def mock_technical():
    return TechnicalSignal(
        symbol="BTC/USDT",
        timestamp=datetime.utcnow(),
        rsi_14=55.0,
        ema_fast=65000.0,
        ema_slow=64000.0,
        atr_14=500.0,
        macd=200.0,
        macd_signal=150.0,
        bb_upper=67000.0,
        bb_lower=63000.0,
        volume_ratio=1.2,
        trend=Side.BUY,
        confidence=0.7,
    )


def test_build_market_context_includes_symbol(mock_technical):
    ctx = _build_market_context("BTC/USDT", mock_technical, None, None)
    assert "BTC/USDT" in ctx
    assert "RSI=55.0" in ctx
    assert "EMA_fast=65000.00" in ctx


def test_build_market_context_no_signals():
    ctx = _build_market_context("ETH/USDT", None, None, None)
    assert "ETH/USDT" in ctx


@pytest.mark.asyncio
async def test_council_deliberate_buy(mock_technical):
    bull_response = {
        "position": "BUY",
        "argument": "Strong momentum with EMA crossover.",
        "quantitative_score": 0.75,
        "supporting_factors": ["EMA crossover", "RSI neutral"],
        "risk_factors": ["High volatility"],
    }
    bear_response = {
        "position": "SELL",
        "argument": "Overbought conditions loom.",
        "quantitative_score": 0.35,
        "supporting_factors": ["Near resistance"],
        "risk_factors": ["EMA still bullish"],
    }
    synthesis_response = {
        "bull_score": 0.70,
        "bear_score": 0.30,
        "consensus_score": 0.40,
        "final_side": "BUY",
        "rationale": "Momentum favors bulls with moderate consensus.",
    }

    call_count = 0

    async def mock_nim_json(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bull_response
        elif call_count == 2:
            return bear_response
        return synthesis_response

    async def mock_stl(**kwargs):
        return STLResult(
            bull_score=0.70,
            bear_score=0.30,
            consensus_score=0.40,
            confidence_pct=85.0,
            final_side="BUY",
            blocked=False,
            block_reason="",
            agent_weights={},
            contradictions=[],
            rationale="Momentum favors bulls with moderate consensus.",
        )

    with patch("agents.council.llm_json", side_effect=mock_nim_json), \
         patch("agents.council.run_stl_protocol", side_effect=mock_stl), \
         patch("agents.council.get_bus") as mock_bus:
        mock_bus.return_value = AsyncMock()
        mock_bus.return_value.publish = AsyncMock()

        agent = CouncilAgent()
        decision = await agent.deliberate("BTC/USDT", technical=mock_technical)

    assert decision.final_side == Side.BUY
    assert decision.bull_score == pytest.approx(0.70)
    assert decision.bear_score == pytest.approx(0.30)
    assert len(decision.debate_log) == 2
    assert decision.symbol == "BTC/USDT"


@pytest.mark.asyncio
async def test_council_low_consensus_hold():
    synthesis_response = {
        "bull_score": 0.51,
        "bear_score": 0.49,
        "consensus_score": 0.02,
        "final_side": "HOLD",
        "rationale": "Insufficient conviction to trade.",
    }

    async def mock_nim_json(messages, **kwargs):
        return {"position": "BUY", "argument": "x", "quantitative_score": 0.5,
                "supporting_factors": [], "risk_factors": []}

    async def mock_synthesis(messages, **kwargs):
        return synthesis_response

    call_count = 0

    async def side_effect(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {"position": "BUY" if call_count == 1 else "SELL",
                    "argument": "x", "quantitative_score": 0.5,
                    "supporting_factors": [], "risk_factors": []}
        return synthesis_response

    with patch("agents.council.llm_json", side_effect=side_effect), \
         patch("agents.council.get_bus") as mock_bus:
        mock_bus.return_value = AsyncMock()
        mock_bus.return_value.publish = AsyncMock()

        agent = CouncilAgent()
        decision = await agent.deliberate("ETH/USDT")

    assert decision.final_side == Side.HOLD
    assert decision.consensus_score < 0.1
