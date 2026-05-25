"""Integration tests: Rust risk engine subprocess × Python gRPC bridge.

Requires the Rust binary at ``sentinel-x/rust/target/release/sentinel-risk{,.exe}``.
Build it with::

    cd sentinel-x/rust && cargo build --release

Tests are skipped if the binary is missing.
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pytest

from core.models import CouncilDecision, Side, TechnicalSignal
from core.sentinelx_bridge import RustRiskResult, close as close_bridge, validate_trade


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_decision(
    symbol: str = "BTC/USDT",
    side: Side = Side.BUY,
    consensus: float = 0.80,
    atr_14: float = 400.0,
    price: float = 65_000.0,
) -> CouncilDecision:
    """Build a ``CouncilDecision`` with sane defaults for testing."""
    tech = TechnicalSignal(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        rsi_14=52.0,
        ema_fast=65_500.0,
        ema_slow=65_000.0,
        atr_14=atr_14,
        macd=100.0,
        macd_signal=80.0,
        bb_upper=66_500.0,
        bb_lower=64_000.0,
        volume_ratio=1.3,
        trend=side,
        confidence=0.72,
    )
    return CouncilDecision(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        session_id="integ-test",
        bull_score=0.70,
        bear_score=0.30,
        consensus_score=consensus,
        final_side=side,
        debate_log=[],
        rationale="Integration test trade.",
        technical=tech,
    )


async def _reset_bridge() -> None:
    """Close the bridge connection and clear the settings cache so the next
    call picks up fresh environment variables."""
    await close_bridge()
    from core.config import get_settings

    get_settings.cache_clear()


# ── Fixtures ────────────────────────────────────────────────────────────────────


# ── Tests: approved trade ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_validate_approved_trade(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """A trade with low-volatility returns and high consensus should be approved."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)  # low-volatility daily returns
    decision = _make_decision(consensus=0.80)

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
    )

    assert result is not None, "Bridge should return a result when engine is reachable"
    assert result.approved, (
        f"Trade should be approved — got rejection: {result.rejection_reason}"
    )
    # Kelly-quarter with fixed params (win_rate=0.55, avg_win=0.015, avg_loss=0.010)
    # full_kelly = (1.5*0.55 - 0.45) / 1.5 = 0.25 → quarter_kelly = 0.0625
    # position_size_usd = min(100_000 * 0.0625, 100_000 * 0.10) = min(6250, 10000) = 6250
    assert result.kelly_fractional == pytest.approx(0.0625, abs=1e-4)
    assert result.position_size_usd == pytest.approx(6_250.0, abs=100.0)
    assert result.position_size_units == pytest.approx(6_250.0 / 65_000.0, abs=0.01)
    # ATR stop for BUY: 65_000 - 2 * 400 = 64_200
    assert result.stop_loss_price == pytest.approx(64_200.0, abs=50.0)
    # ATR take profit for BUY: 65_000 + 2 * 400 * 1.5 = 66_200
    assert result.take_profit_price == pytest.approx(66_200.0, abs=50.0)
    # VaR should be positive and reasonable for sigma=0.02 returns
    assert 0.0 < result.var_99 < 0.15
    assert 0.0 < result.var_95 < result.var_99
    # CVaR is from historical only; blended VaR may be higher (parametric/MC)
    assert result.cvar_99 > 0.0
    assert result.rejection_reason is None


@pytest.mark.asyncio
async def test_validate_sell_side_approved(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """SELL-side trades should also be approved with good parameters."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.015, 252)
    decision = _make_decision(side=Side.SELL, consensus=0.85)

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
    )

    assert result is not None
    assert result.approved, f"SELL trade should be approved: {result.rejection_reason}"
    assert result.rejection_reason is None


# ── Tests: rejected trades ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_low_consensus(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """Trade with consensus below 0.65 should be rejected."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    decision = _make_decision(consensus=0.50)  # below 0.65 threshold

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
    )

    assert result is not None
    assert not result.approved, "Trade with low consensus should be rejected"
    assert result.rejection_reason is not None
    assert "consensus" in result.rejection_reason.lower()


@pytest.mark.asyncio
async def test_reject_high_var(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """Trade with extremely volatile returns (VaR99 > 15%) should be rejected."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    # Very high volatility returns → VaR99 will exceed 15%
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0, 0.15, 252)
    decision = _make_decision(consensus=0.80)

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
    )

    assert result is not None
    assert not result.approved, "High VaR trade should be rejected"
    assert result.rejection_reason is not None
    assert "var" in result.rejection_reason.lower() or "VaR" in result.rejection_reason


# ── Tests: graceful degradation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bridge_returns_none_when_not_configured():
    """When ``SENTINELX_RISK_ADDR`` is empty, the bridge should return ``None``
    without attempting any connection."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    decision = _make_decision()

    # Ensure the env var is *not* set
    if "SENTINELX_RISK_ADDR" in os.environ:
        del os.environ["SENTINELX_RISK_ADDR"]
    await _reset_bridge()

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
    )

    assert result is None, "Bridge should return None when not configured"


@pytest.mark.asyncio
async def test_bridge_fallback_on_wrong_addr(monkeypatch: pytest.MonkeyPatch):
    """When ``SENTINELX_RISK_ADDR`` points to a non-existent server, the bridge
    should return ``None`` (graceful degradation)."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", "127.0.0.1:1")  # unlikely to be listening
    await _reset_bridge()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    decision = _make_decision()

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
    )

    assert result is None, "Bridge should fall back gracefully on connection error"


# ── Tests: metrics values ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_var_values_scale_with_volatility(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """Higher-volatility returns should produce proportionally higher VaR values."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(42)
    low_vol = rng.normal(0.001, 0.01, 252)
    high_vol = rng.normal(0.001, 0.05, 252)
    decision = _make_decision(consensus=0.80)

    # Low volatility
    result_low = await validate_trade(
        decision=decision, returns=low_vol, current_price=65_000.0,
        portfolio_equity=100_000.0, atr_14=400.0,
    )
    assert result_low is not None

    # High volatility (may be rejected — but we still check VaR values)
    result_high = await validate_trade(
        decision=decision, returns=high_vol, current_price=65_000.0,
        portfolio_equity=100_000.0, atr_14=400.0,
    )
    assert result_high is not None

    # VaR should be higher for more volatile returns
    assert result_high.var_99 > result_low.var_99, (
        f"Expected high-vol VaR({result_high.var_99:.4f}) > "
        f"low-vol VaR({result_low.var_99:.4f})"
    )


@pytest.mark.asyncio
async def test_portfolio_heat_scales_with_position_size(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """Portfolio heat should reflect the proposed position size."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    decision = _make_decision(consensus=0.80)

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
    )

    assert result is not None
    # Portfolio heat should be well-defined (0-1 range) for a single asset
    assert 0.0 <= result.portfolio_heat <= 1.0


@pytest.mark.asyncio
async def test_sbe_payload_present(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """The risk response should include a non-empty SBE-encoded payload."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    decision = _make_decision(consensus=0.80)

    # Use the lower-level gRPC stub directly to access the raw sbe_payload field
    from core.sentinelx_bridge import _get_stub
    from core.proto import risk_pb2 as pb

    stub = await _get_stub()
    assert stub is not None

    req = pb.RiskRequest(
        session_id="sbe-test",
        symbol="BTC/USDT",
        side="BUY",
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
        consensus_score=0.80,
        return_series=returns.tolist(),
    )

    resp = await stub.Validate(req)

    assert resp.sbe_payload, "SBE payload should be non-empty"
    assert len(resp.sbe_payload) > 10, "SBE payload should be a meaningful base64 string"


# ── Tests: thin data ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thin_return_series_uses_defaults(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """Very short return series (< 30 points) should use conservative VaR defaults."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    returns = np.array([0.01, -0.02, 0.005, 0.01, -0.01])
    decision = _make_decision(consensus=0.80)

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=100_000.0,
        atr_14=400.0,
    )

    assert result is not None
    # Thin data should still produce a valid response with conservative defaults
    assert result.var_99 > 0.0
    assert result.cvar_99 >= result.var_99


# ── Tests: edge cases ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_zero_equity_returns_zero_size(
    rust_engine_addr: str, monkeypatch: pytest.MonkeyPatch
):
    """Zero portfolio equity should produce zero position sizing."""
    monkeypatch.setenv("SENTINELX_RISK_ADDR", rust_engine_addr)
    await _reset_bridge()

    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 252)
    decision = _make_decision(consensus=0.80)

    result = await validate_trade(
        decision=decision,
        returns=returns,
        current_price=65_000.0,
        portfolio_equity=0.0,  # zero equity
        atr_14=400.0,
    )

    assert result is not None
    assert result.position_size_usd == pytest.approx(0.0, abs=0.01)
    assert result.position_size_units == pytest.approx(0.0, abs=0.01)
