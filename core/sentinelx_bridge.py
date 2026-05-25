"""
Sentinel-X Bridge: Python gRPC client for the Rust Risk Engine.

Wraps the Rust gRPC service with:
  - Async connection pooling (lazy singleton)
  - Automatic fallback to pure-Python risk engine on gRPC failure
  - Prometheus latency tracking
  - Timeout & retry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import grpc
import numpy as np

from core.config import get_settings
from core.models import CouncilDecision, RiskReport, Side
from core.observability import AGENT_LATENCY, VAR_GAUGE
from core.proto import risk_pb2 as pb
from core.proto import risk_pb2_grpc as pb_grpc

logger = logging.getLogger(__name__)

# ── In-memory gRPC channel cache ──────────────────────────────────────────────
_channel: grpc.aio.Channel | None = None
_stub: pb_grpc.RiskEngineStub | None = None


async def _get_stub() -> pb_grpc.RiskEngineStub | None:
    """Return a cached gRPC stub, or None if the bridge is disabled."""
    global _channel, _stub
    if _stub is not None:
        return _stub

    cfg = get_settings()
    if not cfg.sentinelx_risk_addr:
        logger.info("sentinelx_risk_addr not set — Rust bridge disabled")
        return None

    try:
        _channel = grpc.aio.insecure_channel(
            cfg.sentinelx_risk_addr,
            options=[
                ("grpc.keepalive_time_ms", 10_000),
                ("grpc.keepalive_timeout_ms", 5_000),
                ("grpc.max_send_message_length", 4 * 1024 * 1024),
                ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ],
        )
        _stub = pb_grpc.RiskEngineStub(_channel)
        # Quick connectivity check
        await _channel.channel_ready()
        logger.info("Connected to Rust Risk Engine at %s", cfg.sentinelx_risk_addr)
        return _stub
    except Exception:
        logger.warning("Rust Risk Engine at %s unreachable — will use Python fallback", cfg.sentinelx_risk_addr)
        _stub = None
        _channel = None
        return None


async def close() -> None:
    """Close the gRPC channel (call on shutdown)."""
    global _channel, _stub
    if _channel:
        await _channel.close()
    _channel = None
    _stub = None


# ── Result type ────────────────────────────────────────────────────────────────
@dataclass
class RustRiskResult:
    approved: bool
    var_95: float
    var_99: float
    cvar_99: float
    kelly_raw: float
    kelly_fractional: float
    position_size_usd: float
    position_size_units: float
    stop_loss_price: float
    take_profit_price: float
    portfolio_heat: float
    rejection_reason: str | None = None


# ── Public API ─────────────────────────────────────────────────────────────────
async def validate_trade(
    decision: CouncilDecision,
    returns: np.ndarray,
    current_price: float,
    portfolio_equity: float,
    atr_14: float | None = None,
) -> RustRiskResult | None:
    """Validate a trade through the Rust Risk Engine.

    Returns ``None`` if the bridge is unavailable — *not* a rejection.
    Callers must fall back to their local Python risk engine.
    """
    stub = await _get_stub()
    if stub is None:
        return None

    cfg = get_settings()
    side_str = "BUY" if decision.final_side == Side.BUY else "SELL"

    req = pb.RiskRequest(
        session_id=decision.session_id,
        symbol=decision.symbol,
        side=side_str,
        proposed_qty=0.0,  # engine computes from Kelly
        current_price=current_price,
        portfolio_equity=portfolio_equity,
        atr_14=atr_14 or current_price * 0.02,
        consensus_score=decision.consensus_score,
        return_series=returns.tolist(),
    )

    try:
        with AGENT_LATENCY.labels(agent="sentinelx_risk").time():
            resp = await stub.Validate(req, timeout=cfg.sentinelx_timeout_s)

        VAR_GAUGE.labels(symbol=decision.symbol, confidence="rust_99").set(resp.var_99)
        VAR_GAUGE.labels(symbol=decision.symbol, confidence="rust_95").set(resp.var_95)

        if not resp.approved:
            logger.info("Rust risk REJECTED %s: %s", decision.symbol, resp.rejection_reason)

        return RustRiskResult(
            approved=resp.approved,
            var_95=resp.var_95,
            var_99=resp.var_99,
            cvar_99=resp.cvar_99,
            kelly_raw=resp.kelly_raw,
            kelly_fractional=resp.kelly_fractional,
            position_size_usd=resp.position_size_usd,
            position_size_units=resp.position_size_units,
            stop_loss_price=resp.stop_loss_price,
            take_profit_price=resp.take_profit_price,
            portfolio_heat=resp.portfolio_heat,
            rejection_reason=resp.rejection_reason if not resp.approved else None,
        )
    except grpc.RpcError as exc:
        logger.warning("Rust risk RPC failed for %s: %s — falling back", decision.symbol, exc)
        return None


async def subscribe_kill_switch(
    subscriber_id: str = "python-main",
) -> grpc.aio.UnaryStreamCall:
    """Subscribe to kill-switch events from the Rust engine.

    Returns an async iterator of ``KillSwitchEvent`` protos.
    """
    stub = await _get_stub()
    if stub is None:
        return None

    req = pb.KillSwitchRequest(subscriber_id=subscriber_id)
    return stub.SubscribeKillSwitch(req)
