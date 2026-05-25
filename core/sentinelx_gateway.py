"""
Sentinel-X Gateway: Async HTTP client for the Go API Gateway + OMS.

Sends trade signals from the Python supervisor to the Go gateway
which handles risk validation (via Rust) and order execution.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

# ── Shared HTTP client ─────────────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient | None:
    """Return cached HTTPX client, or None if gateway addr is not set."""
    global _client
    if _client is not None:
        return _client

    cfg = get_settings()
    if not cfg.sentinelx_gateway_url:
        logger.info("sentinelx_gateway_url not set — Go gateway disabled")
        return None

    _client = httpx.AsyncClient(
        base_url=cfg.sentinelx_gateway_url,
        timeout=cfg.sentinelx_timeout_s,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
    )
    logger.info("Go Gateway client ready at %s", cfg.sentinelx_gateway_url)
    return _client


async def close() -> None:
    """Close HTTP client (call on shutdown)."""
    global _client
    if _client:
        await _client.aclose()
    _client = None


# ── Result type ────────────────────────────────────────────────────────────────
@dataclass
class GatewayResult:
    approved: bool
    order_id: str | None = None
    status: str | None = None
    reason: str | None = None


# ── Public API ─────────────────────────────────────────────────────────────────
async def submit_trade(
    session_id: str,
    symbol: str,
    side: str,  # "BUY" | "SELL"
    current_price: float,
    consensus_score: float,
    atr_14: float | None = None,
    return_series: list[float] | None = None,
    rationale: str = "",
) -> GatewayResult:
    """Submit a trade signal to the Go Gateway for execution.

    Returns a GatewayResult with approval status. If the gateway is
    unavailable, returns a rejected result so the caller can fall back
    to its local execution engine.
    """
    client = await _get_client()
    if client is None:
        return GatewayResult(approved=False, reason="Gateway not configured")

    payload: dict[str, Any] = {
        "session_id": session_id,
        "symbol": symbol,
        "side": side,
        "current_price": current_price,
        "consensus_score": consensus_score,
        "confidence_pct": max(consensus_score * 100.0, 50.0),
        "atr_14": atr_14 or current_price * 0.02,
        "rationale": rationale,
        "return_series": return_series or [],
    }

    try:
        resp = await client.post("/v1/trade", json=payload)
        if resp.status_code != 200:
            logger.warning("Gateway returned %d: %s", resp.status_code, resp.text[:200])
            return GatewayResult(approved=False, reason=f"HTTP {resp.status_code}")

        data = resp.json()
        approved = data.get("approved", False)
        order_data = data.get("order", {})

        logger.info(
            "Gateway trade %s for %s %s",
            "APPROVED" if approved else "REJECTED",
            symbol,
            side,
        )
        return GatewayResult(
            approved=approved,
            order_id=order_data.get("OrderId") or order_data.get("order_id"),
            status=order_data.get("Status") or order_data.get("status"),
            reason=data.get("reason"),
        )

    except httpx.RequestError as exc:
        logger.warning("Gateway request failed for %s %s: %s", symbol, side, exc)
        return GatewayResult(approved=False, reason=f"Gateway unreachable: {exc}")


async def health_check() -> bool:
    """Check if the Go gateway is healthy."""
    client = await _get_client()
    if client is None:
        return False
    try:
        resp = await client.get("/v1/health")
        data = resp.json()
        return data.get("healthy", False)
    except Exception:
        return False
