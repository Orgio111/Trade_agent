"""FastAPI web server — serves the live trading dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import get_settings
from core.models import Order, OrderStatus, PortfolioState, Side

logger = logging.getLogger(__name__)

# ── In-memory store ───────────────────────────────────────────────────────────
# Shared between the trading system and the web dashboard.
class DashboardState:
    """Holds the latest snapshot for the dashboard to serve."""

    def __init__(self) -> None:
        self.portfolio = PortfolioState(equity=100_000.0, cash=100_000.0)
        self.recent_orders: list[dict] = []
        self.agent_signals: dict[str, dict] = {}
        self.risk_metrics: dict[str, dict] = {}
        self.market_prices: dict[str, float] = {}
        self.last_update: float = time.time()
        self._lock = asyncio.Lock()

    async def update_portfolio(self, state: PortfolioState) -> None:
        async with self._lock:
            self.portfolio = state
            self.last_update = time.time()

    async def add_order(self, order: Order) -> None:
        async with self._lock:
            self.recent_orders.insert(0, {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": order.avg_fill_price,
                "status": order.status.value,
                "created_at": order.created_at.isoformat() if order.created_at else "",
                "type": order.order_type.value,
            })
            if len(self.recent_orders) > 100:
                self.recent_orders.pop()

    async def update_agent_signal(self, agent: str, data: dict) -> None:
        async with self._lock:
            self.agent_signals[agent] = {**data, "timestamp": datetime.utcnow().isoformat()}

    async def update_risk(self, symbol: str, metrics: dict) -> None:
        async with self._lock:
            self.risk_metrics[symbol] = {**metrics, "timestamp": datetime.utcnow().isoformat()}

    async def update_price(self, symbol: str, price: float) -> None:
        async with self._lock:
            self.market_prices[symbol] = price

    async def snapshot(self) -> dict:
        async with self._lock:
            eq = self.portfolio.equity
            peak = self.portfolio.peak_equity or eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            return {
                "portfolio": {
                    "equity": round(eq, 2),
                    "cash": round(self.portfolio.cash, 2),
                    "daily_pnl": round(self.portfolio.daily_pnl, 2),
                    "daily_pnl_pct": round(self.portfolio.daily_pnl_pct * 100, 4),
                    "drawdown_pct": round(max(dd, self.portfolio.current_drawdown_pct * 100), 4),
                    "peak_equity": round(peak, 2),
                    "kill_switch": self.portfolio.kill_switch_active,
                    "positions": self.portfolio.positions,
                },
                "orders": self.recent_orders[:50],
                "agents": dict(self.agent_signals),
                "risk": dict(self.risk_metrics),
                "prices": dict(self.market_prices),
                "updated_at": datetime.utcnow().isoformat(),
            }


# ── Singleton ─────────────────────────────────────────────────────────────────
_state: DashboardState | None = None


def get_dashboard_state() -> DashboardState:
    global _state
    if _state is None:
        _state = DashboardState()
    return _state


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Trade Agent Dashboard", version="0.1.0")

# ── Static files ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_STATIC = _HERE / "static"
_TEMPLATES = _HERE / "templates"

_STATIC.mkdir(exist_ok=True)
_TEMPLATES.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the main dashboard HTML."""
    html_path = _TEMPLATES / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard template not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/snapshot")
async def api_snapshot():
    """JSON snapshot of the current dashboard state."""
    ds = get_dashboard_state()
    return JSONResponse(await ds.snapshot())


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/config")
async def api_config():
    """Expose non-sensitive config for the dashboard."""
    cfg = get_settings()
    return {
        "exchange": cfg.exchange,
        "symbols": cfg.symbols,
        "paper_trading": cfg.paper_trading,
        "initial_capital": cfg.initial_capital,
        "min_consensus_score": cfg.min_consensus_score,
        "max_daily_drawdown_pct": cfg.max_daily_drawdown_pct,
        "kelly_fraction": cfg.kelly_fraction,
    }


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push live updates every 1-2 seconds while a client is connected."""
    await ws.accept()
    ds = get_dashboard_state()
    try:
        while True:
            data = await ds.snapshot()
            await ws.send_json(data)
            await asyncio.sleep(1.5)
            # Wait for pings so we detect disconnects quickly
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket client disconnected")
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)


