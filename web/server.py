"""FastAPI web server — serves the live trading dashboard."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import get_settings
from core.models import (
    CouncilDecision,
    FeatureSignal,
    Order,
    OrderStatus,
    PortfolioState,
    Side,
)

try:
    import aiohttp
    _HAVE_AIOHTTP = True
except ImportError:
    _HAVE_AIOHTTP = False

logger = logging.getLogger(__name__)


# ── Log capture ring buffer ───────────────────────────────────────────────────
class LogCaptureHandler(logging.Handler):
    """In-memory ring buffer that captures recent log records for the dashboard."""

    def __init__(self, maxlen: int = 200) -> None:
        super().__init__()
        self.buffer: deque[dict] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "time": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "msg": self.format(record),
            }
            self.buffer.append(entry)
        except Exception:
            pass


_log_handler = LogCaptureHandler()
_log_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_log_handler)


def get_logs(limit: int = 100) -> list[dict]:
    return list(_log_handler.buffer)[-limit:]


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
        self.feature_signals: dict[str, dict] = {}
        self.council_decisions: dict[str, dict] = {}
        self.equity_history: list[dict] = []
        self.last_update: float = time.time()

        # ── Model deployment & inference tracking ────────────────────────
        self.serve_health: dict | None = None
        """Cached Ray Serve /health response."""
        self.deployment_health: dict[str, dict] = {}
        """Per-deployment health statuses, e.g. {"ppo": {"status": "ready", ...}}."""
        self.model_deployment: dict = {
            "version": 0,
            "source": "unknown",
            "local_version": 0,
            "latest_registry_version": 0,
            "model_loaded": False,
        }
        """Current model version, source, and loading state."""
        self.ppo_latency_history: deque = deque(maxlen=200)
        """Rolling buffer of PPO inference latencies (ms)."""
        self.reload_events: deque = deque(maxlen=50)
        """Rolling buffer of auto-reload events."""
        self._lock = asyncio.Lock()

    async def update_portfolio(self, state: PortfolioState) -> None:
        async with self._lock:
            self.portfolio = state
            self.equity_history.append({
                "t": datetime.utcnow().isoformat(),
                "v": round(state.equity, 2),
            })
            if len(self.equity_history) > 500:
                self.equity_history = self.equity_history[-500:]
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

    async def update_features(self, symbol: str, signal: FeatureSignal) -> None:
        async with self._lock:
            self.feature_signals[symbol] = {
                "ofi": signal.ofi,
                "cvd": signal.cvd,
                "cvd_delta": signal.cvd_delta,
                "funding_rate": signal.funding_rate,
                "funding_rate_delta": signal.funding_rate_delta,
                "open_interest": signal.open_interest,
                "open_interest_delta": signal.open_interest_delta,
                "oi_price_delta_corr": signal.oi_price_delta_corr,
                "trade_strength": signal.trade_strength,
                "trend": signal.trend.value,
                "confidence": signal.confidence,
                "timestamp": datetime.utcnow().isoformat(),
            }

    # ── Model deployment tracking ───────────────────────────────────────

    async def update_model_deployment(self, deployment: dict) -> None:
        async with self._lock:
            self.model_deployment.update(deployment)

    async def record_ppo_latency(self, latency_ms: float) -> None:
        async with self._lock:
            self.ppo_latency_history.append({
                "t": datetime.utcnow().isoformat(),
                "v": round(latency_ms, 2),
            })

    async def record_reload_event(self, event: dict) -> None:
        async with self._lock:
            self.reload_events.append({
                **event,
                "timestamp": datetime.utcnow().isoformat(),
            })

    async def update_serve_health(self, health: dict) -> None:
        async with self._lock:
            self.serve_health = health

    async def update_deployment_health(self, name: str, status: dict) -> None:
        async with self._lock:
            self.deployment_health[name] = status

    async def update_council_decision(self, symbol: str, decision: CouncilDecision) -> None:
        async with self._lock:
            self.council_decisions[symbol] = {
                "session_id": decision.session_id,
                "bull_score": decision.bull_score,
                "bear_score": decision.bear_score,
                "consensus_score": decision.consensus_score,
                "final_side": decision.final_side.value,
                "rationale": decision.rationale,
                "debate_log": [
                    {
                        "agent": a.agent_name,
                        "position": a.position.value,
                        "argument": a.argument,
                        "score": a.quantitative_score,
                        "supporting": a.supporting_factors[:3],
                        "risks": a.risk_factors[:3],
                    }
                    for a in decision.debate_log
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

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
                "features": dict(self.feature_signals),
                "council": dict(self.council_decisions),
                "equity_history": list(self.equity_history),
                "deployment": dict(self.model_deployment),
                "serve_health": self.serve_health,
                "deployment_health": dict(self.deployment_health),
                "ppo_latency": list(self.ppo_latency_history),
                "reload_events": list(self.reload_events),
                "logs": get_logs(50),
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


@app.get("/api/serve-status")
async def api_serve_status():
    """Check Ray Serve endpoint health and return deployment statuses."""
    cfg = get_settings()
    serve_url = cfg.ray_serve_url
    result = {
        "configured": bool(serve_url),
        "serve_url": serve_url,
        "reachable": False,
        "deployments": {},
    }

    if serve_url and _HAVE_AIOHTTP:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5),
            ) as session:
                # Check aggregate health
                async with session.get(f"{serve_url}/health") as resp:
                    if resp.ok:
                        health_data = await resp.json()
                        result["reachable"] = True
                        result["cluster_health"] = health_data

                # Check individual deployments
                for dep_name in ["ppo", "llm", "embed", "market"]:
                    try:
                        async with session.get(f"{serve_url}/{dep_name}/health") as dep_resp:
                            if dep_resp.ok:
                                result["deployments"][dep_name] = await dep_resp.json()
                            else:
                                result["deployments"][dep_name] = {"status": "error", "code": dep_resp.status}
                    except Exception as exc:
                        result["deployments"][dep_name] = {"status": "unreachable", "error": str(exc)}

            # Cache in dashboard state
            ds = get_dashboard_state()
            await ds.update_serve_health(result)
        except Exception as exc:
            result["error"] = str(exc)
    elif serve_url and not _HAVE_AIOHTTP:
        result["error"] = "aiohttp not installed — cannot check serve health"

    return JSONResponse(result)


@app.get("/api/models")
async def api_models():
    """List registered models from the model registry."""
    try:
        from mlops.model_registry import list_models
        models = list_models()
        return JSONResponse([{
            "version": m.version,
            "created_at": m.created_at,
            "model_type": m.model_type,
            "train_sharpe": m.train_sharpe,
            "test_sharpe": m.test_sharpe,
            "test_max_drawdown": m.test_max_drawdown,
            "test_win_rate": m.test_win_rate,
            "total_timesteps": m.total_timesteps,
            "test_information_ratio": m.test_information_ratio,
            "test_calmar_ratio": m.test_calmar_ratio,
        } for m in models])
    except ImportError:
        return JSONResponse({"error": "Model registry not available"}, status_code=503)


@app.get("/api/config")
async def api_config():
    """Expose non-sensitive config for the dashboard."""
    cfg = get_settings()
    return {
        "exchange": cfg.exchange,
        "exchanges": cfg.exchanges,
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


