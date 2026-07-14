"""
Dashboard Service — Real-time trading dashboard with WebSocket streaming.

Provides:
- Real-time WebSocket server for frontend
- Multi-asset dashboard (Crypto, Forex, Stocks)
- Level 2 orderbook visualization (heatmap)
- AI explanation panel
- RL performance graphs
- PnL tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES FOR DASHBOARD
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    ws_path: str = "/ws"
    max_history: int = 1000


@dataclass
class AssetDashboardData:
    """Data for a single asset on dashboard."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float

    # Technical
    rsi: float = 50.0
    macd: float = 0.0
    bb_position: float = 0.5

    # Orderbook
    bid: float = 0.0
    ask: float = 0.0
    spread_bps: float = 0.0
    order_flow_imbalance: float = 0.0

    # AI signals
    ai_signal: str = "HOLD"
    ai_confidence: float = 0.0
    ai_reasoning: list[str] = field(default_factory=list)

    # Risk
    position_size: float = 0.0
    unrealized_pnl: float = 0.0

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_24h": self.change_24h,
            "volume_24h": self.volume_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "rsi": self.rsi,
            "macd": self.macd,
            "bb_position": self.bb_position,
            "bid": self.bid,
            "ask": self.ask,
            "spread_bps": self.spread_bps,
            "order_flow_imbalance": self.order_flow_imbalance,
            "ai_signal": self.ai_signal,
            "ai_confidence": self.ai_confidence,
            "ai_reasoning": self.ai_reasoning,
            "position_size": self.position_size,
            "unrealized_pnl": self.unrealized_pnl,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioDashboardData:
    """Portfolio-level dashboard data."""
    total_value: float
    cash: float
    daily_pnl: float
    total_pnl: float
    daily_pnl_pct: float
    total_pnl_pct: float
    drawdown: float
    max_drawdown: float
    sharpe: float = 0.0
    positions: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_pnl_pct": self.total_pnl_pct,
            "drawdown": self.drawdown,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "positions": self.positions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RLPerformanceData:
    """RL performance for dashboard."""
    total_reward: float
    episode: int
    strategy_allocations: dict[str, float]
    strategy_performance: dict[str, dict]
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)


@dataclass
class AIExplanation:
    """AI decision explanation."""
    trade_id: str
    symbol: str
    signal: str
    confidence: float
    reasoning: list[str]
    risk_factors: list[str]
    alternative_signals: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_factors": self.risk_factors,
            "alternative_signals": self.alternative_signals,
            "timestamp": self.timestamp.isoformat(),
        }


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.subscriptions: dict[str, set[str]] = {}  # client_id -> set of channels

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        logger.info(f"[Dashboard] Client connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.subscriptions:
            del self.subscriptions[client_id]
        logger.info(f"[Dashboard] Client disconnected: {client_id}")

    def subscribe(self, client_id: str, channel: str):
        """Subscribe client to a channel."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].add(channel)

    def unsubscribe(self, client_id: str, channel: str):
        """Unsubscribe client from a channel."""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].discard(channel)

    async def broadcast(self, channel: str, message: dict):
        """Broadcast message to all subscribers of a channel."""
        if not self.active_connections:
            return

        message_str = json.dumps({"channel": channel, "data": message})
        disconnected = []

        for client_id, ws in self.active_connections.items():
            if channel in self.subscriptions.get(client_id, set()):
                try:
                    await ws.send_text(message_str)
                except Exception as e:
                    logger.warning(f"[Dashboard] Send error to {client_id}: {e}")
                    disconnected.append(client_id)

        # Clean up disconnected
        for client_id in disconnected:
            self.disconnect(client_id)

    async def send_personal(self, client_id: str, message: dict):
        """Send message to specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"[Dashboard] Personal send error: {e}")
                self.disconnect(client_id)


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD STATE MANAGER
# ═══════════════════════════════════════════════════════════════════

class DashboardState:
    """Manages all dashboard state and broadcasts updates."""

    def __init__(self, config: DashboardConfig | None = None):
        self.config = config or DashboardConfig()
        self.manager = ConnectionManager()

        # Asset data
        self.assets: dict[str, AssetDashboardData] = {}
        self.asset_history: dict[str, deque] = {}

        # Portfolio
        self.portfolio: PortfolioDashboardData | None = None
        self.equity_curve: deque = deque(maxlen=self.config.max_history)

        # RL
        self.rl_data: RLPerformanceData | None = None

        # AI Explanations
        self.explanations: deque = deque(maxlen=100)

        # Orderbook heatmaps
        self.orderbook_heatmaps: dict[str, dict] = {}

        # Background task
        self._broadcast_task: asyncio.Task | None = None

    def update_asset(self, data: AssetDashboardData):
        """Update asset data and broadcast."""
        self.assets[data.symbol] = data

        # Store history
        if data.symbol not in self.asset_history:
            self.asset_history[data.symbol] = deque(maxlen=self.config.max_history)
        self.asset_history[data.symbol].append(data.to_dict())

    def update_portfolio(self, data: PortfolioDashboardData):
        """Update portfolio data."""
        self.portfolio = data
        self.equity_curve.append((datetime.now(), data.total_value))

    def update_rl(self, data: RLPerformanceData):
        """Update RL performance data."""
        self.rl_data = data

    def add_explanation(self, explanation: AIExplanation):
        """Add AI explanation."""
        self.explanations.append(explanation.to_dict())

    def update_orderbook_heatmap(self, symbol: str, heatmap_data: dict):
        """Update orderbook heatmap for symbol."""
        self.orderbook_heatmaps[symbol] = {
            **heatmap_data,
            "timestamp": datetime.now().isoformat(),
        }

    async def start_broadcast(self):
        """Start periodic broadcast task."""
        async def broadcast_loop():
            while True:
                try:
                    # Broadcast asset updates
                    for symbol, data in self.assets.items():
                        await self.manager.broadcast("asset", data.to_dict())

                    # Broadcast portfolio
                    if self.portfolio:
                        await self.manager.broadcast("portfolio", self.portfolio.to_dict())

                    # Broadcast RL data
                    if self.rl_data:
                        pass  # Would broadcast RL data

                    # Broadcast orderbook heatmaps
                    for symbol, heatmap in self.orderbook_heatmaps.items():
                        await self.manager.broadcast(f"orderbook_{symbol}", heatmap)

                except Exception as e:
                    logger.error(f"[Dashboard] Broadcast error: {e}")

                await asyncio.sleep(1)  # 1 second updates

        self._broadcast_task = asyncio.create_task(broadcast_loop())

    async def stop_broadcast(self):
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass


# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════

def create_dashboard_app(config: DashboardConfig | None = None) -> FastAPI:
    """Create FastAPI dashboard application."""
    cfg = config or DashboardConfig()

    app = FastAPI(title="AURORA Trading Dashboard", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global state
    state = DashboardState(cfg)

    # HTML Dashboard
    @app.get("/")
    async def dashboard():
        return HTMLResponse(DASHBOARD_HTML)

    # REST endpoints
    @app.get("/api/assets")
    async def get_assets():
        return {s: d.to_dict() for s, d in state.assets.items()}

    @app.get("/api/assets/{symbol}")
    async def get_asset(symbol: str):
        if symbol in state.assets:
            return state.assets[symbol].to_dict()
        return {"error": "Not found"}

    @app.get("/api/assets/{symbol}/history")
    async def get_asset_history(symbol: str, limit: int = 100):
        if symbol in state.asset_history:
            return list(state.asset_history[symbol])[-limit:]
        return []

    @app.get("/api/portfolio")
    async def get_portfolio():
        if state.portfolio:
            return state.portfolio.to_dict()
        return {}

    @app.get("/api/rl")
    async def get_rl():
        if state.rl_data:
            return {
                "total_reward": state.rl_data.total_reward,
                "episode": state.rl_data.episode,
                "allocations": state.rl_data.strategy_allocations,
                "performance": state.rl_data.strategy_performance,
            }
        return {}

    @app.get("/api/explanations")
    async def get_explanations(limit: int = 20):
        return list(state.explanations)[-limit:]

    @app.get("/api/orderbook/{symbol}")
    async def get_orderbook(symbol: str):
        if symbol in state.orderbook_heatmaps:
            return state.orderbook_heatmaps[symbol]
        return {}

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "clients": len(state.manager.active_connections)}

    # WebSocket endpoint
    @app.websocket(cfg.ws_path)
    async def websocket_endpoint(websocket: WebSocket):
        client_id = f"client_{datetime.now().timestamp()}"
        await state.manager.connect(websocket, client_id)

        try:
            # Send initial data
            await websocket.send_text(json.dumps({
                "channel": "init",
                "data": {
                    "assets": {s: d.to_dict() for s, d in state.assets.items()},
                    "portfolio": state.portfolio.to_dict() if state.portfolio else {},
                    "orderbooks": state.orderbook_heatmaps,
                }
            }))

            # Handle messages
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "subscribe":
                    for channel in msg.get("channels", []):
                        state.manager.subscribe(client_id, channel)

                elif msg.get("type") == "unsubscribe":
                    for channel in msg.get("channels", []):
                        state.manager.unsubscribe(client_id, channel)

                elif msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

        except WebSocketDisconnect:
            state.manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"[Dashboard] WS error: {e}")
            state.manager.disconnect(client_id)

    # Store state on app
    app.state.dashboard = state

    @app.on_event("startup")
    async def startup():
        await state.start_broadcast()
        logger.info(f"[Dashboard] Started on {cfg.host}:{cfg.port}")

    @app.on_event("shutdown")
    async def shutdown():
        await state.stop_broadcast()

    return app


# ═══════════════════════════════════════════════════════════════════
# DASHBOARD HTML (Embedded)
# ═══════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AURORA Trading OS v2</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; }
        .header { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d; }
        .header h1 { font-size: 1.5rem; font-weight: 600; }
        .status { display: flex; gap: 16px; font-size: 0.875rem; }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #238636; }
        .main { display: grid; grid-template-columns: 320px 1fr 320px; grid-template-rows: auto 1fr auto; height: calc(100vh - 56px); gap: 8px; padding: 8px; }
        .panel { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
        .panel h3 { font-size: 0.875rem; font-weight: 600; margin-bottom: 8px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
        .left-panel { grid-row: 1 / 4; }
        .center-panel { grid-column: 2; grid-row: 1 / 3; display: flex; flex-direction: column; gap: 8px; }
        .right-panel { grid-row: 1 / 4; }
        .chart-container { flex: 1; min-height: 300px; }
        .asset-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .asset-card { background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; }
        .asset-card h4 { font-size: 0.875rem; font-weight: 600; margin-bottom: 4px; }
        .asset-price { font-size: 1.25rem; font-weight: 600; }
        .asset-change { font-size: 0.75rem; }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .signal-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
        .signal-buy { background: #238636; color: white; }
        .signal-sell { background: #da3633; color: white; }
        .signal-hold { background: #8b949e; color: white; }
        .orderbook-heatmap { height: 200px; background: #0d1117; border-radius: 4px; position: relative; overflow: hidden; }
        .heatmap-row { display: flex; height: 50%; }
        .heatmap-cell { flex: 1; border: 1px solid #30363d; display: flex; align-items: center; justify-content: center; font-size: 0.625rem; }
        .bid-cell { background: rgba(63, 185, 80, 0.1); }
        .ask-cell { background: rgba(248, 81, 73, 0.1); }
        .explanation-list { max-height: 200px; overflow-y: auto; }
        .explanation-item { padding: 8px; border-bottom: 1px solid #30363d; font-size: 0.75rem; }
        .explanation-item:last-child { border-bottom: none; }
        .explanation-signal { font-weight: 600; }
        .orderbook-side { display: flex; flex-direction: column; gap: 2px; font-size: 0.7rem; }
        .orderbook-level { display: flex; justify-content: space-between; }
        .orderbook-bid { color: #3fb950; }
        .orderbook-ask { color: #f85149; }
        .portfolio-summary { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .metric { background: #0d1117; padding: 8px; border-radius: 4px; }
        .metric-label { font-size: 0.625rem; color: #8b949e; }
        .metric-value { font-weight: 600; font-size: 0.875rem; }
        .rl-allocation { display: flex; flex-direction: column; gap: 4px; }
        .rl-strategy { display: flex; justify-content: space-between; font-size: 0.75rem; }
        .rl-bar { height: 8px; background: #30363d; border-radius: 4px; overflow: hidden; }
        .rl-bar-fill { height: 100%; background: #58a6ff; border-radius: 4px; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AURORA Trading OS v2</h1>
        <div class="status">
            <span><span class="status-dot"></span> Connected</span>
            <span id="ws-status">WS: Connecting...</span>
        </div>
    </div>

    <div class="main">
        <!-- LEFT PANEL -->
        <div class="left-panel panel">
            <h3>Portfolio</h3>
            <div class="portfolio-summary">
                <div class="metric">
                    <div class="metric-label">Total Value</div>
                    <div class="metric-value" id="total-value">$0.00</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Daily PnL</div>
                    <div class="metric-value" id="daily-pnl">$0.00</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total PnL</div>
                    <div class="metric-value" id="total-pnl">$0.00</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Drawdown</div>
                    <div class="metric-value" id="drawdown">0.00%</div>
                </div>
            </div>

            <h3 style="margin-top: 16px;">Positions</h3>
            <div id="positions"></div>

            <h3 style="margin-top: 16px;">AI Explanations</h3>
            <div class="explanation-list" id="explanations"></div>
        </div>

        <!-- CENTER PANEL -->
        <div class="center-panel">
            <div class="panel chart-container">
                <h3>BTCUSDT Chart</h3>
                <div id="chart" style="width: 100%; height: 100%;"></div>
            </div>

            <div class="panel" style="flex: 0 0 220px;">
                <h3>Order Book (BTCUSDT)</h3>
                <div class="orderbook-heatmap" id="orderbook-heatmap"></div>
                <div style="display: flex; gap: 16px; margin-top: 8px;">
                    <div class="orderbook-side">
                        <div style="font-weight: 600; color: #3fb950;">BIDS</div>
                        <div id="orderbook-bids"></div>
                    </div>
                    <div class="orderbook-side">
                        <div style="font-weight: 600; color: #f85149;">ASKS</div>
                        <div id="orderbook-asks"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- RIGHT PANEL -->
        <div class="right-panel panel">
            <h3>Assets</h3>
            <div class="asset-grid" id="assets"></div>

            <h3 style="margin-top: 16px;">RL Allocation</h3>
            <div class="rl-allocation" id="rl-allocation"></div>

            <h3 style="margin-top: 16px;">RL Performance</h3>
            <div class="metric">
                <div class="metric-label">Total Reward</div>
                <div class="metric-value" id="rl-reward">0.00</div>
            </div>
            <div class="metric">
                <div class="metric-label">Episode</div>
                <div class="metric-value" id="rl-episode">0</div>
            </div>
            <div class="metric">
                <div class="metric-label">Sharpe</div>
                <div class="metric-value" id="rl-sharpe">0.00</div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        let chart = null;

        ws.onopen = () => {
            document.getElementById('ws-status').textContent = 'WS: Connected';
            ws.send(JSON.stringify({
                type: 'subscribe',
                channels: ['asset', 'portfolio', 'orderbook_BTCUSDT']
            }));
        };

        ws.onclose = () => {
            document.getElementById('ws-status').textContent = 'WS: Disconnected';
            setTimeout(() => location.reload(), 5000);
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            const channel = msg.channel;
            const data = msg.data;

            if (channel === 'init') {
                initDashboard(data);
            } else if (channel === 'asset') {
                updateAsset(data);
            } else if (channel === 'portfolio') {
                updatePortfolio(data);
            } else if (channel === 'orderbook_BTCUSDT') {
                updateOrderbook(data);
            }
        };

        function initDashboard(data) {
            if (data.assets) {
                Object.values(data.assets).forEach(updateAsset);
            }
            if (data.portfolio) {
                updatePortfolio(data.portfolio);
            }
            if (data.orderbooks) {
                Object.entries(data.orderbooks).forEach(([symbol, ob]) => updateOrderbook(ob));
            }
            initChart();
        }

        function initChart() {
            const chartElement = document.getElementById('chart');
            chart = LightweightCharts.createChart(chartElement, {
                width: chartElement.clientWidth,
                height: 400,
                layout: { background: { color: '#0d1117' }, textColor: '#c9d1d9' },
                grid: { vertLines: { color: '#30363d' }, horLines: { color: '#30363d' } },
                rightPriceScale: { borderColor: '#30363d' },
                timeScale: { borderColor: '#30363d', timeVisible: true },
            });
            candleSeries = chart.addCandlestickSeries({
                upColor: '#3fb950', downColor: '#f85149',
                borderUpColor: '#3fb950', borderDownColor: '#f85149',
                wickUpColor: '#3fb950', wickDownColor: '#f85149',
            });
        }

        function updateAsset(data) {
            const container = document.getElementById('assets');
            let card = document.getElementById(`asset-${data.symbol}`);

            if (!card) {
                card = document.createElement('div');
                card.id = `asset-${data.symbol}`;
                card.className = 'asset-card';
                container.appendChild(card);
            }

            const changeClass = data.change_24h >= 0 ? 'positive' : 'negative';
            const signalClass = data.ai_signal === 'BUY' ? 'signal-buy' : 
                              data.ai_signal === 'SELL' ? 'signal-sell' : 'signal-hold';

            card.innerHTML = `
                <h4>${data.symbol}</h4>
                <div class="asset-price">${data.price.toFixed(2)}</div>
                <div class="asset-change ${changeClass}">${data.change_24h >= 0 ? '+' : ''}${data.change_24h.toFixed(2)}%</div>
                <div style="margin-top: 8px; display: flex; justify-content: space-between; font-size: 0.7rem;">
                    <span>RSI: ${data.rsi.toFixed(1)}</span>
                    <span class="${signalClass} signal-badge">${data.ai_signal} (${(data.ai_confidence*100).toFixed(0)}%)</span>
                </div>
            `;
        }

        function updatePortfolio(data) {
            document.getElementById('total-value').textContent = `$${data.total_value.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('daily-pnl').textContent = `$${data.daily_pnl.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('daily-pnl').className = 'metric-value ' + (data.daily_pnl >= 0 ? 'positive' : 'negative');
            document.getElementById('total-pnl').textContent = `$${data.total_pnl.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById('total-pnl').className = 'metric-value ' + (data.total_pnl >= 0 ? 'positive' : 'negative');
            document.getElementById('drawdown').textContent = `${(data.drawdown*100).toFixed(2)}%`;

            // Update positions
            const posContainer = document.getElementById('positions');
            if (data.positions && data.positions.length > 0) {
                posContainer.innerHTML = data.positions.map(p => `
                    <div style="display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.75rem;">
                        <span>${p.symbol}</span>
                        <span>${p.side} ${p.size}</span>
                        <span class="${p.unrealized_pnl >= 0 ? 'positive' : 'negative'}">${p.unrealized_pnl.toFixed(2)}</span>
                    </div>
                `).join('');
            }
        }

        function updateAssetCandle(data) {
            if (!candleSeries) return;
            candleSeries.update({
                time: Math.floor(new Date(data.timestamp).getTime() / 1000),
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
            });
        }

        function updateOrderbook(data) {
            // Update heatmap
            const heatmap = document.getElementById('orderbook-heatmap');
            if (data.bids && data.asks) {
                const maxSize = Math.max(
                    ...data.bids.map(b => b[1]),
                    ...data.asks.map(a => a[1])
                );
                heatmap.innerHTML = `
                    <div class="heatmap-row">
                        ${data.asks.slice(0, 10).reverse().map(([price, size]) => 
                            `<div class="heatmap-cell ask-cell" style="width: ${(size/maxSize)*100}%">${price.toFixed(2)}</div>`
                        ).join('')}
                    </div>
                    <div class="heatmap-row">
                        ${data.bids.slice(0, 10).map(([price, size]) => 
                            `<div class="heatmap-cell bid-cell" style="width: ${(size/maxSize)*100}%">${price.toFixed(2)}</div>`
                        ).join('')}
                    </div>
                `;
            }

            // Update orderbook lists
            const bidsEl = document.getElementById('orderbook-bids');
            const asksEl = document.getElementById('orderbook-asks');

            if (data.bids) {
                bidsEl.innerHTML = data.bids.slice(0, 10).map(([price, size]) => 
                    `<div class="orderbook-level"><span class="orderbook-bid">${price.toFixed(2)}</span><span>${size.toFixed(4)}</span></div>`
                ).join('');
            }
            if (data.asks) {
                asksEl.innerHTML = data.asks.slice(0, 10).map(([price, size]) => 
                    `<div class="orderbook-level"><span class="orderbook-ask">${price.toFixed(2)}</span><span>${size.toFixed(4)}</span></div>`
                ).join('');
            }
        }
    </script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════

def create_dashboard_service(config: DashboardConfig | None = None) -> tuple[FastAPI, DashboardState]:
    """Create dashboard app and state."""
    app = create_dashboard_app(config)
    return app, app.state.dashboard


def run_dashboard(config: DashboardConfig | None = None):
    """Run dashboard server."""
    cfg = config or DashboardConfig()
    app, _ = create_dashboard_service(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    import asyncio

    async def test():
        app, state = create_dashboard_service()
        print("Dashboard app created")
        print(f"Routes: {[r.path for r in app.routes]}")

    asyncio.run(test())