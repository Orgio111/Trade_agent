"""Prometheus metrics registry — single source of truth for all metrics."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# ── Agent latency ─────────────────────────────────────────────────────────────
AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Per-agent processing latency",
    ["agent"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ── Signals ───────────────────────────────────────────────────────────────────
SIGNAL_COUNTER = Counter(
    "trading_signals_total",
    "Total signals emitted",
    ["agent", "symbol", "side"],
)

COUNCIL_CONSENSUS = Histogram(
    "council_consensus_score",
    "Distribution of council consensus scores",
    ["symbol", "side"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# ── Risk ──────────────────────────────────────────────────────────────────────
VAR_GAUGE = Gauge(
    "risk_var",
    "Value at Risk",
    ["symbol", "confidence"],
)

# ── Execution ─────────────────────────────────────────────────────────────────
ORDER_COUNTER = Counter(
    "orders_total",
    "Total orders placed",
    ["symbol", "side", "status"],
)

SLIPPAGE_HIST = Histogram(
    "execution_slippage_bps",
    "Order execution slippage in basis points",
    ["symbol"],
    buckets=(0, 1, 2, 5, 10, 20, 50, 100),
)

# ── Portfolio ─────────────────────────────────────────────────────────────────
PNL_GAUGE = Gauge(
    "portfolio_pnl_usd",
    "Real-time portfolio PnL in USD",
    ["symbol"],
)

DRAWDOWN_GAUGE = Gauge(
    "portfolio_drawdown_pct",
    "Current portfolio drawdown percentage",
)

EQUITY_GAUGE = Gauge(
    "portfolio_equity_usd",
    "Total portfolio equity in USD",
)

# ── Market data ───────────────────────────────────────────────────────────────
TICK_COUNTER = Counter(
    "market_ticks_total",
    "WebSocket ticks received",
    ["symbol"],
)

WS_RECONNECTS = Counter(
    "websocket_reconnects_total",
    "WebSocket reconnection count",
    ["symbol"],
)

# ── MLOps ─────────────────────────────────────────────────────────────────────
PSI_GAUGE = Gauge(
    "mlops_psi_score",
    "Population Stability Index for model drift",
    ["model"],
)

RETRAIN_COUNTER = Counter(
    "mlops_retrains_total",
    "Number of model retraining events triggered",
    ["model"],
)
