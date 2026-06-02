"""
QUANTEX Prometheus Metrics — Live trading performance monitoring.

Exposes metrics for Prometheus scraping at the /metrics endpoint.
Grafana dashboards consume these for real-time visualization.
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from starlette.responses import Response


# ── Adaptive Routing Metrics ─────────────────────────────

ADAPTIVE_EMA_LATENCY = Gauge(
    "adaptive_ema_latency_ms",
    "EMA-smoothed inference latency in milliseconds per agent→provider pair",
    ["agent", "provider"],
)

ADAPTIVE_P50_LATENCY = Gauge(
    "adaptive_p50_latency_ms",
    "P50 (median) observed inference latency in milliseconds per agent→provider pair",
    ["agent", "provider"],
)

ADAPTIVE_SUCCESS_RATE = Gauge(
    "adaptive_success_rate",
    "Request success rate (0-1) per agent→provider pair",
    ["agent", "provider"],
)

ADAPTIVE_SAMPLES_TOTAL = Gauge(
    "adaptive_samples_total",
    "Total request samples collected per agent→provider pair",
    ["agent", "provider"],
)

ADAPTIVE_CHAIN_ADAPTED = Gauge(
    "adaptive_chain_adapted",
    "Whether the agent's provider chain has been reordered by adaptive routing (1=adapted, 0=base chain)",
    ["agent"],
)

ADAPTIVE_CHAIN_ADAPTATION_EVENTS = Counter(
    "adaptive_chain_adaptation_events_total",
    "Total number of times each agent's provider chain has been reordered",
    ["agent"],
)


# ── Trading Metrics ──────────────────────────────────────

TRADES_TOTAL = Counter(
    "trades_total",
    "Total trades executed",
    ["symbol", "direction", "outcome"],
)

TRADE_PNL = Histogram(
    "trade_pnl_usd",
    "Trade PnL in USD",
    buckets=[-10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, 50],
)

WIN_RATE = Gauge(
    "win_rate_rolling_20",
    "Rolling 20-trade win rate",
)

BALANCE = Gauge(
    "account_balance_usd",
    "Current account balance",
)

EQUITY = Gauge(
    "account_equity_usd",
    "Current account equity (balance + unrealized PnL)",
)

DRAWDOWN = Gauge(
    "current_drawdown_pct",
    "Current drawdown percentage",
)

OPEN_POSITIONS = Gauge(
    "open_positions_count",
    "Number of open positions",
)

CONSECUTIVE_LOSSES = Gauge(
    "consecutive_losses_total",
    "Consecutive losing trades",
)

TOTAL_PNL = Gauge(
    "total_pnl_usd",
    "Total realized PnL",
)

# ── AI Agent Metrics ─────────────────────────────────────

AGENT_INFERENCE_LATENCY = Histogram(
    "agent_inference_ms",
    "Agent inference latency in milliseconds",
    ["agent", "model"],
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000],
)

SWARM_CONSENSUS = Gauge(
    "swarm_consensus_score",
    "Latest swarm consensus confidence score",
)

# ── ML Model Metrics ─────────────────────────────────────

ML_PREDICTION_CONFIDENCE = Gauge(
    "ml_prediction_confidence",
    "ML model prediction confidence",
    ["direction"],
)

ML_MODEL_ACCURACY = Gauge(
    "ml_model_accuracy",
    "ML model training accuracy",
)

ML_MODEL_AGE_HOURS = Gauge(
    "ml_model_age_hours",
    "Hours since last ML model retraining",
)

# ── System Metrics ───────────────────────────────────────

BACKTEST_DURATION = Histogram(
    "backtest_duration_seconds",
    "Backtest execution duration",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

STRATEGY_SIGNALS = Counter(
    "strategy_signals_total",
    "Strategy signals generated",
    ["direction"],
)

ORDER_LATENCY = Histogram(
    "order_execution_ms",
    "Order execution latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500],
)

# ── Metrics Endpoint ─────────────────────────────────────

async def metrics_endpoint():
    """FastAPI-compatible metrics endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; charset=utf-8",
    )


# ── Helper Functions ────────────────────────────────────

def record_trade(symbol: str, direction: str, pnl: float):
    """Record a completed trade in metrics."""
    outcome = "win" if pnl > 0 else "loss"
    TRADES_TOTAL.labels(symbol=symbol, direction=direction, outcome=outcome).inc()
    TRADE_PNL.observe(pnl)


def update_portfolio_metrics(balance: float, equity: float, drawdown: float,
                              open_positions: int, consec_losses: int, total_pnl: float):
    """Update portfolio metrics from account snapshot."""
    BALANCE.set(balance)
    EQUITY.set(equity)
    DRAWDOWN.set(drawdown * 100)  # Convert to percentage
    OPEN_POSITIONS.set(open_positions)
    CONSECUTIVE_LOSSES.set(consec_losses)
    TOTAL_PNL.set(total_pnl)


def record_agent_latency(agent: str, model: str, latency_ms: float):
    """Record agent inference latency."""
    AGENT_INFERENCE_LATENCY.labels(agent=agent, model=model).observe(latency_ms)


def update_ml_metrics(confidence: float, direction: str, accuracy: float, age_hours: float):
    """Update ML model metrics."""
    ML_PREDICTION_CONFIDENCE.labels(direction=direction).set(confidence)
    ML_MODEL_ACCURACY.set(accuracy)
    ML_MODEL_AGE_HOURS.set(age_hours)


def update_adaptive_routing_metrics(agent_router):
    """
    Sync AgentModelRouter adaptive routing state into Prometheus gauges.

    Reads per-agent per-provider performance data from the router and updates:
      - ADAPTIVE_EMA_LATENCY
      - ADAPTIVE_P50_LATENCY
      - ADAPTIVE_SUCCESS_RATE
      - ADAPTIVE_SAMPLES_TOTAL
      - ADAPTIVE_CHAIN_ADAPTED

    Call this periodically (e.g. every 5s from the metrics loop) to keep
    Prometheus metrics up to date with adaptive routing state.

    Args:
        agent_router: An AgentModelRouter instance with performance data
    """
    summary = agent_router.get_adaptive_routing_summary()

    for agent_id, providers in summary.items():
        for provider, perf in providers.items():
            if provider.startswith("_"):
                # Skip internal keys like _adaptive_ready, _adapted_chain, _chain_adapted
                continue

            ADAPTIVE_EMA_LATENCY.labels(
                agent=agent_id, provider=provider
            ).set(perf.get("ema_latency_ms", 0.0))

            ADAPTIVE_P50_LATENCY.labels(
                agent=agent_id, provider=provider
            ).set(perf.get("p50_latency_ms", 0.0))

            ADAPTIVE_SUCCESS_RATE.labels(
                agent=agent_id, provider=provider
            ).set(perf.get("success_rate", 0.0))

            ADAPTIVE_SAMPLES_TOTAL.labels(
                agent=agent_id, provider=provider
            ).set(perf.get("samples", 0))

        # Set chain adaptation status
        adapted = providers.get("_chain_adapted", False)
        ADAPTIVE_CHAIN_ADAPTED.labels(agent=agent_id).set(1 if adapted else 0)


def record_chain_adaptation(agent_id: str):
    """
    Increment the chain adaptation counter for an agent.

    Call this when `get_optimal_provider_chain()` returns a different
    ordering than the static base chain for the first time (or whenever
    the ordering changes).

    Args:
        agent_id: The agent whose chain was reordered
    """
    ADAPTIVE_CHAIN_ADAPTATION_EVENTS.labels(agent=agent_id).inc()
