"""
NautilusTrader Bridge — Connects NautilusTrader (Rust-core + Python API)
to our Trinity Architecture via NATS JetStream.

Components:
  - OrderbookEngine:  L2/L3 depth imbalance metrics for Brain 9
  - BacktestNode:     NautilusTrader BacktestEngine wrapper
  - NATSBridge:       Routes aggregated signals → execution → signals.executed

Architecture:
  Layer A (Python)                    Layer B (Go)              NautilusTrader
  ┌─────────────────────┐      ┌──────────────┐      ┌──────────────────────┐
  │ 9 Brains (ML/TA)    │      │ Orchestrator │      │ Rust Core Engine     │
  │ + OrderFlowNautilus │─┐    │ Aggregation  │      │ ┌──────────────────┐ │
  │   (orderbook depth) │ │    │ 20s window   │      │ │ Orderbook L2/L3 │ │
  └─────────────────────┘ │    └──────┬───────┘      │ │ Backtest Sim     │ │
                          │           │              │ │ Live Execution   │ │
                          │    signals.raw           │ └──────────────────┘ │
                          ├──────────→signals.aggregated→NATSBridge         │
                          │           │              └──────────────────────┘
                          │           │                        │
                          │           │              signals.executed
"""

from .orderbook_engine import OrderbookEngine, OrderbookMetrics
from .backtest_node import BacktestNode
from .nats_bridge import NATSBridge, ExecutionResult

__all__ = [
    "OrderbookEngine",
    "OrderbookMetrics",
    "BacktestNode",
    "NATSBridge",
    "ExecutionResult",
]
