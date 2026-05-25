"""Web monitoring dashboard for the trading system.

Serves a real-time dashboard over HTTP with:
  - Portfolio equity / PnL / drawdown gauges
  - Agent status cards (technical, fundamental, sentiment, council, risk)
  - Live trade log
  - Risk metrics (VaR, Kelly, ATR stops)
  - Kill-switch indicator
  - REST API backed by the Prometheus metrics registry & in-memory state.
"""

from __future__ import annotations
