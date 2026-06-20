"""Brain #7: On-Chain / Mempool / Whale Activity Monitor.

Tracks large transactions, mempool activity, and whale wallet movements.
Weight in Go orchestrator: 0.05.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class OnChainBrain(BaseBrain):
    """On-chain analytics brain.

    Monitors:
      - Whale transactions (large BTC/ETH transfers to/from exchanges)
      - Mempool congestion (fee pressure)
      - Exchange flow (net inflow/outflow)

    Falls back to neutral if no on-chain data source is configured.
    """

    @property
    def brain_id(self) -> str:
        return "onchain_whale"

    def __init__(self) -> None:
        self._whale_threshold = float(os.getenv("WHALE_THRESHOLD_BTC", "10.0"))
        self._exchange_flows: deque[dict] = deque(maxlen=200)
        self._whale_txns: deque[dict] = deque(maxlen=100)
        self._mempool_fees: deque[float] = deque(maxlen=50)

    async def warmup(self) -> None:
        logger.info(f"[onchain_whale] Ready (whale_threshold={self._whale_threshold} BTC)")

    async def compute_score(self, symbol: str) -> BrainSignal:
        # Sum up signals from on-chain metrics
        flow_score = self._exchange_flow_score()
        whale_score = self._whale_score()
        mempool_score = self._mempool_score()

        # Weighted combination: exchange flow is most actionable
        combined = 0.5 * flow_score + 0.35 * whale_score + 0.15 * mempool_score

        confidence = 0.3
        if len(self._exchange_flows) > 10:
            confidence += 0.2
        if len(self._whale_txns) > 5:
            confidence += 0.15
        if len(self._mempool_fees) > 10:
            confidence += 0.1
        confidence = min(0.85, confidence)

        has_data = bool(self._exchange_flows) or bool(self._whale_txns)

        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(combined, -1.0, 1.0)),
            confidence=confidence if has_data else 0.1,
            metadata={
                "exchange_flow": round(flow_score, 4),
                "whale_signal": round(whale_score, 4),
                "mempool": round(mempool_score, 4),
                "data_available": has_data,
            },
        )

    def push_exchange_flow(self, inflow: float, outflow: float) -> None:
        """Push net exchange flow data (in BTC/ETH)."""
        self._exchange_flows.append({
            "inflow": inflow,
            "outflow": outflow,
            "net": outflow - inflow,  # positive = net outflow (bullish)
        })

    def push_whale_txn(self, amount: float, direction: str, usd_value: float = 0) -> None:
        """Push a whale transaction."""
        self._whale_txns.append({
            "amount": amount,
            "direction": direction,  # "inflow" (to exchange) or "outflow" (from exchange)
            "usd_value": usd_value,
        })

    def push_mempool_fee(self, median_fee_sats: float) -> None:
        """Push mempool median fee rate (sat/vB)."""
        self._mempool_fees.append(median_fee_sats)

    def _exchange_flow_score(self) -> float:
        """Net outflow = bullish (coins leaving exchanges).
        Net inflow = bearish (coins entering exchanges for selling)."""
        if not self._exchange_flows:
            return 0.0
        recent = list(self._exchange_flows)[-24:]  # last 24 data points
        net_flow = sum(f["net"] for f in recent)
        # Normalize: 100 BTC net outflow = strong bullish signal
        return float(np.tanh(net_flow / 100.0))

    def _whale_score(self) -> float:
        """Large outflows from exchanges → bullish.
        Large inflows to exchanges → bearish."""
        if not self._whale_txns:
            return 0.0
        recent = list(self._whale_txns)[-10:]
        inflow_vol = sum(t["amount"] for t in recent if t["direction"] == "inflow")
        outflow_vol = sum(t["amount"] for t in recent if t["direction"] == "outflow")
        net = outflow_vol - inflow_vol
        return float(np.tanh(net / self._whale_threshold))

    def _mempool_score(self) -> float:
        """High mempool fees → high activity → may precede price moves.
        Mild positive correlation with volatility."""
        if not self._mempool_fees:
            return 0.0
        fees = list(self._mempool_fees)
        avg_fee = float(np.mean(fees[-10:]))
        # Higher fees suggest more on-chain activity (neutral-slightly-bullish bias)
        return float(np.tanh(avg_fee / 100.0)) * 0.3
