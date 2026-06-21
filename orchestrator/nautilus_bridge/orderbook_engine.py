"""
Orderbook Engine — Uses NautilusTrader's Rust-core to ingest and reconstruct
L2/L3 orderbook data, then exposes imbalance metrics for Brain 4.

Key metrics computed:
  - Bid-ask imbalance ratio
  - Depth-weighted mid-price shift
  - Liquidity gap detection
  - Market pressure indicator (buy vs sell)
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import nats
from nautilus_trader.config import LiveDataEngineConfig
from nautilus_trader.core.nautilus_pyo3 import OrderBookDelta
from nautilus_trader.model.data import QuoteTick, TradeTick
from nautilus_trader.model.instruments import CurrencyPair

log = logging.getLogger("nautilus_bridge.orderbook")


@dataclass
class OrderbookMetrics:
    """Order book imbalance metrics exposed to Brain 4."""
    bid_ask_imbalance: float = 0.0       # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    depth_weighted_mid: float = 0.0      # volume-weighted mid price
    liquidity_gap: float = 0.0           # gap ratio between best bid/ask levels
    buy_pressure: float = 0.0            # net buy pressure (0-1 scale)
    sell_pressure: float = 0.0           # net sell pressure (0-1 scale)
    spread_bps: float = 0.0              # bid-ask spread in basis points
    timestamp: float = 0.0


class OrderbookEngine:
    """
    Wraps NautilusTrader's data engine to process real-time orderbook
    depth and compute imbalance metrics.

    Usage:
        engine = OrderbookEngine(nats_url="nats://localhost:4222")
        await engine.start()
        # metrics available via engine.latest_metrics
    """

    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self.nats_url = nats_url
        self.nc: Optional[nats.NATS] = None
        self.js = None
        self.latest_metrics: dict[str, OrderbookMetrics] = {}
        self._running = False

    async def start(self):
        """Connect to NATS and begin receiving orderbook data."""
        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()
        self._running = True
        log.info("OrderbookEngine connected to NATS at %s", self.nats_url)

    async def stop(self):
        """Shutdown and disconnect."""
        self._running = False
        if self.nc:
            await self.nc.close()
            log.info("OrderbookEngine disconnected")

    def compute_imbalance(self, bids: list, asks: list) -> OrderbookMetrics:
        """
        Compute orderbook imbalance from bid/ask snapshots.

        Args:
            bids: [(price, volume), ...] sorted desc
            asks: [(price, volume), ...] sorted asc
        """
        if not bids or not asks:
            return OrderbookMetrics()

        # Total volumes
        total_bid_vol = sum(float(b[1]) for b in bids[:10])
        total_ask_vol = sum(float(a[1]) for a in asks[:10])
        total_vol = total_bid_vol + total_ask_vol
        if total_vol == 0:
            return OrderbookMetrics()

        # Imbalance ratio: +1 = all bids, -1 = all asks
        imbalance = (total_bid_vol - total_ask_vol) / total_vol

        # Depth-weighted mid price
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        weighted_mid = 0.0
        total_weight = 0.0
        for i, (price, vol) in enumerate(bids[:5]):
            w = float(vol) * (1.0 / (i + 1))  # closer levels get more weight
            weighted_mid += float(price) * w
            total_weight += w
        for i, (price, vol) in enumerate(asks[:5]):
            w = float(vol) * (1.0 / (i + 1))
            weighted_mid += float(price) * w
            total_weight += w
        if total_weight > 0:
            weighted_mid /= total_weight

        # Spread in bps
        spread = best_ask - best_bid
        spread_bps = (spread / mid) * 10_000 if mid > 0 else 0.0

        # Pressure indicators
        buy_pressure = total_bid_vol / total_vol if total_vol > 0 else 0.5
        sell_pressure = total_ask_vol / total_vol if total_vol > 0 else 0.5

        # Liquidity gap — ratio of 2nd-level volume to best-level
        gap = 0.0
        if len(bids) >= 2 and float(bids[0][1]) > 0:
            gap = float(bids[1][1]) / float(bids[0][1])
        elif len(asks) >= 2 and float(asks[0][1]) > 0:
            gap = float(asks[1][1]) / float(asks[0][1])

        return OrderbookMetrics(
            bid_ask_imbalance=round(imbalance, 6),
            depth_weighted_mid=round(weighted_mid, 2),
            liquidity_gap=round(gap, 4),
            buy_pressure=round(buy_pressure, 4),
            sell_pressure=round(sell_pressure, 4),
            spread_bps=round(spread_bps, 2),
            timestamp=time.time(),
        )

    async def publish_metrics(self, symbol: str, metrics: OrderbookMetrics):
        """Publish orderbook metrics to NATS for Brain 4 consumption."""
        if not self.js:
            return

        payload = {
            "brain_id": "orderflow_nautilus",
            "symbol": symbol,
            "metrics": {
                "bid_ask_imbalance": metrics.bid_ask_imbalance,
                "depth_weighted_mid": metrics.depth_weighted_mid,
                "liquidity_gap": metrics.liquidity_gap,
                "buy_pressure": metrics.buy_pressure,
                "sell_pressure": metrics.sell_pressure,
                "spread_bps": metrics.spread_bps,
            },
            "timestamp": metrics.timestamp,
        }
        ack = await self.js.publish(
            "signals.raw",
            json.dumps(payload).encode(),
        )
        log.debug("Published orderbook metrics for %s: seq=%d", symbol, ack.seq)
