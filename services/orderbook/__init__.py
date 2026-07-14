"""
Order Book Service — Level 2 order book processing and analysis.

Provides:
- Real-time order book maintenance
- Order flow imbalance (OFI) computation
- Liquidity metrics
- Spoofing/manipulation detection
- CVD (Cumulative Volume Delta)
"""

from __future__ import annotations

import asyncio
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from services.market_data import OrderBookSnapshot, Trade, create_market_data_service


# ═════════════════════════════════════════════════════════════════════

@dataclass
class OrderBookState:
    """Maintained order book state."""
    symbol: str
    timestamp: datetime
    bids: dict[float, float]  # price -> size
    asks: dict[float, float]
    exchange: str

    def get_bid_price(self) -> float:
        return max(self.bids.keys()) if self.bids else 0.0

    def get_ask_price(self) -> float:
        return min(self.asks.keys()) if self.asks else 0.0

    def get_mid_price(self) -> float:
        bid = self.get_bid_price()
        ask = self.get_ask_price()
        return (bid + ask) / 2 if bid and ask else 0.0

    def get_spread(self) -> float:
        return self.get_ask_price() - self.get_bid_price()

    def get_spread_bps(self) -> float:
        mid = self.get_mid_price()
        return (self.get_spread() / mid * 10000) if mid > 0 else 0.0

    def get_depth(self, side: str, levels: int = 10) -> list[tuple[float, float]]:
        """Get top N levels of depth."""
        if side == "bid":
            sorted_prices = sorted(self.bids.keys(), reverse=True)
        else:
            sorted_prices = sorted(self.asks.keys())
        return [(p, self.bids.get(p, 0) if side == "bid" else self.asks.get(p, 0)) for p in sorted_prices[:levels]]


@dataclass
class OrderBookMetrics:
    """Computed order book metrics."""
    symbol: str
    timestamp: datetime

    # Basic
    bid_price: float
    ask_price: float
    mid_price: float
    spread: float
    spread_bps: float

    # Depth
    bid_depth_10: float
    ask_depth_10: float
    bid_depth_50: float
    ask_depth_50: float

    # Imbalance
    order_flow_imbalance: float  # OFI
    volume_imbalance: float

    # CVD
    cvd: float
    cvd_1m: float
    cvd_5m: float

    # Liquidity
    liquidity_score: float
    whale_bid_volume: float
    whale_ask_volume: float

    # Microstructure
    bid_ask_volume_ratio: float
    trade_flow_imbalance: float

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_depth_10": self.bid_depth_10,
            "ask_depth_10": self.ask_depth_10,
            "bid_depth_50": self.bid_depth_50,
            "ask_depth_50": self.ask_depth_50,
            "order_flow_imbalance": self.order_flow_imbalance,
            "volume_imbalance": self.volume_imbalance,
            "cvd": self.cvd,
            "cvd_1m": self.cvd_1m,
            "cvd_5m": self.cvd_5m,
            "liquidity_score": self.liquidity_score,
            "whale_bid_volume": self.whale_bid_volume,
            "whale_ask_volume": self.whale_ask_volume,
            "bid_ask_volume_ratio": self.bid_ask_volume_ratio,
            "trade_flow_imbalance": self.trade_flow_imbalance,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════
# ORDER BOOK MAINTAINER
# ═══════════════════════════════════════════════════════════════════

class OrderBookMaintainer:
    """
    Maintains L2 order book from snapshots and incremental updates.
    Computes real-time metrics.
    """

    def __init__(self, symbol: str, max_depth: int = 100):
        self.symbol = symbol
        self.max_depth = max_depth
        self.state: Optional[OrderBookState] = None
        self.prev_state: Optional[OrderBookState] = None

        # CVD tracking
        self.cvd = 0.0
        self.cvd_history: deque = deque(maxlen=300)  # 5 min at 1s
        self.cvd_1m_history: deque = deque(maxlen=60)
        self.cvd_5m_history: deque = deque(maxlen=300)

        # Trade flow
        self.trade_buy_volume: deque = deque(maxlen=60)
        self.trade_sell_volume: deque = deque(maxlen=60)

        # Metrics history for smoothing
        self.ofi_history: deque = deque(maxlen=60)
        self.volume_imbalance_history: deque = deque(maxlen=60)

    def update_snapshot(self, snapshot: OrderBookSnapshot) -> OrderBookMetrics:
        """Update from full snapshot."""
        self.prev_state = self.state
        self.state = OrderBookState(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            bids=dict(snapshot.bids),
            asks=dict(snapshot.asks),
            exchange=snapshot.exchange,
        )
        return self._compute_metrics()

    def update_incremental(self, side: str, price: float, size: float) -> OrderBookMetrics:
        """Update from incremental change (price level update)."""
        if self.state is None:
            return None

        self.prev_state = OrderBookState(
            symbol=self.state.symbol,
            timestamp=self.state.timestamp,
            bids=self.state.bids.copy(),
            asks=self.state.asks.copy(),
            exchange=self.state.exchange,
        )

        if side == "bid":
            if size == 0:
                self.state.bids.pop(price, None)
            else:
                self.state.bids[price] = size
        else:
            if size == 0:
                self.state.asks.pop(price, None)
            else:
                self.state.asks[price] = size

        self.state.timestamp = datetime.now()
        return self._compute_metrics()

    def update_trade(self, trade: Trade):
        """Update CVD from trade."""
        if trade.side == "buy":
            self.cvd += trade.size
            self.trade_buy_volume.append(trade.size)
            self.trade_sell_volume.append(0.0)
        else:
            self.cvd -= trade.size
            self.trade_buy_volume.append(0.0)
            self.trade_sell_volume.append(trade.size)

        self.cvd_history.append(self.cvd)
        self.cvd_1m_history.append(self.cvd)
        self.cvd_5m_history.append(self.cvd)

    def _compute_metrics(self) -> OrderBookMetrics:
        """Compute all order book metrics."""
        if not self.state:
            return None

        bid = self.state.get_bid_price()
        ask = self.state.get_ask_price()
        mid = self.state.get_mid_price()
        spread = self.state.get_spread()
        spread_bps = self.state.get_spread_bps()

        # Depth
        bid_depth_10 = sum(s for _, s in self.state.get_depth("bid", 10))
        ask_depth_10 = sum(s for _, s in self.state.get_depth("ask", 10))
        bid_depth_50 = sum(s for _, s in self.state.get_depth("bid", 50))
        ask_depth_50 = sum(s for _, s in self.state.get_depth("ask", 50))

        # Order Flow Imbalance (OFI)
        ofi = 0.0
        if self.prev_state:
            ofi = self._compute_ofi()
        self.ofi_history.append(ofi)

        # Volume imbalance
        vol_imb = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10 + 1e-10)
        self.volume_imbalance_history.append(vol_imb)

        # CVD
        cvd_1m = self.cvd_1m_history[-1] - self.cvd_1m_history[0] if len(self.cvd_1m_history) > 1 else 0
        cvd_5m = self.cvd_5m_history[-1] - self.cvd_5m_history[0] if len(self.cvd_5m_history) > 1 else 0

        # Trade flow imbalance
        tfi = 0.0
        if self.trade_buy_volume and self.trade_sell_volume:
            total_buy = sum(self.trade_buy_volume)
            total_sell = sum(self.trade_sell_volume)
            tfi = (total_buy - total_sell) / (total_buy + total_sell + 1e-10)

        # Whale detection (orders > 95th percentile)
        whale_bid = self._detect_whale("bid")
        whale_ask = self._detect_whale("ask")

        # Liquidity score
        liquidity = self._compute_liquidity(bid_depth_10, ask_depth_10, spread_bps)

        # Bid/ask volume ratio
        ba_ratio = bid_depth_10 / (ask_depth_10 + 1e-10)

        return OrderBookMetrics(
            symbol=self.symbol,
            timestamp=datetime.now(),
            bid_price=bid,
            ask_price=ask,
            mid_price=mid,
            spread=spread,
            spread_bps=spread_bps,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            bid_depth_50=bid_depth_50,
            ask_depth_50=ask_depth_50,
            order_flow_imbalance=ofi,
            volume_imbalance=vol_imb,
            cvd=self.cvd,
            cvd_1m=cvd_1m,
            cvd_5m=cvd_5m,
            liquidity_score=liquidity,
            whale_bid_volume=whale_bid,
            whale_ask_volume=whale_ask,
            bid_ask_volume_ratio=ba_ratio,
            trade_flow_imbalance=tfi,
            metadata={
                "bid_levels": len(self.state.bids),
                "ask_levels": len(self.state.asks),
            },
        )

    def _compute_ofi(self) -> float:
        """Compute Order Flow Imbalance from book changes."""
        if not self.prev_state:
            return 0.0

        # Simplified OFI: (bid depth change - ask depth change) / (bid depth + ask depth)
        prev_bid_10 = sum(s for _, s in self.prev_state.get_depth("bid", 10))
        prev_ask_10 = sum(s for _, s in self.prev_state.get_depth("ask", 10))
        curr_bid_10 = sum(s for _, s in self.state.get_depth("bid", 10))
        curr_ask_10 = sum(s for _, s in self.state.get_depth("ask", 10))

        bid_change = curr_bid_10 - prev_bid_10
        ask_change = curr_ask_10 - prev_ask_10

        return (bid_change - ask_change) / (curr_bid_10 + curr_ask_10 + 1e-10)

    def _detect_whale(self, side: str) -> float:
        """Detect whale orders (top 5% by size)."""
        if side == "bid":
            sizes = list(self.state.bids.values())
        else:
            sizes = list(self.state.asks.values())

        if not sizes:
            return 0.0

        threshold = np.percentile(sizes, 95)
        return sum(s for s in sizes if s >= threshold)

    def _compute_liquidity(self, bid_depth: float, ask_depth: float, spread_bps: float) -> float:
        """Compute liquidity score (0-1)."""
        depth_score = min((bid_depth + ask_depth) / 1000, 1.0)  # Normalize
        spread_score = max(1.0 - spread_bps / 50, 0.0)  # 50 bps = 0 score
        return (depth_score + spread_score) / 2


# ═══════════════════════════════════════════════════════════════════
# SPOOFING / MANIPULATION DETECTOR
# ═══════════════════════════════════════════════════════════════════

class SpoofingDetector:
    """
    Detects potential spoofing/manipulation patterns:
    - Large orders placed and quickly cancelled
    - Layering (multiple orders at different prices)
    - Momentum ignition
    """

    def __init__(self, symbol: str, window_seconds: int = 60):
        self.symbol = symbol
        self.window_seconds = window_seconds

        # Track order lifecycle
        self.order_history: deque = deque(maxlen=10000)
        self.cancel_history: deque = deque(maxlen=10000)

        # Metrics
        self.spoof_score: float = 0.0
        self.layering_score: float = 0.0

    def record_order(self, side: str, price: float, size: float, order_id: str):
        """Record new order placement."""
        self.order_history.append({
            "timestamp": datetime.now(),
            "side": side,
            "price": price,
            "size": size,
            "order_id": order_id,
        })

    def record_cancel(self, side: str, price: float, size: float, order_id: str):
        """Record order cancellation."""
        self.cancel_history.append({
            "timestamp": datetime.now(),
            "side": side,
            "price": price,
            "size": size,
            "order_id": order_id,
        })

    def analyze(self) -> dict:
        """Analyze for spoofing patterns."""
        now = datetime.now()
        cutoff = now.timestamp() - self.window_seconds

        # Filter recent
        recent_orders = [o for o in self.order_history if o["timestamp"].timestamp() > cutoff]
        recent_cancels = [c for c in self.cancel_history if c["timestamp"].timestamp() > cutoff]

        if not recent_orders:
            return {"spoof_score": 0.0, "layering_score": 0.0, "flags": []}

        flags = []

        # Pattern 1: High cancellation rate
        cancel_rate = len(recent_cancels) / (len(recent_orders) + 1e-10)
        if cancel_rate > 0.8:
            flags.append(f"High cancel rate: {cancel_rate:.1%}")
            self.spoof_score = min(self.spoof_score + 0.1, 1.0)
        else:
            self.spoof_score = max(self.spoof_score - 0.02, 0.0)

        # Pattern 2: Layering (orders at multiple price levels, same side)
        by_side_price = {}
        for o in recent_orders:
            key = (o["side"], round(o["price"], 2))
            by_side_price[key] = by_side_price.get(key, 0) + 1

        for (side, price), count in by_side_price.items():
            if count > 5:  # Many orders at same price level
                flags.append(f"Layering detected: {side} @ {price} ({count} orders)")
                self.layering_score = min(self.layering_score + 0.1, 1.0)

        # Pattern 3: Large orders that don't execute
        if recent_orders:
            size_percentile = np.percentile([o["size"] for o in recent_orders], 90)
            large_cancels = [c for c in recent_cancels if c["size"] > size_percentile]
        else:
            large_cancels = []
        if len(large_cancels) > 3:
            flags.append(f"Large order cancellations: {len(large_cancels)}")
            self.spoof_score = min(self.spoof_score + 0.15, 1.0)

        return {
            "spoof_score": self.spoof_score,
            "layering_score": self.layering_score,
            "cancel_rate": cancel_rate,
            "recent_orders": len(recent_orders),
            "recent_cancels": len(recent_cancels),
            "flags": flags,
        }


# ═══════════════════════════════════════════════════════════════════
# ORDER BOOK SERVICE
# ═══════════════════════════════════════════════════════════════════

class OrderBookService:
    """
    Multi-symbol order book service.
    Maintains books, computes metrics, detects manipulation.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.market_service = create_market_data_service(self.config.get("market_data", {}))
        self.books: dict[str, OrderBookMaintainer] = {}
        self.detectors: dict[str, SpoofingDetector] = {}
        self._metrics_callbacks: list[Callable] = []
        self._alert_callbacks: list[Callable] = []

        # Wire up callbacks
        self.market_service.on_orderbook(self._on_orderbook)
        self.market_service.on_trade(self._on_trade)

    def on_metrics(self, callback: Callable[[OrderBookMetrics], None]):
        self._metrics_callbacks.append(callback)

    def on_alert(self, callback: Callable[[dict], None]):
        self._alert_callbacks.append(callback)

    def _get_or_create_book(self, symbol: str) -> OrderBookMaintainer:
        if symbol not in self.books:
            self.books[symbol] = OrderBookMaintainer(symbol)
            self.detectors[symbol] = SpoofingDetector(symbol)
        return self.books[symbol]

    async def _on_orderbook(self, snapshot: OrderBookSnapshot):
        book = self._get_or_create_book(snapshot.symbol)
        metrics = book.update_snapshot(snapshot)

        if metrics:
            for cb in self._metrics_callbacks:
                try:
                    await cb(metrics)
                except Exception as e:
                    print(f"[OrderBookService] Metrics callback error: {e}")

    async def _on_trade(self, trade: Trade):
        book = self._get_or_create_book(trade.symbol)
        book.update_trade(trade)

        # Re-emit metrics with updated CVD
        metrics = book._compute_metrics()
        if metrics:
            for cb in self._metrics_callbacks:
                try:
                    await cb(metrics)
                except Exception as e:
                    print(f"[OrderBookService] Trade metrics callback error: {e}")

    async def start(self, symbols: list[str]):
        """Start the service."""
        await self.market_service.start(symbols)

    async def stop(self):
        await self.market_service.stop()

    def get_metrics(self, symbol: str) -> OrderBookMetrics | None:
        """Get latest metrics for symbol."""
        if symbol in self.books:
            return self.books[symbol]._compute_metrics()
        return None

    def analyze_spoofing(self, symbol: str) -> dict:
        """Run spoofing analysis for symbol."""
        if symbol in self.detectors:
            return self.detectors[symbol].analyze()
        return {}


def create_order_book_service(config: dict | None = None) -> OrderBookService:
    """Factory for OrderBookService."""
    return OrderBookService(config)


if __name__ == "__main__":
    import asyncio

    async def test():
        config = {
            "market_data": {"exchanges": ["binance"], "binance": {"testnet": True}},
        }
        service = create_order_book_service(config)

        async def on_metrics(m: OrderBookMetrics):
            print(f"{m.symbol} OFI={m.order_flow_imbalance:.4f} CVD={m.cvd:.2f} Spread={m.spread_bps:.1f}bps")

        service.on_metrics(on_metrics)
        await service.start(["BTCUSDT", "ETHUSDT"])
        await asyncio.sleep(60)
        await service.stop()

    asyncio.run(test())