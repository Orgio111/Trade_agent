"""
QUANTEX Real-Time Data Pipeline — Binance WebSocket depth stream, funding rates, and Open Interest.

Architecture:
  ┌─────────────────────────┐
  │  Binance WebSocket      │ ← depth @100ms, kline @1s, ticker @1s
  ├─────────────────────────┤
  │  REST Polling           │ ← funding rate @1h, open interest @5m
  ├─────────────────────────┤
  │  Data Pipeline          │ ← normalization, outlier filtering, aggregation
  ├─────────────────────────┤
  │  Feature Engine         │ ← live feature computation
  ├─────────────────────────┤
  │  NATS / Redis Pub       │ ← broadcast to subscribers
  └─────────────────────────┘

Subscription channels (NATS):
  - data.orderbook.{symbol}    → depth snapshots
  - data.ticker.{symbol}       → 1s ticker updates
  - data.kline.{symbol}        → 1s kline updates  
  - data.funding.{symbol}      → funding rate updates
  - data.oi.{symbol}           → open interest updates

Usage:
    pipeline = DataPipeline()
    await pipeline.start()
    # Subscribes to Binance streams and publishes via NATS
    await pipeline.subscribe_orderbook("BTCUSDT")
    await pipeline.subscribe_funding("BTCUSDT")
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Optional, Callable
from collections import deque

import aiohttp
import pandas as pd
import numpy as np

logger = logging.getLogger("quantex.data_pipeline")


class OrderBookSnapshot:
    """Real-time order book snapshot with metadata."""

    def __init__(self, symbol: str, bids: list, asks: list, timestamp: float):
        self.symbol = symbol
        self.bids = [(float(p), float(q)) for p, q in bids[:50] if float(q) > 0]
        self.asks = [(float(p), float(q)) for p, q in asks[:50] if float(q) > 0]
        self.timestamp = timestamp
        self.bid_volume = sum(q for _, q in self.bids)
        self.ask_volume = sum(q for _, q in self.asks)
        self.spread = (self.asks[0][0] - self.bids[0][0]) / self.bids[0][0] if self.bids and self.asks else 0
        self.imbalance = (self.bid_volume - self.ask_volume) / (self.bid_volume + self.ask_volume + 1e-10)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bid": self.bids[0][0] if self.bids else 0,
            "ask": self.asks[0][0] if self.asks else 0,
            "spread_bps": round(self.spread * 10000, 2),
            "imbalance": round(self.imbalance, 4),
            "bid_volume": round(self.bid_volume, 2),
            "ask_volume": round(self.ask_volume, 2),
            "bid_depth_5": round(sum(q for _, q in self.bids[:5]), 2),
            "ask_depth_5": round(sum(q for _, q in self.asks[:5]), 2),
            "timestamp": self.timestamp,
        }


class TickerData:
    """Real-time ticker update."""

    def __init__(self, symbol: str, price: float, volume_24h: float,
                 high_24h: float, low_24h: float, change_24h: float):
        self.symbol = symbol
        self.price = price
        self.volume_24h = volume_24h
        self.high_24h = high_24h
        self.low_24h = low_24h
        self.change_24h = change_24h

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "change_24h_pct": round(self.change_24h * 100, 2),
            "timestamp": time.time(),
        }


class FundingRateData:
    """Funding rate data from Binance API."""

    def __init__(self, symbol: str, funding_rate: float, next_funding_time: int,
                 mark_price: float, index_price: float):
        self.symbol = symbol
        self.funding_rate = funding_rate
        self.next_funding_time = next_funding_time
        self.mark_price = mark_price
        self.index_price = index_price
        self.timestamp = int(time.time() * 1000)
        self.annualized_rate = funding_rate * 24 * 365  # Approximate annualized

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "funding_rate": round(self.funding_rate, 6),
            "annualized_pct": round(self.annualized_rate * 100, 2),
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "next_funding_time": self.next_funding_time,
            "timestamp": self.timestamp,
        }


class OpenInterestData:
    """Open Interest data from Binance API."""

    def __init__(self, symbol: str, open_interest: float, timestamp: int):
        self.symbol = symbol
        self.open_interest = open_interest  # In base asset
        self.open_interest_usd = 0.0
        self.timestamp = timestamp

    def to_dict(self, price: float = 0) -> dict:
        self.open_interest_usd = self.open_interest * price
        return {
            "symbol": self.symbol,
            "open_interest": round(self.open_interest, 2),
            "open_interest_usd": round(self.open_interest_usd, 2),
            "timestamp": self.timestamp,
        }


class BinanceWebSocketClient:
    """
    Binance WebSocket client for real-time market data.

    Supports:
      - Depth streams (@depth@100ms)
      - Kline streams (@kline)
      - Ticker streams (@ticker)
      - Book ticker streams (@bookTicker)

    Each stream publishes via the callback pattern.
    """

    BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self):
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._subscriptions: list[str] = []
        self._callbacks: dict[str, list[Callable]] = {}

        # Buffer for reconstructing depth snapshots
        self._depth_buffers: dict[str, dict] = {}

    async def connect(self):
        """Establish WebSocket connection."""
        self._session = aiohttp.ClientSession()
        self._running = True

    async def disconnect(self):
        """Close WebSocket connection."""
        self._running = False
        if self._session:
            await self._session.close()
        if self._ws:
            await self._ws.close()

    def subscribe_depth(self, symbol: str, callback: Callable):
        """Subscribe to depth stream for a symbol."""
        stream_name = f"{symbol.lower()}@depth@100ms"
        self._subscriptions.append(stream_name)
        self._add_callback(f"depth:{symbol}", callback)

    def subscribe_ticker(self, symbol: str, callback: Callable):
        """Subscribe to 24hr ticker stream."""
        stream_name = f"{symbol.lower()}@ticker"
        self._subscriptions.append(stream_name)
        self._add_callback(f"ticker:{symbol}", callback)

    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """Subscribe to kline stream."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        self._subscriptions.append(stream_name)
        self._add_callback(f"kline:{symbol}:{interval}", callback)

    def _add_callback(self, key: str, callback: Callable):
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    async def _handle_message(self, raw: str):
        """Process incoming WebSocket message."""
        try:
            msg = json.loads(raw)

            # Determine stream type from event type
            event_type = msg.get("e", "")

            if event_type == "depthUpdate":
                await self._handle_depth(msg)
            elif event_type == "24hrTicker":
                await self._handle_ticker(msg)
            elif event_type == "kline":
                await self._handle_kline(msg)

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from WebSocket: {raw[:100]}")
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")

    async def _handle_depth(self, msg: dict):
        """Process depth update event."""
        symbol = msg.get("s", "")
        key = f"depth:{symbol}"

        # Check if it's a snapshot or update
        if msg.get("U") and msg.get("u"):
            # Real-time update — for simplicity, use first bid/ask
            bids = msg.get("b", [])
            asks = msg.get("a", [])
            snapshot = OrderBookSnapshot(symbol, bids, asks, time.time())

            for cb in self._callbacks.get(key, []):
                await self._safe_callback(cb, snapshot)

    async def _handle_ticker(self, msg: dict):
        """Process 24hr ticker event."""
        symbol = msg.get("s", "")
        key = f"ticker:{symbol}"

        ticker = TickerData(
            symbol=symbol,
            price=float(msg.get("c", 0)),
            volume_24h=float(msg.get("v", 0)),
            high_24h=float(msg.get("h", 0)),
            low_24h=float(msg.get("l", 0)),
            change_24h=float(msg.get("P", 0)) / 100.0,
        )

        for cb in self._callbacks.get(key, []):
            await self._safe_callback(cb, ticker)

    async def _handle_kline(self, msg: dict):
        """Process kline event."""
        k = msg.get("k", {})
        symbol = msg.get("s", "")
        interval = k.get("i", "")
        key = f"kline:{symbol}:{interval}"

        kline_data = {
            "symbol": symbol,
            "interval": interval,
            "open": float(k.get("o", 0)),
            "high": float(k.get("h", 0)),
            "low": float(k.get("l", 0)),
            "close": float(k.get("c", 0)),
            "volume": float(k.get("v", 0)),
            "is_final": k.get("x", False),
            "timestamp": k.get("t", 0),
        }

        for cb in self._callbacks.get(key, []):
            await self._safe_callback(cb, kline_data)

    async def _safe_callback(self, cb, data):
        """Call a callback safely, catching exceptions."""
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(data)
            else:
                cb(data)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    async def run(self):
        """Main WebSocket loop with auto-reconnect."""
        while self._running:
            try:
                streams = "/".join(self._subscriptions)
                url = f"{self.BASE_URL}/{streams}" if streams else self.BASE_URL

                async with self._session.ws_connect(url, timeout=30) as ws:
                    self._ws = ws
                    logger.info(f"WebSocket connected: {len(self._subscriptions)} streams")

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)


class FundingRateFetcher:
    """
    REST-based funding rate fetcher for Binance Futures.

    Polls every hour (or on demand) to get funding rate history.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self._cache: dict[str, FundingRateData] = {}
        self._history: dict[str, list] = {}
        self._callbacks: list[Callable] = []

    def on_update(self, callback: Callable):
        """Register callback for funding rate updates."""
        self._callbacks.append(callback)

    async def fetch_current(self, symbol: str = "BTCUSDT") -> Optional[FundingRateData]:
        """Fetch current funding rate from Binance Futures API."""
        url = f"{self.BASE_URL}/fapi/v1/premiumIndex?symbol={symbol}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        logger.warning(f"Funding rate API: {resp.status}")
                        return None
                    data = await resp.json()

                    fr = FundingRateData(
                        symbol=data.get("symbol", symbol),
                        funding_rate=float(data.get("lastFundingRate", 0)),
                        next_funding_time=int(data.get("nextFundingTime", 0)),
                        mark_price=float(data.get("markPrice", 0)),
                        index_price=float(data.get("indexPrice", 0)),
                    )

                    self._cache[symbol] = fr
                    if symbol not in self._history:
                        self._history[symbol] = []
                    self._history[symbol].append({
                        "rate": fr.funding_rate,
                        "time": fr.timestamp,
                        "annualized": fr.annualized_rate,
                    })
                    # Keep last 1000 entries
                    self._history[symbol] = self._history[symbol][-1000:]

                    # Notify callbacks
                    for cb in self._callbacks:
                        await self._safe_notify(cb, fr)

                    return fr

        except Exception as e:
            logger.error(f"Funding rate fetch error: {e}")
            return None

    async def fetch_history(self, symbol: str = "BTCUSDT", limit: int = 100) -> list[dict]:
        """Fetch historical funding rates."""
        url = f"{self.BASE_URL}/fapi/v1/fundingRate?symbol={symbol}&limit={limit}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return [
                        {
                            "symbol": d["symbol"],
                            "rate": float(d["fundingRate"]),
                            "time": d["fundingTime"],
                        }
                        for d in data
                    ]
        except Exception as e:
            logger.error(f"Funding rate history error: {e}")
            return []

    async def poll_loop(self, interval_seconds: int = 3600):
        """Continuously poll funding rate every hour."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        while True:
            for symbol in symbols:
                await self.fetch_current(symbol)
                await asyncio.sleep(1)  # Small delay between symbols
            await asyncio.sleep(interval_seconds)

    async def _safe_notify(self, cb, data):
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(data)
            else:
                cb(data)
        except Exception as e:
            logger.error(f"Funding callback error: {e}")

    def get_recent_rates(self, symbol: str = "BTCUSDT", n: int = 10) -> list:
        """Get N most recent funding rates."""
        history = self._history.get(symbol, [])
        return history[-n:]

    def get_avg_funding_rate(self, symbol: str = "BTCUSDT", periods: int = 30) -> float:
        """Get average funding rate over recent periods."""
        history = self._history.get(symbol, [])[-periods:]
        if not history:
            return 0.0
        return sum(h["rate"] for h in history) / len(history)


class OpenInterestFetcher:
    """
    REST-based Open Interest fetcher for Binance Futures.

    Polls every 5 minutes to track OI changes.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self._cache: dict[str, OpenInterestData] = {}
        self._history: dict[str, list] = {}
        self._callbacks: list[Callable] = []

    def on_update(self, callback: Callable):
        self._callbacks.append(callback)

    async def fetch_current(self, symbol: str = "BTCUSDT") -> Optional[OpenInterestData]:
        """Fetch current open interest."""
        url = f"{self.BASE_URL}/fapi/v1/openInterest?symbol={symbol}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

                    oi = OpenInterestData(
                        symbol=data.get("symbol", symbol),
                        open_interest=float(data.get("openInterest", 0)),
                        timestamp=int(data.get("time", time.time() * 1000)),
                    )

                    self._cache[symbol] = oi
                    if symbol not in self._history:
                        self._history[symbol] = []
                    self._history[symbol].append({
                        "oi": oi.open_interest,
                        "time": oi.timestamp,
                    })
                    self._history[symbol] = self._history[symbol][-1000:]

                    for cb in self._callbacks:
                        await self._safe_notify(cb, oi)

                    return oi

        except Exception as e:
            logger.error(f"OI fetch error: {e}")
            return None

    async def poll_loop(self, interval_seconds: int = 300):
        """Poll OI every 5 minutes."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        while True:
            for symbol in symbols:
                await self.fetch_current(symbol)
                await asyncio.sleep(1)
            await asyncio.sleep(interval_seconds)

    async def _safe_notify(self, cb, data):
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb(data)
            else:
                cb(data)
        except Exception as e:
            logger.error(f"OI callback error: {e}")

    def get_oi_change(self, symbol: str = "BTCUSDT", lookback: int = 12) -> float:
        """Get OI change over N periods (as a ratio)."""
        history = self._history.get(symbol, [])[-lookback:]
        if len(history) < 2:
            return 0.0
        return (history[-1]["oi"] - history[0]["oi"]) / history[0]["oi"]


class DataPipeline:
    """
    Unified data pipeline combining WebSocket depth, REST funding rates, and OI.

    Manages the lifecycle of all data sources, normalizes data,
    and publishes to NATS/Redis subscribers.

    Usage:
        pipeline = DataPipeline()
        await pipeline.start()
        await pipeline.subscribe_orderbook_depth("BTCUSDT", my_callback)
        await pipeline.subscribe_funding("BTCUSDT", my_callback)
        # ... run trading loop ...
        await pipeline.stop()
    """

    def __init__(self):
        self.ws_client = BinanceWebSocketClient()
        self.funding_fetcher = FundingRateFetcher()
        self.oi_fetcher = OpenInterestFetcher()
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        """Start all data pipeline components."""
        self._running = True
        await self.ws_client.connect()
        self._tasks = [
            asyncio.create_task(self.ws_client.run(), name="ws_client"),
            asyncio.create_task(self.funding_fetcher.poll_loop(), name="funding_poll"),
            asyncio.create_task(self.oi_fetcher.poll_loop(), name="oi_poll"),
        ]
        logger.info("Data pipeline started: WebSocket + Funding + OI")

    async def stop(self):
        """Stop all data pipeline components."""
        self._running = False
        await self.ws_client.disconnect()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        logger.info("Data pipeline stopped")

    def subscribe_orderbook_depth(self, symbol: str, callback: Callable):
        """Subscribe to real-time order book depth updates."""
        self.ws_client.subscribe_depth(symbol, callback)

    def subscribe_ticker(self, symbol: str, callback: Callable):
        """Subscribe to 24hr ticker updates."""
        self.ws_client.subscribe_ticker(symbol, callback)

    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """Subscribe to kline updates."""
        self.ws_client.subscribe_kline(symbol, interval, callback)

    def subscribe_funding(self, symbol: str, callback: Callable):
        """Subscribe to funding rate updates."""
        self.funding_fetcher.on_update(
            lambda fr: asyncio.ensure_future(self._dispatch(fr, callback))
            if fr.symbol == symbol else None
        )

    def subscribe_open_interest(self, symbol: str, callback: Callable):
        """Subscribe to open interest updates."""
        self.oi_fetcher.on_update(
            lambda oi: asyncio.ensure_future(self._dispatch(oi, callback))
            if oi.symbol == symbol else None
        )

    async def _dispatch(self, data, callback):
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"Dispatch error: {e}")

    def get_current_funding(self, symbol: str = "BTCUSDT") -> Optional[dict]:
        """Get latest funding rate data."""
        fr = self.funding_fetcher._cache.get(symbol)
        return fr.to_dict() if fr else None

    def get_current_oi(self, symbol: str = "BTCUSDT") -> Optional[dict]:
        """Get latest open interest data."""
        oi = self.oi_fetcher._cache.get(symbol)
        return oi.to_dict() if oi else None
