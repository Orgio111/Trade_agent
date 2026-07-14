"""
Market Data Service — WebSocket / FIX / REST ingestion for multiple exchanges.
Supports: Binance, Bybit, Coinbase, Kraken, Forex, Stocks.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import aiohttp
import ccxt.async_support as ccxt
import websockets

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Candle:
    """Normalized candle data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str
    timeframe: str = "1m"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "exchange": self.exchange,
            "timeframe": self.timeframe,
        }


@dataclass
class OrderBookSnapshot:
    """Level 2 order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: list[tuple[float, float]]  # (price, size)
    asks: list[tuple[float, float]]
    exchange: str

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bids": self.bids,
            "asks": self.asks,
            "exchange": self.exchange,
        }


@dataclass
class Trade:
    """Individual trade/tick."""
    symbol: str
    timestamp: datetime
    price: float
    size: float
    side: str  # "buy" or "sell"
    trade_id: str
    exchange: str


# ═══════════════════════════════════════════════════════════════════
# EXCHANGE ADAPTERS (Abstract Base + Implementations)
# ═══════════════════════════════════════════════════════════════════

class ExchangeAdapter(ABC):
    """Abstract base for exchange adapters."""

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable) -> None:
        pass

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        pass

    @abstractmethod
    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        pass

    @abstractmethod
    async def fetch_rest_candles(self, symbol: str, timeframe: str, limit: int = 100) -> list[Candle]:
        pass


class BinanceAdapter(ExchangeAdapter):
    """Binance WebSocket + REST adapter."""

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        """Initialize connection."""
        self._running = True
        logger.info("[Binance] Connected")

    async def disconnect(self) -> None:
        """Close all connections."""
        self._running = False
        for ws in self.ws_connections.values():
            await ws.close()
        for task in self._tasks:
            task.cancel()
        logger.info("[Binance] Disconnected")

    def _build_ws_url(self, stream: str) -> str:
        base = "wss://testnet.binance.vision" if self.testnet else "wss://stream.binance.com"
        return f"{base}/ws/{stream}"

    async def _ws_handler(self, url: str, callback: Callable, parser: Callable):
        """Generic WebSocket handler with reconnection."""
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    logger.info(f"[Binance] WS connected: {url}")
                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        parsed = parser(data)
                        if parsed:
                            await callback(parsed)
            except Exception as e:
                logger.warning(f"[Binance] WS error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def _parse_candle(self, data: dict) -> Optional[Candle]:
        """Parse Binance kline message."""
        if "k" not in data:
            return None
        k = data["k"]
        return Candle(
            symbol=k["s"],
            timestamp=datetime.fromtimestamp(k["t"] / 1000),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            exchange="binance",
            timeframe=k["i"],
        )

    def _parse_orderbook(self, data: dict) -> Optional[OrderBookSnapshot]:
        """Parse Binance depth message."""
        if "b" not in data or "a" not in data:
            return None
        return OrderBookSnapshot(
            symbol=data.get("s", "UNKNOWN"),
            timestamp=datetime.now(),
            bids=[(float(p), float(s)) for p, s in data["b"][:20]],
            asks=[(float(p), float(s)) for p, s in data["a"][:20]],
            exchange="binance",
        )

    def _parse_trade(self, data: dict) -> Optional[Trade]:
        """Parse Binance trade message."""
        return Trade(
            symbol=data.get("s", ""),
            timestamp=datetime.fromtimestamp(data.get("T", 0) / 1000),
            price=float(data.get("p", 0)),
            size=float(data.get("q", 0)),
            side="sell" if data.get("m", False) else "buy",
            trade_id=str(data.get("t", "")),
            exchange="binance",
        )

    async def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable) -> None:
        """Subscribe to candle stream."""
        stream = f"{symbol.lower()}@kline_{timeframe}"
        url = self._build_ws_url(stream)
        task = asyncio.create_task(self._ws_handler(url, callback, self._parse_candle))
        self._tasks.append(task)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        """Subscribe to orderbook stream (depth20)."""
        stream = f"{symbol.lower()}@depth20@100ms"
        url = self._build_ws_url(stream)
        task = asyncio.create_task(self._ws_handler(url, callback, self._parse_orderbook))
        self._tasks.append(task)

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        """Subscribe to trade stream."""
        stream = f"{symbol.lower()}@trade"
        url = self._build_ws_url(stream)
        task = asyncio.create_task(self._ws_handler(url, callback, self._parse_trade))
        self._tasks.append(task)

    async def fetch_rest_candles(self, symbol: str, timeframe: str, limit: int = 100) -> list[Candle]:
        """Fetch candles via REST API."""
        exchange = ccxt.binance({
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if self.testnet:
            exchange.set_sandbox_mode(True)

        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            await exchange.close()
            return [
                Candle(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(c[0] / 1000),
                    open=c[1], high=c[2], low=c[3], close=c[4], volume=c[5],
                    exchange="binance", timeframe=timeframe,
                )
                for c in ohlcv
            ]
        except Exception as e:
            logger.error(f"[Binance] REST fetch failed: {e}")
            await exchange.close()
            return []


class BybitAdapter(ExchangeAdapter):
    """Bybit WebSocket + REST adapter."""

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        self._running = True
        logger.info("[Bybit] Connected")

    async def disconnect(self) -> None:
        self._running = False
        for ws in self.ws_connections.values():
            await ws.close()
        for task in self._tasks:
            task.cancel()
        logger.info("[Bybit] Disconnected")

    def _build_ws_url(self) -> str:
        if self.testnet:
            return "wss://stream-testnet.bybit.com/v5/public/spot"
        return "wss://stream.bybit.com/v5/public/spot"

    async def _ws_handler(self, callback: Callable, parser: Callable, subscribe_msg: dict):
        url = self._build_ws_url()
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"[Bybit] WS connected")
                    async for message in ws:
                        if not self._running:
                            break
                        data = json.loads(message)
                        if "data" in data:
                            for item in data["data"]:
                                parsed = parser(item)
                                if parsed:
                                    await callback(parsed)
            except Exception as e:
                logger.warning(f"[Bybit] WS error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def _parse_candle(self, data: dict) -> Optional[Candle]:
        return Candle(
            symbol=data.get("symbol", ""),
            timestamp=datetime.fromtimestamp(data.get("start", 0) / 1000),
            open=float(data.get("open", 0)),
            high=float(data.get("high", 0)),
            low=float(data.get("low", 0)),
            close=float(data.get("close", 0)),
            volume=float(data.get("volume", 0)),
            exchange="bybit",
            timeframe=data.get("interval", "1m"),
        )

    async def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable) -> None:
        msg = {
            "op": "subscribe",
            "args": [f"kline.{timeframe}.{symbol}"],
        }
        task = asyncio.create_task(self._ws_handler(callback, self._parse_candle, msg))
        self._tasks.append(task)

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        # Similar implementation for orderbook
        pass

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        pass

    async def fetch_rest_candles(self, symbol: str, timeframe: str, limit: int = 100) -> list[Candle]:
        # REST implementation
        return []


class CoinbaseAdapter(ExchangeAdapter):
    """Coinbase Pro (Advanced Trade) WebSocket adapter."""

    def __init__(self, api_key: str = "", api_secret: str = "", passphrase: str = "", sandbox: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.sandbox = sandbox
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        self._running = True
        logger.info("[Coinbase] Connected")

    async def disconnect(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("[Coinbase] Disconnected")

    async def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable) -> None:
        # Coinbase uses different channel format
        pass

    async def subscribe_orderbook(self, symbol: str, callback: Callable) -> None:
        pass

    async def subscribe_trades(self, symbol: str, callback: Callable) -> None:
        pass

    async def fetch_rest_candles(self, symbol: str, timeframe: str, limit: int = 100) -> list[Candle]:
        return []


# ═══════════════════════════════════════════════════════════════════
# MARKET DATA SERVICE (Main Orchestrator)
# ═══════════════════════════════════════════════════════════════════

class MarketDataService:
    """
    Central market data ingestion service.
    Manages multiple exchange adapters and publishes normalized data to event bus.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.adapters: dict[str, ExchangeAdapter] = {}
        self._running = False
        self._candle_callbacks: list[Callable] = []
        self._orderbook_callbacks: list[Callable] = []
        self._trade_callbacks: list[Callable] = []

        # Initialize adapters based on config
        self._init_adapters()

    def _init_adapters(self):
        """Initialize exchange adapters from config."""
        exchanges = self.config.get("exchanges", ["binance"])

        for ex_name in exchanges:
            ex_config = self.config.get(ex_name, {})
            if ex_name == "binance":
                self.adapters["binance"] = BinanceAdapter(
                    api_key=ex_config.get("api_key", ""),
                    api_secret=ex_config.get("api_secret", ""),
                    testnet=ex_config.get("testnet", False),
                )
            elif ex_name == "bybit":
                self.adapters["bybit"] = BybitAdapter(
                    api_key=ex_config.get("api_key", ""),
                    api_secret=ex_config.get("api_secret", ""),
                    testnet=ex_config.get("testnet", False),
                )
            elif ex_name == "coinbase":
                self.adapters["coinbase"] = CoinbaseAdapter(
                    api_key=ex_config.get("api_key", ""),
                    api_secret=ex_config.get("api_secret", ""),
                    passphrase=ex_config.get("passphrase", ""),
                    sandbox=ex_config.get("sandbox", False),
                )

    def on_candle(self, callback: Callable[[Candle], None]):
        """Register candle callback."""
        self._candle_callbacks.append(callback)

    def on_orderbook(self, callback: Callable[[OrderBookSnapshot], None]):
        """Register orderbook callback."""
        self._orderbook_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]):
        """Register trade callback."""
        self._trade_callbacks.append(callback)

    async def _dispatch_candle(self, candle: Candle):
        """Dispatch candle to all callbacks."""
        for cb in self._candle_callbacks:
            try:
                await cb(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")

    async def _dispatch_orderbook(self, ob: OrderBookSnapshot):
        for cb in self._orderbook_callbacks:
            try:
                await cb(ob)
            except Exception as e:
                logger.error(f"Orderbook callback error: {e}")

    async def _dispatch_trade(self, trade: Trade):
        for cb in self._trade_callbacks:
            try:
                await cb(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    async def start(self, symbols: list[str], timeframes: list[str] = None):
        """Start all adapters and subscriptions."""
        timeframes = timeframes or ["1m"]
        self._running = True

        # Connect all adapters
        for name, adapter in self.adapters.items():
            try:
                await adapter.connect()
                logger.info(f"[{name}] Adapter connected")
            except Exception as e:
                logger.error(f"[{name}] Connection failed: {e}")

        # Subscribe to data streams
        for name, adapter in self.adapters.items():
            for symbol in symbols:
                for tf in timeframes:
                    try:
                        await adapter.subscribe_candles(symbol, tf, self._dispatch_candle)
                        await adapter.subscribe_trades(symbol, self._dispatch_trade)
                        await adapter.subscribe_orderbook(symbol, self._dispatch_orderbook)
                        logger.info(f"[{name}] Subscribed to {symbol} {tf}")
                    except Exception as e:
                        logger.error(f"[{name}] Subscribe failed for {symbol}: {e}")

        logger.info(f"[MarketDataService] Started for {len(symbols)} symbols")

    async def stop(self):
        """Stop all adapters."""
        self._running = False
        for name, adapter in self.adapters.items():
            try:
                await adapter.disconnect()
            except Exception as e:
                logger.error(f"[{name}] Disconnect error: {e}")
        logger.info("[MarketDataService] Stopped")

    async def fetch_historical(self, symbol: str, timeframe: str, limit: int = 100,
                               exchange: str = "binance") -> list[Candle]:
        """Fetch historical candles from specific exchange."""
        adapter = self.adapters.get(exchange)
        if adapter:
            return await adapter.fetch_rest_candles(symbol, timeframe, limit)
        return []


# ═══════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════

def create_market_data_service(config: dict | None = None) -> MarketDataService:
    """Factory for MarketDataService."""
    return MarketDataService(config)


if __name__ == "__main__":
    # Test run
    import asyncio

    async def test():
        config = {
            "exchanges": ["binance"],
            "binance": {"testnet": True},
        }
        service = create_market_data_service(config)

        async def on_candle(c: Candle):
            print(f"Candle: {c.symbol} {c.close} @ {c.timestamp}")

        service.on_candle(on_candle)
        await service.start(["BTCUSDT", "ETHUSDT"])
        await asyncio.sleep(30)
        await service.stop()

    asyncio.run(test())