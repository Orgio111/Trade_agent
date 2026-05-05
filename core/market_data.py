"""WebSocket market data feed with auto-reconnect and uvloop integration."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime

import ccxt.pro as ccxtpro
import numpy as np

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import TickData

logger = logging.getLogger(__name__)


class MarketDataFeed:
    """Real-time OHLCV + orderbook feed via ccxt.pro WebSockets."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.symbols = cfg.symbols
        self.exchange_id = cfg.exchange
        self._exchange: ccxtpro.Exchange | None = None
        self._running = False
        # Ring buffer: last 500 ticks per symbol for indicator warmup
        self._tick_buffer: dict[str, deque[TickData]] = {
            s: deque(maxlen=500) for s in self.symbols
        }

    async def start(self) -> None:
        cfg = get_settings()
        ExchangeClass = getattr(ccxtpro, self.exchange_id)
        self._exchange = ExchangeClass({"enableRateLimit": True})
        self._running = True
        tasks = [self._stream_symbol(s) for s in self.symbols]
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False
        if self._exchange:
            await self._exchange.close()

    async def _stream_symbol(self, symbol: str) -> None:
        cfg = get_settings()
        bus = await get_bus()
        reconnect_delay = cfg.ws_reconnect_delay
        while self._running:
            try:
                async with asyncio.timeout(10):
                    ohlcv = await self._exchange.watch_ohlcv(symbol, "1m")
                if ohlcv:
                    bar = ohlcv[-1]
                    tick = TickData(
                        symbol=symbol,
                        timestamp=datetime.utcfromtimestamp(bar[0] / 1000),
                        open=bar[1],
                        high=bar[2],
                        low=bar[3],
                        close=bar[4],
                        volume=bar[5],
                    )
                    self._tick_buffer[symbol].append(tick)
                    await bus.publish(
                        cfg.stream_market_data,
                        MsgType.TICK,
                        tick.model_dump(mode="json"),
                    )
                    reconnect_delay = cfg.ws_reconnect_delay  # reset on success
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                logger.warning(
                    "WS error for %s: %s — reconnecting in %.1fs",
                    symbol,
                    exc,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)

    def get_closes(self, symbol: str) -> np.ndarray:
        return np.array([t.close for t in self._tick_buffer[symbol]])

    def get_highs(self, symbol: str) -> np.ndarray:
        return np.array([t.high for t in self._tick_buffer[symbol]])

    def get_lows(self, symbol: str) -> np.ndarray:
        return np.array([t.low for t in self._tick_buffer[symbol]])

    def get_volumes(self, symbol: str) -> np.ndarray:
        return np.array([t.volume for t in self._tick_buffer[symbol]])

    def latest_tick(self, symbol: str) -> TickData | None:
        buf = self._tick_buffer[symbol]
        return buf[-1] if buf else None
