"""
Thin Binance REST client focused on OHLCV fetch with retry/timeout.
Prefer existing `orchestrator.backtest.DataLoader.from_binance_api`
when historical data is required; this module is for runtime usage.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx

from .adapter import to_binance_symbol, to_binance_interval, normalize_symbols, normalize_timeframes

logger = logging.getLogger("quantex.binance.client")


@dataclass(frozen=True)
class BinanceMarket:
    symbol: str
    timeframe: str
    step_minutes: int


_BINANCE_STEP: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


class BinanceClient:
    def __init__(
        self,
        base_url: str = "https://api.binance.com/api/v3",
        timeout: float = 15.0,
        max_retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._markets = self._build_markets(["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    def _build_markets(self, symbols: Sequence[str]) -> dict[str, BinanceMarket]:
        markets: dict[str, BinanceMarket] = {}
        for sym in symbols:
            ccxt = sym if "/" in sym else f"{sym[:3]}/{sym[3:]}" if len(sym) == 8 and sym.endswith("USDT") else sym
            if ccxt in markets:
                continue
            bs = to_binance_symbol(ccxt)
            for tf, minutes in _BINANCE_STEP.items():
                markets[f"{bs}|{tf}"] = BinanceMarket(symbol=bs, timeframe=tf, step_minutes=minutes)
        return markets

    @property
    def available_markets(self) -> list[str]:
        return sorted({m.symbol for m in self._markets.values()})

    def market_for(self, symbol: str, timeframe: str) -> BinanceMarket:
        key = f"{to_binance_symbol(symbol)}|{to_binance_interval(timeframe)}"
        if key not in self._markets:
            raise KeyError(f"Unsupported market {symbol} {timeframe}")
        return self._markets[key]

    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[str]]:
        market = self.market_for(symbol, interval)
        url = f"{self.base_url}/klines"
        params: dict[str, object] = {
            "symbol": market.symbol,
            "interval": market.timeframe,
            "limit": min(limit, 1500),
        }
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.get(url, params=params)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as exc:
                last_exc = exc
                logger.warning("klines attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(self.backoff * (2 ** (attempt - 1)))
        raise RuntimeError(f"Binance klines failed after {self.max_retries} attempts") from last_exc

    async def fetch_24h_ticker(self, symbol: str) -> dict:
        bs = to_binance_symbol(symbol)
        url = f"{self.base_url}/ticker/24hr"
        params = {"symbol": bs}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    async def health(self) -> dict:
        url = f"{self.base_url}/ping"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return {"exchange": "binance_spot", "status": "ok"}
        except Exception as exc:
            return {"exchange": "binance_spot", "status": "error", "error": str(exc)}
