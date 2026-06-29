"""Binance Market Ingestion Engine — 24/7 perpetual OHLCV.

Architecture:
  - Hydrates candles from Binance REST API (ccxt throttled)
  - 1.2x safety buffer: fetches 1728 candles for 1440-min lookback
  - Persistent FIFO cache at ~/.cache/moss-trade-bot-factory/v1.0.28/data_cache/
  - Data freshness enforcement: REST-only fill-ins (no WebSocket)

Candle schema:
  {
    "open": float, "high": float, "low": float, "close": float, 
    "volume": float, "timestamp_ms": int, "symbol": str, "interval": str
  }

Security: API keys loaded exclusively from .env (ccxt dict).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import pandas as pd

from .schemas import MOSS_FACTORY_VERSION

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────
LOOKBACK_MINUTES: int = 1440  # 24h
SAFETY_BUFFER_X: float = 1.2
CANDLE_LOOKBACK: int = int(LOOKBACK_MINUTES * SAFETY_BUFFER_X)  # 1728 candles
DATA_CACHE_DIR: Path = (
    Path.home() / ".cache" / "moss-trade-bot-factory" / MOSS_FACTORY_VERSION / "data_cache"
)
TICK_TIMEOUT_SECS: float = 300  # REST retries every 5 minutes

# Ensure cache directory
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION C:  BinanceIngestionEngine
# ══════════════════════════════════════════════════════════════════════

class BinanceIngestionEngine:
    """Binance OHLCV ingestion with lookback and safety buffer.

    Hydrates candles from Binance REST and maintains an in-memory
    FIFO buffer with 1.2x lookback (1728 candles → 1440 candles safe).
    Uses filesystem cache at DATA_CACHE_DIR keyed by symbol+interval.
    """

    def __init__(self, exchange_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ccxt binance instance (or testnet).

        exchange_config: optional ccxt kwargs.
        """
        self.exchange = ccxt.binance({
            "apiKey": os.getenv("BINANCE_API_KEY", ""),
            "secret": os.getenv("BINANCE_SECRET", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
            **(exchange_config or {})
        })
        # In-memory buffer: deque(maxlen=SAFETY_BUFFER_X * LOOKBACK_MINUTES)
        self._buffer: Dict[str, Optional[deque]] = {}  # "symbol_interval" → deque[candle]
        self._last_fetch: Dict[str, float] = {}         # "symbol_interval" → timestamp
        self._version: str = MOSS_FACTORY_VERSION

    def _make_cache_key(self, symbol: str, interval: str) -> Path:
        """Symbol+Interval cache path at ~/.cache/moss-trade-bot-factory/v1.0.28/."""
        key = f"{symbol}_{interval}".replace("/", "_")
        return DATA_CACHE_DIR / f"{key}.cache"

    def _candle_to_dict(self, candle: Tuple, symbol: str, interval: str) -> Dict[str, Any]:
        """Convert ccxt candle array to normalized dict."""
        return {
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4]),
            "volume": float(candle[5]),
            "timestamp_ms": int(candle[0]),
            "symbol": symbol,
            "interval": interval,
        }

    def _load_from_cache(self, symbol: str, interval: str) -> List[Dict[str, Any]]:
        """Load cached candles from disk."""
        cache_path = self._make_cache_key(symbol, interval)
        if cache_path.exists():
            try:
                data = pd.read_parquet(cache_path)
                return data.to_dict("records")
            except Exception:
                return []
        return []

    def _save_to_cache(self, symbol: str, interval: str, candles: List[Dict[str, Any]]) -> None:
        """Save candles to cold cache."""
        df = pd.DataFrame(candles)
        df = df.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
        df.to_parquet(self._make_cache_key(symbol, interval))

    def _fetch_from_binance(
        self,
        symbol: str,
        interval: str,
        lookback_minutes: int = LOOKBACK_MINUTES,
        force_max: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch candles from Binance REST with 1.2x safety buffer.

        Returns list of candle dicts.  Throws ccxt errors.
        """
        limit = int(lookback_minutes * SAFETY_BUFFER_X) if not force_max else 1000
        ccxt_interval = self._to_ccxt_interval(interval)
        candles_raw = self.exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=ccxt_interval,
            limit=limit,
        )
        return [self._candle_to_dict(candle, symbol, interval) for candle in candles_raw]

    def _to_ccxt_interval(self, interval: str) -> str:
        """Map trade_agent intervals → ccxt."""
        map_ = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
        }
        return map_.get(interval, interval)

    def _ensure_buffer_exists(self, symbol: str, interval: str) -> deque:
        """Initialize in-memory buffer for a stream."""
        buf = self._buffer.get(f"{symbol}_{interval}")
        if buf is None:
            # Try cache warmup
            cached = self._load_from_cache(symbol, interval)
            buf = deque(cached, maxlen=CANDLE_LOOKBACK)
            self._buffer[f"{symbol}_{interval}"] = buf
        return buf

    def _ensure_freshness(
        self,
        symbol: str,
        interval: str,
        lookback_minutes: int = LOOKBACK_MINUTES,
        allow_partial: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ensure buffer has fresh + lookback candles.

        Returns fresh candles fetched (empty if no fetch occurred).
        """
        buf_key = f"{symbol}_{interval}"
        buf = self._ensure_buffer_exists(symbol, interval)
        lookback = lookback_minutes
        now = int(time.time() * 1000)

        # Find newest & oldest candle in buffer
        oldest_ts = buf[0]["timestamp_ms"] if buf else 0
        newest_ts = buf[-1]["timestamp_ms"] if buf else 0
        expected_newest = now - (now % (60 * 1000 * self._interval_to_minutes(interval)))
        
        if (newest_ts >= expected_newest) and ((len(buf) >= lookback) or (allow_partial and buf)):
            logger.debug("[%s:%s] Buffer fresh — no fetch", symbol, interval)
            return []

        logger.info("[%s:%s] Buffer stale — fetching %d candles", symbol, interval, CANDLE_LOOKBACK)
        try:
            fresh_candles = self._fetch_from_binance(symbol, interval)
            buf.extend(fresh_candles)
            self._last_fetch[buf_key] = time.monotonic()
            # Save to cold cache
            self._save_to_cache(symbol, interval, list(buf))
            return fresh_candles
        except ccxt.BaseError as exc:
            logger.error("[%s:%s] Fetch error: %s", symbol, interval, exc)
            return []

    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        map_ = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        return map_.get(interval, 1)

    # ── Public API ────────────────────────

    def get_candles(
        self,
        symbol: str,
        interval: str,
        lookback_minutes: Optional[int] = None,
        force_fetch: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get lookback minutes of OHLCV for symbol+interval.

        Enforces 1.2x safety buffer.  Returns list of candle dicts
        sorted ascending by timestamp.
        """
        lookback = lookback_minutes or LOOKBACK_MINUTES
        if force_fetch:
            self._ensure_freshness(symbol, interval, lookback=lookback, allow_partial=False)
        else:
            self._ensure_freshness(symbol, interval)

        buf = self._ensure_buffer_exists(symbol, interval)
        # Return at least lookback candles, youngest first
        end_idx = len(buf)
        start_idx = max(0, end_idx - lookback)
        return [buf[i] for i in range(start_idx, end_idx)]

    def bootstrap_stream(self, symbol: str, interval: str) -> List[Dict[str, Any]]:
        """Cold start of a stream — fetches full lookback."""
        return self.get_candles(symbol, interval, lookback_minutes=LOOKBACK_MINUTES, force_fetch=True)

    def tick_stream(self, symbol: str, interval: str) -> List[Dict[str, Any]]:
        """Attempt to fetch new candles, returning newest.

        Returns only new candles (empty if none).
        """
        buf_key = f"{symbol}_{interval}"
        threshold = self._last_fetch.get(buf_key, 0) + TICK_TIMEOUT_SECS

        if time.monotonic() > threshold:
            fresh = self._ensure_freshness(symbol, interval)
            if fresh:
                # Return only newly fetched candles
                return [c for c in reversed(fresh)][:len(fresh)]
        return []

    @property
    def last_fetch_timestamps(self) -> Dict[str, float]:
        """symbol_interval → last monotonic fetch timestamp."""
        return self._last_fetch.copy()

    def invalidate_cache(self, symbol: str, interval: str) -> None:
        """Erase disk and memory cache for a stream."""
        buf_key = f"{symbol}_{interval}"
        if buf_key in self._buffer:
            del self._buffer[buf_key]
        if buf_key in self._last_fetch:
            del self._last_fetch[buf_key]
        cache_path = self._make_cache_key(symbol, interval)
        if cache_path.exists():
            cache_path.unlink()

    @property
    def version(self) -> str:
        return self._version

    def inject_into_registry(self, registry: Any) -> None:
        """Compat shim."""
        pass


# ══════════════════════════════════════════════════════════════════════
# __main__ Simulation Runner
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    import json
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_bootstrap():
        """Demo bootstrap BTC/USDT 15m and print latest candle."""
        ingestion = BinanceIngestionEngine()
        candles = ingestion.bootstrap_stream("BTC/USDT", "15m")
        print(f"Bootstrapped {len(candles)} candles")
        latest = candles[-1]
        print(json.dumps(latest, indent=2))
        print(f"Latest OHLC: {latest['open']:.2f} {latest['high']:.2f} {latest['low']:.2f} {latest['close']:.2f}")
        return ingestion
    
    asyncio.run(demo_bootstrap())
