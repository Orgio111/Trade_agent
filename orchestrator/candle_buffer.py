"""
QUANTEX Candle Buffer — Hot state for last N candles per symbol.

Provides O(1) access to recent candle history for:
  - Feature computation (rolling indicators need history)
  - VLM agent (chart rendering needs recent candles)
  - RAG agent (pattern matching needs context window)

Architecture:
  Redis (primary) → in-memory dict fallback (no Redis available)

Usage:
    buffer = CandleBuffer(max_candles=200)
    await buffer.push(symbol, candle)
    history = await buffer.get(symbol, limit=100)
    df = await buffer.to_dataframe(symbol, limit=100)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("quantex.candle_buffer")


class CandleBuffer:
    """
    Per-symbol circular buffer of recent candles.

    Each candle is a dict with keys:
        timestamp, open, high, low, close, volume, symbol, interval

    Falls back to in-memory deque if Redis is unavailable.
    """

    def __init__(
        self,
        max_candles: int = 200,
        redis_url: str | None = None,
    ):
        self.max_candles = max_candles
        self.redis_url = redis_url
        self._redis = None
        self._memory: dict[str, deque[dict]] = {}
        self._connected = False

    async def connect(self) -> bool:
        """Try to connect to Redis. Falls back to in-memory on failure."""
        if self._connected:
            return True

        if self.redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=3,
                )
                await self._redis.ping()
                self._connected = True
                logger.info(f"CandleBuffer connected to Redis ({self.redis_url})")
                return True
            except Exception as e:
                logger.warning(f"CandleBuffer Redis unavailable ({e}), using in-memory")
                self._redis = None

        # In-memory fallback
        self._connected = True
        logger.info("CandleBuffer using in-memory fallback")
        return True

    def _key(self, symbol: str) -> str:
        return f"candle_buffer:{symbol}"

    async def push(self, symbol: str, candle: dict) -> None:
        """
        Push a new candle into the buffer.

        If a candle with the same timestamp already exists, it is replaced.
        Only the last `max_candles` are retained.
        """
        if not self._connected:
            await self.connect()

        # Ensure timestamp is int (ms)
        if "timestamp" not in candle:
            candle["timestamp"] = int(time.time() * 1000)

        if self._redis is not None:
            try:
                key = self._key(symbol)
                # Use a list, trim to max_candles
                await self._redis.rpush(key, json.dumps(candle, default=str))
                await self._redis.ltrim(key, -self.max_candles, -1)
                return
            except Exception as e:
                logger.warning(f"Redis push failed ({e}), falling back to memory")

        # In-memory fallback
        if symbol not in self._memory:
            self._memory[symbol] = deque(maxlen=self.max_candles)

        buf = self._memory[symbol]
        # Replace if same timestamp
        if buf and buf[-1].get("timestamp") == candle.get("timestamp"):
            buf[-1] = candle
        else:
            buf.append(candle)

    async def get(self, symbol: str, limit: int = 100) -> list[dict]:
        """
        Get the last `limit` candles for a symbol.

        Returns list of candle dicts, oldest first.
        """
        if not self._connected:
            await self.connect()

        if self._redis is not None:
            try:
                key = self._key(symbol)
                raw = await self._redis.lrange(key, -limit, -1)
                return [json.loads(r) for r in raw]
            except Exception as e:
                logger.warning(f"Redis get failed ({e}), falling back to memory")

        # In-memory fallback
        buf = self._memory.get(symbol, deque())
        items = list(buf)
        return items[-limit:] if len(items) > limit else items

    async def to_dataframe(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get last N candles as a pandas DataFrame.

        Columns: timestamp, open, high, low, close, volume, symbol
        Index: DatetimeIndex from timestamp.
        """
        candles = await self.get(symbol, limit)
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # Ensure required columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0

        # Convert timestamp to datetime index
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("datetime")
            df = df.sort_index()

        return df

    async def count(self, symbol: str) -> int:
        """Get number of buffered candles for a symbol."""
        if not self._connected:
            await self.connect()

        if self._redis is not None:
            try:
                key = self._key(symbol)
                return await self._redis.llen(key)
            except Exception:
                pass

        return len(self._memory.get(symbol, deque()))

    async def clear(self, symbol: str | None = None) -> None:
        """Clear buffer for one or all symbols."""
        if not self._connected:
            await self.connect()

        if self._redis is not None:
            try:
                if symbol:
                    await self._redis.delete(self._key(symbol))
                else:
                    keys = await self._redis.keys("candle_buffer:*")
                    if keys:
                        await self._redis.delete(*keys)
            except Exception:
                pass

        if symbol:
            self._memory.pop(symbol, None)
        else:
            self._memory.clear()

    async def close(self) -> None:
        """Close Redis connection if open."""
        if self._redis is not None:
            try:
                await self._redis.close()
            except Exception:
                pass
        self._connected = False
