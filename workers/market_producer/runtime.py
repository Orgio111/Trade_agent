"""Binance public WebSocket producer for closed 1m BTC/ETH candles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
import json
import random
from typing import Protocol

import websockets

from packages.domain import MarketEvent, SourceMode
from packages.event_bus import CoreSubject
from workers.market_data.adapters import (
    AdapterPayloadError,
    OpenCandleIgnored,
    normalize_binance_kline,
)


_BINANCE_PUBLIC_STREAM = (
    "wss://stream.binance.com:9443/stream?streams="
    "btcusdt@kline_1m/ethusdt@kline_1m"
)
_ALLOWED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT"})
_INTERVAL = "1m"


class MarketPublisher(Protocol):
    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> object: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BinanceMarketProducer:
    """Normalize only authoritative closed public candles and publish raw v1."""

    def __init__(
        self,
        bus: MarketPublisher,
        *,
        stream_name: str = "QUANTEX_CORE",
        clock: Callable[[], datetime] = _utc_now,
        ingest_run_id: str | None = None,
    ) -> None:
        self._bus = bus
        self._stream_name = stream_name
        self._clock = clock
        self._ingest_run_id = ingest_run_id or "binance-public-live-v1"
        self.last_message_at: datetime | None = None
        self.reconnect_count = 0
        self.invalid_count = 0

    async def handle_message(self, raw: str | bytes) -> MarketEvent | None:
        received_at = self._clock()
        try:
            payload = json.loads(raw)
            event = normalize_binance_kline(
                payload,
                received_ts=received_at,
                ingest_run_id=self._ingest_run_id,
                source_mode=SourceMode.PAPER_LIVE,
            )
        except OpenCandleIgnored:
            self.last_message_at = received_at
            return None
        except (AdapterPayloadError, json.JSONDecodeError, TypeError) as exc:
            self.invalid_count += 1
            raise ValueError("invalid Binance public market message") from exc
        if event.instrument_id not in _ALLOWED_SYMBOLS:
            self.invalid_count += 1
            raise ValueError("Binance symbol is outside canonical allowlist")
        if event.payload.interval != _INTERVAL:
            self.invalid_count += 1
            raise ValueError("Binance interval is outside canonical allowlist")
        await self._bus.publish(
            CoreSubject.MARKET_RAW,
            event.canonical_json().encode("utf-8"),
            message_id=str(event.event_id),
            stream=self._stream_name,
        )
        self.last_message_at = received_at
        return event

    async def run_forever(self) -> None:
        """Reconnect forever with bounded exponential backoff and jitter."""

        attempt = 0
        while True:
            try:
                async with websockets.connect(
                    _BINANCE_PUBLIC_STREAM,
                    open_timeout=10,
                    ping_interval=20,
                    ping_timeout=10,
                    max_queue=64,
                    max_size=1_048_576,
                ) as socket:
                    attempt = 0
                    async for message in socket:
                        try:
                            await self.handle_message(message)
                        except ValueError:
                            continue
            except asyncio.CancelledError:
                raise
            except Exception:
                self.reconnect_count += 1
                attempt = min(attempt + 1, 8)
                delay = min(30.0, 0.5 * (2 ** (attempt - 1)))
                await asyncio.sleep(delay + random.uniform(0, delay * 0.2))


__all__ = ["BinanceMarketProducer"]
