"""Canonical Binance public market producer behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from packages.event_bus import CoreSubject
from workers.market_producer.runtime import BinanceMarketProducer


class FakeBus:
    def __init__(self) -> None:
        self.publications: list[tuple[CoreSubject, bytes, str]] = []

    async def publish(
        self,
        subject: CoreSubject,
        payload: bytes,
        *,
        message_id: str,
        stream: str | None = None,
    ) -> object:
        self.publications.append((subject, payload, message_id))
        return SimpleNamespace()


def kline(*, symbol: str = "BTCUSDT", interval: str = "1m", closed: bool = True) -> str:
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@kline_{interval}",
            "data": {
                "e": "kline",
                "E": 1784080860050,
                "s": symbol,
                "k": {
                    "t": 1784080800000,
                    "T": 1784080859999,
                    "s": symbol,
                    "i": interval,
                    "o": "100.00",
                    "h": "102.00",
                    "l": "99.00",
                    "c": "101.00",
                    "v": "12.5",
                    "q": "1262.5",
                    "n": 42,
                    "x": closed,
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_closed_allowlisted_candle_publishes_canonical_raw_event() -> None:
    bus = FakeBus()
    producer = BinanceMarketProducer(bus, clock=lambda: datetime(2026, 7, 15, tzinfo=UTC))

    event = await producer.handle_message(kline())

    assert event is not None
    assert event.instrument_id == "BTCUSDT"
    assert bus.publications == [
        (CoreSubject.MARKET_RAW, event.canonical_json().encode(), str(event.event_id))
    ]


@pytest.mark.asyncio
async def test_open_candle_is_ignored_without_publish() -> None:
    bus = FakeBus()
    producer = BinanceMarketProducer(bus)

    assert await producer.handle_message(kline(closed=False)) is None
    assert bus.publications == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [kline(symbol="BNBUSDT"), kline(interval="5m")],
)
async def test_noncanonical_symbol_or_interval_fails_closed(payload: str) -> None:
    bus = FakeBus()
    producer = BinanceMarketProducer(bus)

    with pytest.raises(ValueError):
        await producer.handle_message(payload)
    assert bus.publications == []


@pytest.mark.asyncio
async def test_reconnect_replay_uses_same_event_identity() -> None:
    first_bus, second_bus = FakeBus(), FakeBus()
    first = BinanceMarketProducer(first_bus)
    restarted = BinanceMarketProducer(second_bus)

    first_event = await first.handle_message(kline())
    second_event = await restarted.handle_message(kline())

    assert first_event is not None and second_event is not None
    assert first_event.event_id == second_event.event_id
