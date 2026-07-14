"""Golden contract tests for Binance and Bybit candle normalization."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from packages.domain import SourceMode
from workers.market_data.adapters import (
    AdapterPayloadError,
    OpenCandleIgnored,
    normalize_binance_kline,
    normalize_bybit_kline,
)


RECEIVED = datetime(2026, 7, 15, 2, 1, 0, 100000, tzinfo=UTC)


def _binance(closed: bool = True) -> dict:
    return {
        "stream": "btcusdt@kline_1m",
        "data": {
            "e": "kline",
            "E": 1784080860050,
            "s": "BTCUSDT",
            "k": {
                "t": 1784080800000,
                "T": 1784080859999,
                "s": "BTCUSDT",
                "i": "1m",
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


def _bybit(confirmed: bool = True) -> dict:
    return {
        "topic": "kline.1.BTCUSDT",
        "type": "snapshot",
        "ts": 1784080860050,
        "data": [
            {
                "start": 1784080800000,
                "end": 1784080859999,
                "interval": "1",
                "open": "100.00",
                "high": "102.00",
                "low": "99.00",
                "close": "101.00",
                "volume": "12.5",
                "turnover": "1262.5",
                "confirm": confirmed,
                "timestamp": 1784080860050,
            }
        ],
    }


@pytest.mark.parametrize(
    ("factory", "payload", "venue"),
    (
        (normalize_binance_kline, _binance(), "binance"),
        (normalize_bybit_kline, _bybit(), "bybit"),
    ),
)
def test_closed_venue_kline_normalizes_to_same_contract(
    factory, payload, venue
) -> None:
    event = factory(
        payload,
        received_ts=RECEIVED,
        ingest_run_id="adapter-contract-001",
        source_mode=SourceMode.REPLAY,
    )

    assert event.venue == venue
    assert event.instrument_id == "BTCUSDT"
    assert event.event_type == "market.candle"
    assert event.payload.interval == "1m"
    assert event.payload.close == Decimal("101")
    assert event.payload.volume == Decimal("12.5")
    assert event.verify_checksum()


@pytest.mark.parametrize(
    ("factory", "payload"),
    (
        (normalize_binance_kline, _binance(False)),
        (normalize_bybit_kline, _bybit(False)),
    ),
)
def test_open_candles_are_not_forwarded(factory, payload) -> None:
    with pytest.raises(OpenCandleIgnored):
        factory(
            payload,
            received_ts=RECEIVED,
            ingest_run_id="adapter-contract-001",
        )


def test_wrong_binance_event_type_fails_without_guessing() -> None:
    payload = _binance()
    payload["data"]["e"] = "trade"
    with pytest.raises(AdapterPayloadError):
        normalize_binance_kline(
            payload,
            received_ts=RECEIVED,
            ingest_run_id="adapter-contract-001",
        )


def test_bybit_batch_is_rejected_until_batch_semantics_are_explicit() -> None:
    payload = _bybit()
    payload["data"].append(dict(payload["data"][0]))
    with pytest.raises(AdapterPayloadError):
        normalize_bybit_kline(
            payload,
            received_ts=RECEIVED,
            ingest_run_id="adapter-contract-001",
        )
