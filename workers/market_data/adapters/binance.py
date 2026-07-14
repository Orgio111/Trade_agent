"""Normalize a closed Binance Spot kline WebSocket event."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from packages.domain import CandlePayload, MarketEvent, SourceMode

from .common import (
    AdapterPayloadError,
    OpenCandleIgnored,
    milliseconds_to_utc,
    require_field,
    require_mapping,
    require_utc,
)


def normalize_binance_kline(
    raw: Mapping[str, Any],
    *,
    received_ts: datetime,
    ingest_run_id: str,
    source_mode: SourceMode = SourceMode.PAPER_LIVE,
    trace_id: str | None = None,
) -> MarketEvent:
    """Convert one raw or combined-stream kline message to MarketEvent v1."""

    root = require_mapping(raw, "payload")
    data = require_mapping(root.get("data", root), "data")
    if require_field(data, "e") != "kline":
        raise AdapterPayloadError("payload is not a Binance kline event")
    kline = require_mapping(require_field(data, "k"), "k")
    if require_field(kline, "x") is not True:
        raise OpenCandleIgnored("Binance kline is not closed")

    symbol = str(require_field(kline, "s")).upper()
    if not symbol:
        raise AdapterPayloadError("Binance symbol is empty")
    open_time = milliseconds_to_utc(require_field(kline, "t"), "k.t")
    close_time = milliseconds_to_utc(require_field(kline, "T"), "k.T")
    exchange_ts = milliseconds_to_utc(require_field(data, "E"), "E")
    received = require_utc(received_ts, "received_ts")
    candle = CandlePayload(
        interval=str(require_field(kline, "i")),
        open_time=open_time,
        close_time=close_time,
        open=require_field(kline, "o"),
        high=require_field(kline, "h"),
        low=require_field(kline, "l"),
        close=require_field(kline, "c"),
        volume=require_field(kline, "v"),
        quote_volume=kline.get("q"),
        trade_count=kline.get("n"),
    )
    resolved_trace = trace_id or f"{ingest_run_id}:binance:{symbol}:{kline['t']}"
    return MarketEvent.create(
        trace_id=resolved_trace,
        event_type="market.candle",
        venue="binance",
        market_type="spot",
        instrument_id=symbol,
        exchange_ts=exchange_ts,
        received_ts=received,
        source_mode=source_mode,
        ingest_run_id=ingest_run_id,
        payload=candle,
    )

