"""Normalize a confirmed Bybit V5 public kline message."""

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


def normalize_bybit_kline(
    raw: Mapping[str, Any],
    *,
    received_ts: datetime,
    ingest_run_id: str,
    source_mode: SourceMode = SourceMode.PAPER_LIVE,
    trace_id: str | None = None,
) -> MarketEvent:
    """Convert one confirmed Bybit V5 kline update to MarketEvent v1."""

    root = require_mapping(raw, "payload")
    topic = str(require_field(root, "topic"))
    if not topic.startswith("kline."):
        raise AdapterPayloadError("payload is not a Bybit kline topic")
    topic_parts = topic.split(".")
    if len(topic_parts) < 3:
        raise AdapterPayloadError("Bybit kline topic is malformed")
    symbol = topic_parts[-1].upper()
    items = require_field(root, "data")
    if not isinstance(items, list) or len(items) != 1:
        raise AdapterPayloadError("Bybit adapter requires exactly one kline item")
    item = require_mapping(items[0], "data[0]")
    if require_field(item, "confirm") is not True:
        raise OpenCandleIgnored("Bybit kline is not confirmed")

    open_time = milliseconds_to_utc(require_field(item, "start"), "data[0].start")
    close_time = milliseconds_to_utc(require_field(item, "end"), "data[0].end")
    exchange_ts_value = (
        item["timestamp"] if item.get("timestamp") is not None else require_field(root, "ts")
    )
    exchange_ts = milliseconds_to_utc(exchange_ts_value, "data[0].timestamp")
    received = require_utc(received_ts, "received_ts")
    interval = str(item.get("interval", topic_parts[1]))
    candle = CandlePayload(
        interval=f"{interval}m" if interval.isdigit() else interval,
        open_time=open_time,
        close_time=close_time,
        open=require_field(item, "open"),
        high=require_field(item, "high"),
        low=require_field(item, "low"),
        close=require_field(item, "close"),
        volume=require_field(item, "volume"),
        quote_volume=item.get("turnover"),
    )
    resolved_trace = trace_id or f"{ingest_run_id}:bybit:{symbol}:{item['start']}"
    return MarketEvent.create(
        trace_id=resolved_trace,
        event_type="market.candle",
        venue="bybit",
        market_type="spot",
        instrument_id=symbol,
        exchange_ts=exchange_ts,
        received_ts=received,
        source_mode=source_mode,
        ingest_run_id=ingest_run_id,
        payload=candle,
    )
