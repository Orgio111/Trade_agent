"""Clock, latency, and staleness gates for market events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from .verdicts import QualityCode, QualityReason, QualityVerdict


def _parse(value: Any, field: str, reasons: list[QualityReason]) -> datetime | None:
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
    if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
        reasons.append(QualityReason(QualityCode.INVALID_TIMESTAMP, f"{field} must be timezone-aware", field))
        return None
    if parsed.utcoffset() != timedelta(0):
        reasons.append(QualityReason(QualityCode.TIMESTAMP_NOT_UTC, f"{field} must be UTC", field))
        return None
    return parsed.astimezone(UTC)


def validate_timestamps(
    exchange_ts: datetime | str,
    received_ts: datetime | str,
    *,
    now: datetime | str | None = None,
    max_staleness: timedelta = timedelta(seconds=5),
    max_transport_lag: timedelta = timedelta(seconds=2),
    max_future_skew: timedelta = timedelta(milliseconds=500),
    allowed_receive_skew: timedelta = timedelta(milliseconds=100),
) -> QualityVerdict:
    """Reject non-UTC, stale, future, or implausibly delayed events."""

    if any(limit < timedelta(0) for limit in (max_staleness, max_transport_lag, max_future_skew, allowed_receive_skew)):
        raise ValueError("timestamp tolerances must be non-negative")
    reasons: list[QualityReason] = []
    exchange = _parse(exchange_ts, "exchange_ts", reasons)
    received = _parse(received_ts, "received_ts", reasons)
    current = _parse(now if now is not None else datetime.now(UTC), "now", reasons)
    if exchange is None or received is None or current is None:
        return QualityVerdict.from_reasons(reasons)

    if exchange > current + max_future_skew:
        reasons.append(QualityReason(QualityCode.EVENT_FROM_FUTURE, "exchange_ts is in the future", "exchange_ts"))
    if received > current + max_future_skew:
        reasons.append(QualityReason(QualityCode.EVENT_FROM_FUTURE, "received_ts is in the future", "received_ts"))
    if received + allowed_receive_skew < exchange:
        reasons.append(
            QualityReason(
                QualityCode.RECEIVED_BEFORE_EXCHANGE,
                "received_ts precedes exchange_ts beyond allowed skew",
                "received_ts",
            )
        )
    if received - exchange > max_transport_lag:
        reasons.append(
            QualityReason(
                QualityCode.EXCESSIVE_TRANSPORT_LAG,
                "exchange-to-receiver latency exceeds the configured limit",
                "received_ts",
            )
        )
    if current - exchange > max_staleness:
        reasons.append(QualityReason(QualityCode.STALE_EVENT, "market event is stale", "exchange_ts"))
    return QualityVerdict.from_reasons(reasons)

