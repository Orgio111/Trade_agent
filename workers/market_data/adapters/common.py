"""Shared strict parsing helpers for venue adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping


class AdapterPayloadError(ValueError):
    """The venue payload cannot be normalized without guessing."""


class OpenCandleIgnored(AdapterPayloadError):
    """The venue says this candle is not final yet."""


def require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdapterPayloadError(f"{field} must be an object")
    return value


def require_field(payload: Mapping[str, Any], field: str) -> Any:
    if field not in payload or payload[field] is None:
        raise AdapterPayloadError(f"missing required field: {field}")
    return payload[field]


def milliseconds_to_utc(value: Any, field: str) -> datetime:
    if isinstance(value, bool):
        raise AdapterPayloadError(f"{field} must be epoch milliseconds")
    try:
        milliseconds = int(value)
    except (TypeError, ValueError) as exc:
        raise AdapterPayloadError(f"{field} must be epoch milliseconds") from exc
    if milliseconds < 0:
        raise AdapterPayloadError(f"{field} cannot be negative")
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    except (OSError, OverflowError, ValueError) as exc:
        raise AdapterPayloadError(f"{field} is outside supported timestamp range") from exc


def require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise AdapterPayloadError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)

