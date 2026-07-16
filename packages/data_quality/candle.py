"""Reason-coded OHLCV validation independent of transport parsing."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from pydantic import BaseModel

from .verdicts import QualityCode, QualityReason, QualityVerdict


def _mapping(value: BaseModel | Mapping[str, Any]) -> Mapping[str, Any]:
    return value.model_dump(mode="python") if isinstance(value, BaseModel) else value


def _number(data: Mapping[str, Any], field: str, reasons: list[QualityReason]) -> Decimal | None:
    if field not in data or data[field] is None:
        reasons.append(QualityReason(QualityCode.MISSING_FIELD, f"{field} is required", field))
        return None
    if isinstance(data[field], bool):
        reasons.append(QualityReason(QualityCode.INVALID_NUMBER, f"{field} is not numeric", field))
        return None
    try:
        value = data[field] if isinstance(data[field], Decimal) else Decimal(str(data[field]))
    except (InvalidOperation, TypeError, ValueError):
        reasons.append(QualityReason(QualityCode.INVALID_NUMBER, f"{field} is not numeric", field))
        return None
    if not value.is_finite():
        reasons.append(QualityReason(QualityCode.NON_FINITE_NUMBER, f"{field} must be finite", field))
        return None
    return value


def _datetime(data: Mapping[str, Any], field: str, reasons: list[QualityReason]) -> datetime | None:
    if field not in data or data[field] is None:
        reasons.append(QualityReason(QualityCode.MISSING_FIELD, f"{field} is required", field))
        return None
    value = data[field]
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    reasons.append(QualityReason(QualityCode.INVALID_TIMESTAMP, f"{field} is invalid", field))
    return None


def validate_candle(candle: BaseModel | Mapping[str, Any]) -> QualityVerdict:
    """Validate raw or parsed candle data without throwing on malformed input."""

    if not isinstance(candle, (BaseModel, Mapping)):
        return QualityVerdict.reject(
            QualityReason(QualityCode.MISSING_FIELD, "candle must be an object", "candle")
        )
    data = _mapping(candle)
    reasons: list[QualityReason] = []
    values = {name: _number(data, name, reasons) for name in ("open", "high", "low", "close", "volume")}
    quote_volume = None
    if data.get("quote_volume") is not None:
        quote_volume = _number(data, "quote_volume", reasons)

    for field in ("open", "high", "low", "close"):
        value = values[field]
        if value is not None and value <= 0:
            reasons.append(QualityReason(QualityCode.NON_POSITIVE_PRICE, f"{field} must be positive", field))
    if values["volume"] is not None and values["volume"] < 0:
        reasons.append(QualityReason(QualityCode.NEGATIVE_VOLUME, "volume cannot be negative", "volume"))
    if quote_volume is not None and quote_volume < 0:
        reasons.append(
            QualityReason(QualityCode.NEGATIVE_VOLUME, "quote_volume cannot be negative", "quote_volume")
        )

    high = values["high"]
    low = values["low"]
    comparable = [
        value
        for name in ("open", "close", "low")
        if (value := values[name]) is not None
    ]
    if high is not None and len(comparable) == 3 and high < max(comparable):
        reasons.append(
            QualityReason(QualityCode.HIGH_BELOW_PRICE, "high is below another OHLC value", "high")
        )
    comparable = [
        value
        for name in ("open", "close", "high")
        if (value := values[name]) is not None
    ]
    if low is not None and len(comparable) == 3 and low > min(comparable):
        reasons.append(QualityReason(QualityCode.LOW_ABOVE_PRICE, "low is above another OHLC value", "low"))

    open_time = _datetime(data, "open_time", reasons)
    close_time = _datetime(data, "close_time", reasons)
    if open_time is not None and close_time is not None and close_time <= open_time:
        reasons.append(
            QualityReason(
                QualityCode.INVALID_CANDLE_WINDOW,
                "close_time must be after open_time",
                "close_time",
            )
        )
    return QualityVerdict.from_reasons(reasons)
