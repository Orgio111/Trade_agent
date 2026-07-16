"""Versioned, deterministic market-event contracts.

The models in this module deliberately contain no broker, transport, or storage
logic.  They are the boundary between those systems.  Decimal values are
serialized as strings and timestamps are normalized to UTC so replay and live
ingestion produce byte-for-byte stable payload checksums.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class SourceMode(str, Enum):
    """Provenance mode for a market event."""

    HISTORICAL = "historical"
    REPLAY = "replay"
    PAPER_LIVE = "paper_live"
    LIVE = "live"


def _finite_decimal(value: Any) -> Decimal:
    """Coerce a JSON-compatible value to a finite Decimal."""

    if isinstance(value, bool):
        raise ValueError("boolean is not a decimal value")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("value must be a valid decimal") from exc
    if not decimal_value.is_finite():
        raise ValueError("decimal value must be finite")
    return decimal_value


def _decimal_text(value: Decimal) -> str:
    """Return one canonical non-exponent decimal representation."""

    if value.is_zero():
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python", exclude_none=False))
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical JSON does not support non-finite decimals")
        return _decimal_text(value)
    if isinstance(value, datetime):
        normalized = _utc_datetime(value)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a value using the stable representation used for checksums."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_payload_checksum(payload: Any) -> str:
    """Return the SHA-256 digest of a payload's canonical JSON."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _market_event_identity(
    *,
    event_type: str,
    venue: str,
    market_type: str,
    instrument_id: str,
    exchange_ts: datetime,
    sequence_start: int | None,
    sequence_end: int | None,
    source_mode: SourceMode,
    ingest_run_id: str,
    payload_checksum: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "event_type": event_type,
        "venue": venue,
        "market_type": market_type,
        "instrument_id": instrument_id,
        "exchange_ts": _utc_datetime(exchange_ts),
        "sequence_start": sequence_start,
        "sequence_end": sequence_end,
        "source_mode": source_mode,
        "ingest_run_id": ingest_run_id,
        "payload_checksum": payload_checksum,
    }


class CandlePayload(BaseModel):
    """A venue-independent OHLCV candle payload.

    Venue, market type, and instrument identity live on :class:`MarketEvent`
    and are intentionally not duplicated here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    interval: str = Field(min_length=1, max_length=32, pattern=r"^[0-9]+[smhdwM]$")
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None = None
    trade_count: int | None = Field(default=None, ge=0)

    @field_validator("open", "high", "low", "close", "volume", "quote_volume", mode="before")
    @classmethod
    def validate_decimals(cls, value: Any) -> Decimal | None:
        return None if value is None else _finite_decimal(value)

    @field_validator("open_time", "close_time")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _utc_datetime(value)

    @model_validator(mode="after")
    def validate_candle_invariants(self) -> CandlePayload:
        if self.close_time <= self.open_time:
            raise ValueError("close_time must be after open_time")
        if any(price <= 0 for price in (self.open, self.high, self.low, self.close)):
            raise ValueError("OHLC prices must be positive")
        if self.volume < 0 or (self.quote_volume is not None and self.quote_volume < 0):
            raise ValueError("volumes must be non-negative")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to open, low, and close")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to open, high, and close")
        return self

    @field_serializer("open", "high", "low", "close", "volume", "quote_volume", when_used="json")
    def serialize_decimal(self, value: Decimal | None) -> str | None:
        return None if value is None else _decimal_text(value)


class MarketEvent(BaseModel):
    """Canonical version 1.0 envelope for a candle market event."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    event_id: UUID
    trace_id: str = Field(min_length=1, max_length=128)
    event_type: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_.-]*$")
    venue: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_-]+$")
    market_type: str = Field(min_length=1, max_length=32, pattern=r"^[a-z0-9_-]+$")
    instrument_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9][A-Z0-9_:/.-]*$")
    exchange_ts: datetime
    received_ts: datetime
    sequence_start: int | None = Field(default=None, ge=0)
    sequence_end: int | None = Field(default=None, ge=0)
    source_mode: SourceMode
    ingest_run_id: str = Field(min_length=1, max_length=128)
    payload: CandlePayload
    payload_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_flags: tuple[str, ...] = ()

    @field_validator("exchange_ts", "received_ts")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        return _utc_datetime(value)

    @field_validator("quality_flags")
    @classmethod
    def normalize_quality_flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for flag in value:
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", flag) is None:
                raise ValueError("quality flags must be uppercase identifiers")
            if flag not in normalized:
                normalized.append(flag)
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def validate_sequence_and_checksum(self) -> MarketEvent:
        if (
            self.sequence_start is not None
            and self.sequence_end is not None
            and self.sequence_end < self.sequence_start
        ):
            raise ValueError("sequence_end must be greater than or equal to sequence_start")
        if not self.verify_checksum():
            raise ValueError("payload_checksum does not match payload")
        expected_id = uuid5(
            NAMESPACE_URL,
            canonical_json(
                _market_event_identity(
                    event_type=self.event_type,
                    venue=self.venue,
                    market_type=self.market_type,
                    instrument_id=self.instrument_id,
                    exchange_ts=self.exchange_ts,
                    sequence_start=self.sequence_start,
                    sequence_end=self.sequence_end,
                    source_mode=self.source_mode,
                    ingest_run_id=self.ingest_run_id,
                    payload_checksum=self.payload_checksum,
                )
            ),
        )
        if self.event_id != expected_id:
            raise ValueError("event_id does not match canonical event provenance")
        return self

    @classmethod
    def create(
        cls,
        *,
        trace_id: str,
        event_type: str,
        venue: str,
        market_type: str,
        instrument_id: str,
        exchange_ts: datetime,
        received_ts: datetime,
        source_mode: SourceMode,
        ingest_run_id: str,
        payload: CandlePayload | Mapping[str, Any],
        sequence_start: int | None = None,
        sequence_end: int | None = None,
        quality_flags: tuple[str, ...] | list[str] = (),
        event_id: UUID | str | None = None,
    ) -> MarketEvent:
        """Build a validated event with deterministic checksum and event ID.

        When ``event_id`` is omitted, the UUID is derived from exchange identity,
        source provenance, sequence, and payload.  Re-ingesting the same source
        event in the same run therefore produces the same idempotency key.
        """

        candle = payload if isinstance(payload, CandlePayload) else CandlePayload.model_validate(payload)
        checksum = compute_payload_checksum(candle)
        exchange_utc = _utc_datetime(exchange_ts)
        identity = _market_event_identity(
            event_type=event_type,
            venue=venue,
            market_type=market_type,
            instrument_id=instrument_id,
            exchange_ts=exchange_utc,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            source_mode=source_mode,
            ingest_run_id=ingest_run_id,
            payload_checksum=checksum,
        )
        resolved_id = (
            UUID(str(event_id))
            if event_id is not None
            else uuid5(NAMESPACE_URL, canonical_json(identity))
        )
        return cls(
            event_id=resolved_id,
            trace_id=trace_id,
            event_type=event_type,
            venue=venue,
            market_type=market_type,
            instrument_id=instrument_id,
            exchange_ts=exchange_utc,
            received_ts=_utc_datetime(received_ts),
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            source_mode=source_mode,
            ingest_run_id=ingest_run_id,
            payload=candle,
            payload_checksum=checksum,
            quality_flags=tuple(quality_flags),
        )

    def verify_checksum(self) -> bool:
        """Verify payload integrity without mutating the envelope."""

        return self.payload_checksum == compute_payload_checksum(self.payload)

    def canonical_json(self) -> str:
        """Return deterministic canonical JSON for transport or fixtures."""

        return canonical_json(self)
