"""Bounded incremental features; no full-history reload per candle."""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from decimal import Decimal, localcontext
from statistics import median
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from packages.domain import CandlePayload, MarketEvent
from packages.domain.events import canonical_json, compute_payload_checksum


class FeatureSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_schema_version: Literal["1.0"] = "1.0"
    input_event_id: UUID
    state_version: int = Field(ge=1)
    symbol: str
    observed_at: datetime
    candle_count: int = Field(ge=1, le=200)
    session_high: Decimal
    session_low: Decimal
    session_vwap: Decimal
    atr: Decimal
    atr_median: Decimal
    volatility: Decimal
    ema_9: Decimal
    ema_21: Decimal
    ema_50: Decimal
    rsi_14: Decimal
    trend_direction: Literal["up", "down", "flat"]
    trend_strength: Decimal
    support: Decimal
    resistance: Decimal
    liquidity_proxy: Decimal
    market_data_age_seconds: Decimal
    fresh: bool
    checksum: str = Field(default="", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bind_checksum(self) -> FeatureSnapshot:
        payload = self.model_dump(exclude={"checksum"})
        checksum = compute_payload_checksum(payload)
        if self.checksum and self.checksum != checksum:
            raise ValueError("feature snapshot checksum mismatch")
        if not self.checksum:
            object.__setattr__(self, "checksum", checksum)
        return self

    def verify_checksum(self) -> bool:
        return self.checksum == compute_payload_checksum(
            self.model_dump(exclude={"checksum"})
        )

    def canonical_json(self) -> str:
        return canonical_json(self)


class IncrementalFeatureEngine:
    def __init__(self, *, max_age_seconds: int = 90) -> None:
        if max_age_seconds < 1:
            raise ValueError("max_age_seconds must be positive")
        self._max_age = Decimal(max_age_seconds)
        self._candles: deque[CandlePayload] = deque(maxlen=200)
        self._event_ids: deque[UUID] = deque(maxlen=200)
        self._version = 0
        self._last_snapshot: FeatureSnapshot | None = None
        self._session_high: Decimal | None = None
        self._session_low: Decimal | None = None
        self._session_volume = Decimal("0")
        self._session_turnover = Decimal("0")

    def update(
        self,
        event: MarketEvent,
        *,
        evaluated_at: datetime | None = None,
    ) -> FeatureSnapshot:
        if event.event_type != "market.candle":
            raise ValueError("feature engine accepts only candle events")
        if self._event_ids and event.event_id == self._event_ids[-1]:
            assert self._last_snapshot is not None
            return self._last_snapshot
        if event.event_id in self._event_ids:
            raise ValueError("out-of-order duplicate market event")
        if self._candles and event.payload.close_time <= self._candles[-1].close_time:
            raise ValueError("market candles must be strictly ordered")
        self._candles.append(event.payload)
        self._event_ids.append(event.event_id)
        self._version += 1
        self._session_high = (
            event.payload.high
            if self._session_high is None
            else max(self._session_high, event.payload.high)
        )
        self._session_low = (
            event.payload.low
            if self._session_low is None
            else min(self._session_low, event.payload.low)
        )
        self._session_volume += event.payload.volume
        self._session_turnover += event.payload.close * event.payload.volume
        now = (evaluated_at or event.received_ts).astimezone(UTC)
        age = Decimal(str((now - event.exchange_ts).total_seconds()))
        if age < 0:
            raise ValueError("market event is from the future")
        with localcontext() as context:
            context.prec = 50
            snapshot = self._snapshot(event, age=age)
        self._last_snapshot = snapshot
        return snapshot

    def _snapshot(self, event: MarketEvent, *, age: Decimal) -> FeatureSnapshot:
        candles = tuple(self._candles)
        closes = tuple(c.close for c in candles)
        true_ranges = tuple(c.high - c.low for c in candles)
        gains: list[Decimal] = []
        losses: list[Decimal] = []
        returns: list[Decimal] = []
        for previous, current in zip(closes, closes[1:], strict=False):
            delta = current - previous
            gains.append(max(delta, Decimal("0")))
            losses.append(max(-delta, Decimal("0")))
            returns.append(abs(delta / previous))
        recent_gains = gains[-14:]
        recent_losses = losses[-14:]
        avg_gain = sum(recent_gains, Decimal("0")) / Decimal(len(recent_gains) or 1)
        avg_loss = sum(recent_losses, Decimal("0")) / Decimal(len(recent_losses) or 1)
        rsi = (
            Decimal("100")
            if avg_loss == 0
            else Decimal("100") - Decimal("100") / (Decimal("1") + avg_gain / avg_loss)
        )
        ema9, ema21, ema50 = (
            self._ema(closes, window) for window in (9, 21, 50)
        )
        trend: Literal["up", "down", "flat"] = (
            "up" if ema9 > ema21 > ema50 else "down" if ema9 < ema21 < ema50 else "flat"
        )
        vwap = (
            self._session_turnover / self._session_volume
            if self._session_volume
            else closes[-1]
        )
        recent_ranges = true_ranges[-14:]
        atr = sum(recent_ranges, Decimal("0")) / Decimal(len(recent_ranges))
        return FeatureSnapshot(
            input_event_id=event.event_id,
            state_version=self._version,
            symbol=event.instrument_id,
            observed_at=event.exchange_ts,
            candle_count=len(candles),
            session_high=self._session_high or candles[-1].high,
            session_low=self._session_low or candles[-1].low,
            session_vwap=vwap,
            atr=atr,
            atr_median=Decimal(str(median(true_ranges))),
            volatility=(
                sum(returns[-20:], Decimal("0")) / Decimal(len(returns[-20:]))
                if returns
                else Decimal("0")
            ),
            ema_9=ema9,
            ema_21=ema21,
            ema_50=ema50,
            rsi_14=rsi,
            trend_direction=trend,
            trend_strength=abs(ema9 - ema50) / closes[-1],
            support=min(c.low for c in candles[-20:]),
            resistance=max(c.high for c in candles[-20:]),
            liquidity_proxy=(
                sum((c.volume for c in candles[-20:]), Decimal("0"))
                / Decimal(len(candles[-20:]))
            ),
            market_data_age_seconds=age,
            fresh=age <= self._max_age,
        )

    @staticmethod
    def _ema(values: tuple[Decimal, ...], window: int) -> Decimal:
        alpha = Decimal("2") / Decimal(window + 1)
        result = values[0]
        for value in values[1:]:
            result = alpha * value + (Decimal("1") - alpha) * result
        return result

    def checkpoint_json(self) -> str:
        return canonical_json(
            {
                "version": self._version,
                "max_age_seconds": self._max_age,
                "event_ids": tuple(self._event_ids),
                "candles": tuple(self._candles),
                "session_high": self._session_high,
                "session_low": self._session_low,
                "session_volume": self._session_volume,
                "session_turnover": self._session_turnover,
                "last_snapshot": self._last_snapshot,
            }
        )

    @classmethod
    def from_checkpoint(cls, encoded: str) -> IncrementalFeatureEngine:
        import json

        payload = json.loads(encoded)
        engine = cls(max_age_seconds=int(payload["max_age_seconds"]))
        candles = tuple(CandlePayload.model_validate(item) for item in payload["candles"])
        event_ids = tuple(UUID(item) for item in payload["event_ids"])
        if len(candles) != len(event_ids) or len(candles) > 200:
            raise ValueError("feature checkpoint is inconsistent")
        engine._candles.extend(candles)
        engine._event_ids.extend(event_ids)
        engine._version = int(payload["version"])
        engine._session_high = Decimal(payload["session_high"])
        engine._session_low = Decimal(payload["session_low"])
        engine._session_volume = Decimal(payload["session_volume"])
        engine._session_turnover = Decimal(payload["session_turnover"])
        if payload.get("last_snapshot") is not None:
            engine._last_snapshot = FeatureSnapshot.model_validate(
                payload["last_snapshot"]
            )
        if engine._version < len(candles):
            raise ValueError("feature checkpoint version is inconsistent")
        return engine
