"""Immutable inputs and outputs for the deterministic risk boundary."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from packages.domain.events import SourceMode

from .policy import (
    FrozenRiskModel,
    NonNegativeDecimal,
    PositiveDecimal,
    PositiveUnitFraction,
    UnitFraction,
)


class RiskReason(StrEnum):
    """Stable machine-readable reason codes persisted with every decision."""

    APPROVED = "APPROVED"

    LIVE_MODE_BLOCKED = "LIVE_MODE_BLOCKED"
    NON_EXECUTABLE_SOURCE_MODE = "NON_EXECUTABLE_SOURCE_MODE"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"

    MISSING_MARKET_DATA = "MISSING_MARKET_DATA"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    MISSING_PORTFOLIO_STATE = "MISSING_PORTFOLIO_STATE"
    STALE_PORTFOLIO_STATE = "STALE_PORTFOLIO_STATE"
    PORTFOLIO_STATE_FROM_FUTURE = "PORTFOLIO_STATE_FROM_FUTURE"
    INSTRUMENT_MISMATCH = "INSTRUMENT_MISMATCH"
    VENUE_MISMATCH = "VENUE_MISMATCH"
    MARKET_TYPE_MISMATCH = "MARKET_TYPE_MISMATCH"

    DAILY_LOSS_LIMIT_REACHED = "DAILY_LOSS_LIMIT_REACHED"
    WEEKLY_LOSS_LIMIT_REACHED = "WEEKLY_LOSS_LIMIT_REACHED"
    CONSECUTIVE_LOSS_LIMIT_REACHED = "CONSECUTIVE_LOSS_LIMIT_REACHED"
    OPEN_POSITION_LIMIT_REACHED = "OPEN_POSITION_LIMIT_REACHED"
    MAX_LEVERAGE_EXCEEDED = "MAX_LEVERAGE_EXCEEDED"

    MISSING_CONFIDENCE = "MISSING_CONFIDENCE"
    CONFIDENCE_BELOW_MINIMUM = "CONFIDENCE_BELOW_MINIMUM"
    MISSING_SPREAD = "MISSING_SPREAD"
    SPREAD_LIMIT_EXCEEDED = "SPREAD_LIMIT_EXCEEDED"
    MISSING_SLIPPAGE = "MISSING_SLIPPAGE"
    SLIPPAGE_LIMIT_EXCEEDED = "SLIPPAGE_LIMIT_EXCEEDED"

    INVALID_STOP = "INVALID_STOP"
    INVALID_TAKE_PROFIT = "INVALID_TAKE_PROFIT"
    PRICE_NOT_TICK_ALIGNED = "PRICE_NOT_TICK_ALIGNED"
    INVALID_MANUAL_RISK = "INVALID_MANUAL_RISK"
    MANUAL_RISK_INCREASE_NOT_ALLOWED = "MANUAL_RISK_INCREASE_NOT_ALLOWED"
    MANUAL_RISK_DISABLED = "MANUAL_RISK_DISABLED"

    QUANTITY_ROUNDS_TO_ZERO = "QUANTITY_ROUNDS_TO_ZERO"
    MIN_QUANTITY_NOT_MET = "MIN_QUANTITY_NOT_MET"
    MIN_NOTIONAL_NOT_MET = "MIN_NOTIONAL_NOT_MET"

    POLICY_RISK_CAPPED = "POLICY_RISK_CAPPED"
    MANUAL_RISK_REDUCED = "MANUAL_RISK_REDUCED"
    EXPOSURE_CAPPED = "EXPOSURE_CAPPED"
    QUANTITY_STEP_ROUNDED = "QUANTITY_STEP_ROUNDED"


class CandidateSignal(FrozenRiskModel):
    """A strategy proposal; it is not permission to place an order."""

    trace_id: str = Field(min_length=1)
    signal_id: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    strategy_version: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    market_type: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    side: Literal["buy", "sell"]

    reference_price: PositiveDecimal
    stop_price: PositiveDecimal
    take_profit_price: PositiveDecimal | None = None

    confidence: UnitFraction | None
    spread_bps: NonNegativeDecimal | None
    expected_slippage_bps: NonNegativeDecimal | None
    data_age_seconds: NonNegativeDecimal | None
    source_mode: SourceMode
    requested_risk_fraction: PositiveUnitFraction

    @field_validator("trace_id", "signal_id", "strategy_id", "strategy_version")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("identifier must not be blank")
        return value

    @field_validator("venue", "market_type")
    @classmethod
    def normalize_market_identity(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("market identity must not be blank")
        return value

    @field_validator("instrument")
    @classmethod
    def normalize_instrument(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("instrument must not be blank")
        return value

    def content_hash(self) -> str:
        """Bind a risk verdict to the exact immutable candidate payload."""

        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class PortfolioState(FrozenRiskModel):
    """Authoritative portfolio projection produced by the reconciled ledger."""

    account_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    equity: PositiveDecimal
    cash: NonNegativeDecimal
    daily_pnl: Decimal = Field(allow_inf_nan=False)
    weekly_pnl: Decimal = Field(allow_inf_nan=False)
    consecutive_losses: int = Field(ge=0)
    open_positions: int = Field(ge=0)
    gross_exposure: NonNegativeDecimal
    reconciled_at: datetime | None
    kill_switch_active: bool

    @field_validator("account_id", "state_id")
    @classmethod
    def state_id_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("state_id must not be blank")
        return value

    @field_validator("reconciled_at")
    @classmethod
    def reconciled_at_must_be_timezone_aware(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("reconciled_at must be timezone-aware")
        return value.astimezone(UTC) if value is not None else None


class RiskDecision(FrozenRiskModel):
    """Immutable, persistence-ready output of a single policy evaluation."""

    decision_id: str = Field(pattern=r"^risk_[0-9a-f]{24}$")
    trace_id: str = Field(min_length=1)
    signal_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved: bool
    reason_codes: tuple[RiskReason, ...] = Field(min_length=1)

    approved_quantity: NonNegativeDecimal
    approved_notional: NonNegativeDecimal
    approved_risk_amount: NonNegativeDecimal
    approved_risk_fraction: UnitFraction

    policy_version: str = Field(min_length=1)
    portfolio_state_id: str = Field(min_length=1)
    evaluated_at: datetime

    @field_validator("evaluated_at")
    @classmethod
    def evaluated_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def decision_values_must_match_verdict(self) -> "RiskDecision":
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason_codes must not contain duplicates")

        numeric_values = (
            self.approved_quantity,
            self.approved_notional,
            self.approved_risk_amount,
            self.approved_risk_fraction,
        )
        if self.approved:
            if RiskReason.APPROVED not in self.reason_codes:
                raise ValueError("approved decision must contain APPROVED reason")
            if any(value <= Decimal("0") for value in numeric_values):
                raise ValueError("approved decision values must be positive")
        else:
            if RiskReason.APPROVED in self.reason_codes:
                raise ValueError("rejected decision must not contain APPROVED reason")
            if any(value != Decimal("0") for value in numeric_values):
                raise ValueError("rejected decision values must all be zero")
        return self
