"""Versioned, immutable policy models for deterministic risk evaluation."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


PositiveDecimal = Annotated[
    Decimal,
    Field(gt=Decimal("0"), allow_inf_nan=False),
]
NonNegativeDecimal = Annotated[
    Decimal,
    Field(ge=Decimal("0"), allow_inf_nan=False),
]
UnitFraction = Annotated[
    Decimal,
    Field(ge=Decimal("0"), le=Decimal("1"), allow_inf_nan=False),
]
PositiveUnitFraction = Annotated[
    Decimal,
    Field(gt=Decimal("0"), le=Decimal("1"), allow_inf_nan=False),
]


class FrozenRiskModel(BaseModel):
    """Shared model settings for auditable risk inputs and outputs."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


class RiskPolicy(FrozenRiskModel):
    """A conservative, versioned set of hard portfolio and order limits.

    All percentages are fractions: ``Decimal("0.005")`` means 0.5%.
    The defaults are the paper-trading starting policy from the build plan;
    they are not a claim that a strategy is safe or profitable.
    """

    version: str = Field(min_length=1)
    paper_only: bool = True

    max_risk_per_trade_fraction: PositiveUnitFraction = Decimal("0.005")
    max_daily_loss_fraction: PositiveUnitFraction = Decimal("0.02")
    max_weekly_loss_fraction: PositiveUnitFraction = Decimal("0.04")
    max_consecutive_losses: int = Field(default=3, ge=1)
    max_open_positions: int = Field(default=3, ge=1)
    max_leverage: PositiveDecimal = Decimal("1")

    max_data_age_seconds: NonNegativeDecimal = Decimal("5")
    max_portfolio_age_seconds: NonNegativeDecimal = Decimal("30")
    max_spread_bps: NonNegativeDecimal = Decimal("10")
    max_slippage_bps: NonNegativeDecimal = Decimal("10")
    min_confidence: UnitFraction = Decimal("0")
    min_notional: NonNegativeDecimal = Decimal("0")

    @field_validator("version")
    @classmethod
    def version_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("version must not be blank")
        return value


class InstrumentConstraints(FrozenRiskModel):
    """Venue constraints needed to turn risk budget into executable quantity."""

    venue: str = Field(min_length=1)
    market_type: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    tick_size: PositiveDecimal
    step_size: PositiveDecimal
    min_quantity: PositiveDecimal
    min_notional: PositiveDecimal

    @field_validator("venue", "market_type")
    @classmethod
    def market_identity_must_not_be_blank(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("market identity must not be blank")
        return value

    @field_validator("instrument")
    @classmethod
    def instrument_must_not_be_blank(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("instrument must not be blank")
        return value
