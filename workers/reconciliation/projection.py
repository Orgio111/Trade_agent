"""Deterministic projection of paper fills into risk portfolio state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from typing import Literal, Mapping, Sequence

from packages.risk import PortfolioState


@dataclass(frozen=True, slots=True)
class FillRow:
    fill_id: str
    instrument: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    price: Decimal
    fee: Decimal
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not self.fill_id.strip() or not self.instrument.strip():
            raise ValueError("fill identity cannot be blank")
        if self.quantity <= 0 or self.price <= 0 or self.fee < 0:
            raise ValueError("fill amounts are invalid")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("fill occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PositionProjection:
    quantity: Decimal
    average_entry_price: Decimal | None
    realized_pnl: Decimal
    fees: Decimal


def project_position(fills: Sequence[FillRow]) -> PositionProjection:
    """Project one signed position; exact close quantity handles flips."""

    quantity = Decimal("0")
    average: Decimal | None = None
    realized = Decimal("0")
    fees = Decimal("0")
    for fill in sorted(fills, key=lambda row: (row.occurred_at, row.fill_id)):
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        fees += fill.fee
        if quantity == 0 or (quantity > 0) == (signed > 0):
            old_notional = abs(quantity) * (average or Decimal("0"))
            quantity += signed
            average = (old_notional + fill.quantity * fill.price) / abs(quantity)
            continue

        closed = min(abs(quantity), abs(signed))
        if quantity > 0:
            realized += closed * (fill.price - (average or fill.price))
        else:
            realized += closed * ((average or fill.price) - fill.price)
        remaining = quantity + signed
        if remaining == 0:
            quantity, average = Decimal("0"), None
        elif (remaining > 0) == (quantity > 0):
            quantity = remaining
        else:
            quantity, average = remaining, fill.price
    return PositionProjection(quantity, average, realized, fees)


class PaperPortfolioProjector:
    def __init__(self, initial_equity: Decimal) -> None:
        if not initial_equity.is_finite() or initial_equity <= 0:
            raise ValueError("initial_equity must be finite and positive")
        self._initial_equity = initial_equity

    def project(
        self,
        *,
        account_id: str,
        fills: Sequence[FillRow],
        marks: Mapping[str, Decimal],
        reconciled_at: datetime,
    ) -> PortfolioState:
        if not account_id.strip():
            raise ValueError("account_id cannot be blank")
        if reconciled_at.tzinfo is None or reconciled_at.utcoffset() is None:
            raise ValueError("reconciled_at must be timezone-aware")

        cash = self._initial_equity
        positions: dict[str, Decimal] = {}
        losses = 0
        for fill in sorted(fills, key=lambda row: (row.occurred_at, row.fill_id)):
            symbol = fill.instrument.upper()
            signed = fill.quantity if fill.side == "buy" else -fill.quantity
            quantity = positions.get(symbol, Decimal("0")) + signed
            if quantity < 0:
                raise ValueError(f"negative paper position for {symbol}")
            positions[symbol] = quantity
            notional = fill.quantity * fill.price
            cash += -notional if fill.side == "buy" else notional
            cash -= fill.fee
            if cash < 0:
                raise ValueError("paper cash became negative")

        gross_exposure = Decimal("0")
        for symbol, quantity in positions.items():
            if quantity == 0:
                continue
            mark = marks.get(symbol)
            if mark is None or not mark.is_finite() or mark <= 0:
                raise ValueError(f"authoritative mark missing for {symbol}")
            gross_exposure += quantity * mark

        equity = cash + gross_exposure
        pnl = equity - self._initial_equity
        if pnl < 0:
            losses = 1
        identity = json.dumps(
            {
                "account_id": account_id,
                "cash": str(cash),
                "equity": str(equity),
                "fills": [fill.fill_id for fill in fills],
                "marks": {key: str(marks[key]) for key in sorted(marks)},
                "reconciled_at": reconciled_at.astimezone(UTC).isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        state_id = f"paper-{hashlib.sha256(identity).hexdigest()[:32]}"
        return PortfolioState(
            account_id=account_id,
            state_id=state_id,
            equity=equity,
            cash=cash,
            daily_pnl=pnl,
            weekly_pnl=pnl,
            consecutive_losses=losses,
            open_positions=sum(quantity > 0 for quantity in positions.values()),
            gross_exposure=gross_exposure,
            reconciled_at=reconciled_at,
            kill_switch_active=False,
        )
