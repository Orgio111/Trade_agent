"""Decimal-safe execution domain models.

The execution package intentionally contains no exchange client and no floating
point arithmetic.  These models are the boundary between an approved risk
decision and a paper broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
from typing import Any, Mapping

from packages.domain.events import SourceMode


_MISSING = object()


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def decimal_value(value: Decimal | int | str, *, field_name: str) -> Decimal:
    """Coerce an exact input to a finite :class:`Decimal`.

    Floats are rejected deliberately: accepting a binary float at the order
    boundary can silently change quantity, notional, fee, or limit-price math.
    """

    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{field_name} must be Decimal, int, or str; floats are unsafe")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} is not a valid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def optional_decimal(
    value: Decimal | int | str | None, *, field_name: str
) -> Decimal | None:
    if value is None:
        return None
    return decimal_value(value, field_name=field_name)


def canonical_decimal(value: Decimal | None) -> str | None:
    """Return one stable, exponent-free representation for hashing."""

    if value is None:
        return None
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).strip().lower()


def _coerce_source_mode(value: SourceMode | str) -> SourceMode:
    if isinstance(value, SourceMode):
        return value
    normalized = _enum_value(value)
    for candidate in SourceMode:
        if _enum_value(candidate) == normalized or candidate.name.lower() == normalized:
            return candidate
    raise ValueError(f"unsupported source mode: {value!r}")


def is_paper_mode(value: SourceMode | str) -> bool:
    normalized = _enum_value(value)
    name = getattr(value, "name", "").lower()
    return normalized in {"paper", "paper_live", "replay"} or name in {
        "paper",
        "paper_live",
        "replay",
    }


def _read_attr(obj: object, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _MISSING:
        return default
    joined = ", ".join(names)
    raise ValueError(f"required field missing (expected one of: {joined})")


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @classmethod
    def coerce(cls, value: OrderSide | str | Enum) -> OrderSide:
        if isinstance(value, cls):
            return value
        try:
            return cls(_enum_value(value))
        except ValueError as exc:
            raise ValueError(f"unsupported order side: {value!r}") from exc


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

    @classmethod
    def coerce(cls, value: OrderType | str | Enum) -> OrderType:
        if isinstance(value, cls):
            return value
        try:
            return cls(_enum_value(value))
        except ValueError as exc:
            raise ValueError(f"unsupported order type: {value!r}") from exc


class OrderStatus(str, Enum):
    PENDING_SUBMIT = "pending_submit"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in {self.FILLED, self.CANCELED, self.REJECTED}


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A durable instruction created only from an approved risk decision."""

    intent_id: str
    client_order_id: str
    account_id: str
    decision_id: str
    signal_id: str
    venue: str
    market_type: str
    instrument: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    source_mode: SourceMode
    limit_price: Decimal | None = None
    reference_price: Decimal | None = None
    created_at: datetime = field(default_factory=utc_now, compare=False)

    def __post_init__(self) -> None:
        for name in ("intent_id", "client_order_id", "account_id", "decision_id", "signal_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        venue = self.venue.strip().lower()
        market_type = self.market_type.strip().lower()
        instrument = self.instrument.strip().upper()
        if not venue or not market_type:
            raise ValueError("venue and market_type cannot be empty")
        if not instrument:
            raise ValueError("instrument cannot be empty")
        quantity = decimal_value(self.quantity, field_name="quantity")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        order_type = OrderType.coerce(self.order_type)
        limit_price = optional_decimal(self.limit_price, field_name="limit_price")
        reference_price = optional_decimal(self.reference_price, field_name="reference_price")
        if order_type is OrderType.LIMIT and (limit_price is None or limit_price <= 0):
            raise ValueError("positive limit_price is required for limit orders")
        if order_type is OrderType.MARKET and limit_price is not None:
            raise ValueError("market orders cannot carry a limit_price")
        if reference_price is not None and reference_price <= 0:
            raise ValueError("reference_price must be positive")

        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "side", OrderSide.coerce(self.side))
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "source_mode", _coerce_source_mode(self.source_mode))
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "reference_price", reference_price)
        object.__setattr__(self, "created_at", _aware_utc(self.created_at, field_name="created_at"))

    @classmethod
    def from_approved_decision(
        cls,
        decision: object,
        candidate: object,
        *,
        account_id: str,
        order_type: OrderType | str = OrderType.MARKET,
        limit_price: Decimal | int | str | None = None,
        created_at: datetime | None = None,
    ) -> OrderIntent:
        """Build an intent from the canonical risk decision and candidate.

        The deterministic IDs exclude wall-clock time. Rebuilding the same
        approved decision for the same account and execution parameters yields
        the same IDs across process restarts.
        """

        if _read_attr(decision, "approved", default=False) is not True:
            raise ValueError("risk decision is not approved")

        resolved_account = str(account_id).strip()
        decision_account = str(_read_attr(decision, "account_id")).strip()
        if not resolved_account or resolved_account != decision_account:
            raise ValueError("execution account_id does not match risk decision")

        content_hash = getattr(candidate, "content_hash", None)
        if not callable(content_hash):
            raise ValueError("candidate does not expose a canonical content hash")
        if str(_read_attr(decision, "candidate_hash")) != str(content_hash()):
            raise ValueError("candidate payload does not match risk decision")

        decision_signal_id = str(_read_attr(decision, "signal_id"))
        candidate_signal_id = str(_read_attr(candidate, "signal_id"))
        if decision_signal_id != candidate_signal_id:
            raise ValueError("risk decision signal_id does not match candidate signal_id")

        quantity = decimal_value(
            _read_attr(decision, "approved_quantity"), field_name="approved_quantity"
        )
        if quantity <= 0:
            raise ValueError("approved_quantity must be positive")

        resolved_type = OrderType.coerce(order_type)
        resolved_limit = optional_decimal(limit_price, field_name="limit_price")
        if resolved_type is OrderType.LIMIT and resolved_limit is None:
            resolved_limit = decimal_value(
                _read_attr(candidate, "reference_price"), field_name="reference_price"
            )

        payload = {
            "schema": "quantex.order-intent.v1",
            "account_id": resolved_account,
            "decision_id": str(_read_attr(decision, "decision_id")),
            "signal_id": candidate_signal_id,
            "venue": str(_read_attr(candidate, "venue")).strip().lower(),
            "market_type": str(_read_attr(candidate, "market_type")).strip().lower(),
            "instrument": str(_read_attr(candidate, "instrument", "symbol")).strip().upper(),
            "side": OrderSide.coerce(_read_attr(candidate, "side")).value,
            "order_type": resolved_type.value,
            "quantity": canonical_decimal(quantity),
            "source_mode": _enum_value(_read_attr(candidate, "source_mode")),
            "limit_price": canonical_decimal(resolved_limit),
            "reference_price": canonical_decimal(
                decimal_value(
                    _read_attr(candidate, "reference_price"), field_name="reference_price"
                )
            ),
        }
        if not payload["account_id"]:
            raise ValueError("account_id cannot be empty")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        intent_digest = hashlib.sha256(b"intent:" + encoded).hexdigest()
        client_digest = hashlib.sha256(b"client-order:" + encoded).hexdigest()

        return cls(
            intent_id=f"int_{intent_digest[:24]}",
            client_order_id=f"qx_{client_digest[:24]}",
            account_id=payload["account_id"],
            decision_id=payload["decision_id"],
            signal_id=candidate_signal_id,
            venue=payload["venue"],
            market_type=payload["market_type"],
            instrument=payload["instrument"],
            side=OrderSide(payload["side"]),
            order_type=resolved_type,
            quantity=quantity,
            source_mode=_coerce_source_mode(_read_attr(candidate, "source_mode")),
            limit_price=resolved_limit,
            reference_price=decimal_value(
                _read_attr(candidate, "reference_price"), field_name="reference_price"
            ),
            created_at=created_at or utc_now(),
        )


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    intent_id: str
    client_order_id: str
    broker_order_id: str
    instrument: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    filled_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        for name in ("fill_id", "intent_id", "client_order_id", "broker_order_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        quantity = decimal_value(self.quantity, field_name="fill.quantity")
        price = decimal_value(self.price, field_name="fill.price")
        fee = decimal_value(self.fee, field_name="fill.fee")
        if quantity <= 0 or price <= 0:
            raise ValueError("fill quantity and price must be positive")
        if fee < 0:
            raise ValueError("fill fee cannot be negative")
        instrument = self.instrument.strip().upper()
        if not instrument:
            raise ValueError("instrument cannot be empty")
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "side", OrderSide.coerce(self.side))
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", fee)
        object.__setattr__(self, "filled_at", _aware_utc(self.filled_at, field_name="filled_at"))

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class OrderRecord:
    intent_id: str
    client_order_id: str
    account_id: str
    venue: str
    market_type: str
    instrument: str
    side: OrderSide
    order_type: OrderType
    requested_quantity: Decimal
    status: OrderStatus
    broker_order_id: str | None = None
    limit_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None
    cumulative_fee: Decimal = Decimal("0")
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now, compare=False)

    def __post_init__(self) -> None:
        status = OrderStatus(self.status)
        side = OrderSide.coerce(self.side)
        order_type = OrderType.coerce(self.order_type)
        requested = decimal_value(self.requested_quantity, field_name="requested_quantity")
        filled = decimal_value(self.filled_quantity, field_name="filled_quantity")
        fee = decimal_value(self.cumulative_fee, field_name="cumulative_fee")
        avg = optional_decimal(self.average_fill_price, field_name="average_fill_price")
        limit = optional_decimal(self.limit_price, field_name="limit_price")
        venue = self.venue.strip().lower()
        market_type = self.market_type.strip().lower()
        instrument = self.instrument.strip().upper()
        if not venue or not market_type:
            raise ValueError("venue and market_type cannot be empty")
        if not instrument:
            raise ValueError("instrument cannot be empty")
        if requested <= 0:
            raise ValueError("requested_quantity must be positive")
        if filled < 0 or filled > requested:
            raise ValueError("filled_quantity must be between zero and requested_quantity")
        if fee < 0:
            raise ValueError("cumulative_fee cannot be negative")
        if (filled == 0) != (avg is None):
            raise ValueError("average_fill_price must be set exactly when fills exist")
        if avg is not None and avg <= 0:
            raise ValueError("average_fill_price must be positive")
        if order_type is OrderType.LIMIT and (limit is None or limit <= 0):
            raise ValueError("limit orders require a positive limit_price")
        if order_type is OrderType.MARKET and limit is not None:
            raise ValueError("market orders cannot carry a limit_price")
        if status is OrderStatus.FILLED and filled != requested:
            raise ValueError("filled orders must have their full requested quantity")
        if status is OrderStatus.PARTIALLY_FILLED and not (0 < filled < requested):
            raise ValueError("partially filled orders need a partial filled quantity")
        if status in {
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.OPEN,
            OrderStatus.REJECTED,
        } and filled != 0:
            raise ValueError(f"{status.value} orders cannot contain fills")
        if status is OrderStatus.CANCELED and filled == requested:
            raise ValueError("a fully filled order cannot be canceled")
        if filled == 0 and fee != 0:
            raise ValueError("an order without fills cannot have fees")

        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "requested_quantity", requested)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "average_fill_price", avg)
        object.__setattr__(self, "cumulative_fee", fee)
        object.__setattr__(self, "limit_price", limit)
        object.__setattr__(self, "created_at", _aware_utc(self.created_at, field_name="created_at"))
        object.__setattr__(self, "updated_at", _aware_utc(self.updated_at, field_name="updated_at"))

    @classmethod
    def pending(cls, intent: OrderIntent) -> OrderRecord:
        return cls(
            intent_id=intent.intent_id,
            client_order_id=intent.client_order_id,
            account_id=intent.account_id,
            venue=intent.venue,
            market_type=intent.market_type,
            instrument=intent.instrument,
            side=intent.side,
            order_type=intent.order_type,
            requested_quantity=intent.quantity,
            status=OrderStatus.PENDING_SUBMIT,
            limit_price=intent.limit_price,
            created_at=intent.created_at,
            updated_at=intent.created_at,
        )

    @property
    def remaining_quantity(self) -> Decimal:
        return self.requested_quantity - self.filled_quantity


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    venue: str
    market_type: str
    instrument: str
    bid: Decimal
    ask: Decimal
    last: Decimal | None = None
    available_bid_quantity: Decimal | None = None
    available_ask_quantity: Decimal | None = None
    observed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        venue = self.venue.strip().lower()
        market_type = self.market_type.strip().lower()
        instrument = self.instrument.strip().upper()
        bid = decimal_value(self.bid, field_name="bid")
        ask = decimal_value(self.ask, field_name="ask")
        last = optional_decimal(self.last, field_name="last")
        bid_qty = optional_decimal(
            self.available_bid_quantity, field_name="available_bid_quantity"
        )
        ask_qty = optional_decimal(
            self.available_ask_quantity, field_name="available_ask_quantity"
        )
        if not venue or not market_type or not instrument:
            raise ValueError("venue, market_type, and instrument cannot be empty")
        if bid <= 0 or ask <= 0 or ask < bid:
            raise ValueError("market snapshot requires positive bid <= ask")
        if last is not None and last <= 0:
            raise ValueError("last must be positive")
        if bid_qty is not None and bid_qty < 0:
            raise ValueError("available_bid_quantity cannot be negative")
        if ask_qty is not None and ask_qty < 0:
            raise ValueError("available_ask_quantity cannot be negative")
        object.__setattr__(self, "venue", venue)
        object.__setattr__(self, "market_type", market_type)
        object.__setattr__(self, "instrument", instrument)
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "last", last)
        object.__setattr__(self, "available_bid_quantity", bid_qty)
        object.__setattr__(self, "available_ask_quantity", ask_qty)
        object.__setattr__(self, "observed_at", _aware_utc(self.observed_at, field_name="observed_at"))

    def executable_price(self, side: OrderSide) -> Decimal:
        return self.ask if OrderSide.coerce(side) is OrderSide.BUY else self.bid

    def available_quantity(self, side: OrderSide) -> Decimal | None:
        return (
            self.available_ask_quantity
            if OrderSide.coerce(side) is OrderSide.BUY
            else self.available_bid_quantity
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    intent: OrderIntent
    order: OrderRecord
    fills: tuple[Fill, ...]
    duplicate: bool = False

    @property
    def total_fee(self) -> Decimal:
        return sum((fill.fee for fill in self.fills), Decimal("0"))


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    sequence: int
    event_type: str
    intent_id: str
    occurred_at: datetime
    details: Mapping[str, str | None]
