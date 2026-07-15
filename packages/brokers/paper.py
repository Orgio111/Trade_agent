"""Deterministic, paper-only execution adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
import hashlib
from threading import RLock

from packages.brokers.base import (
    DuplicateBrokerIntent,
    InvalidOrderError,
    PaperModeRequired,
)
from packages.execution.models import (
    Fill,
    MarketSnapshot,
    OrderIntent,
    OrderRecord,
    OrderSide,
    OrderStatus,
    OrderType,
    decimal_value,
    is_paper_mode,
)


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Paper fill assumptions expressed as exact decimal rates."""

    fee_rate: Decimal = Decimal("0.001")
    slippage_bps: Decimal = Decimal("0")
    liquidity_participation: Decimal | None = None

    def __post_init__(self) -> None:
        fee_rate = decimal_value(self.fee_rate, field_name="fee_rate")
        slippage = decimal_value(self.slippage_bps, field_name="slippage_bps")
        participation = (
            None
            if self.liquidity_participation is None
            else decimal_value(
                self.liquidity_participation, field_name="liquidity_participation"
            )
        )
        if not (Decimal("0") <= fee_rate < Decimal("1")):
            raise ValueError("fee_rate must be in [0, 1)")
        if not (Decimal("0") <= slippage < Decimal("10000")):
            raise ValueError("slippage_bps must be in [0, 10000)")
        if participation is not None and not (Decimal("0") < participation <= Decimal("1")):
            raise ValueError("liquidity_participation must be in (0, 1]")
        object.__setattr__(self, "fee_rate", fee_rate)
        object.__setattr__(self, "slippage_bps", slippage)
        object.__setattr__(self, "liquidity_participation", participation)


class PaperBroker:
    """A deterministic broker with no network or live-order capability.

    When ``liquidity_participation`` is ``None``, eligible orders fill their
    entire remaining quantity. When configured, each snapshot can fill at most
    ``visible_side_liquidity * participation`` and therefore supports explicit
    partial-fill scenarios.
    """

    def __init__(self, config: PaperBrokerConfig | None = None) -> None:
        self.config = config or PaperBrokerConfig()
        self._lock = RLock()
        self._orders: dict[str, OrderRecord] = {}
        self._intent_ids: set[str] = set()
        self._client_order_ids: set[str] = set()
        self._fills: list[Fill] = []
        self._fill_sequences: dict[str, int] = {}

    def submit_order(self, intent: OrderIntent, market: MarketSnapshot) -> OrderRecord:
        with self._lock:
            self._validate_submission(intent, market)
            broker_order_id = self._broker_order_id(intent.client_order_id)
            record = OrderRecord(
                intent_id=intent.intent_id,
                client_order_id=intent.client_order_id,
                account_id=intent.account_id,
                venue=intent.venue,
                market_type=intent.market_type,
                instrument=intent.instrument,
                side=intent.side,
                order_type=intent.order_type,
                requested_quantity=intent.quantity,
                status=OrderStatus.OPEN,
                broker_order_id=broker_order_id,
                limit_price=intent.limit_price,
                created_at=intent.created_at,
                updated_at=market.observed_at,
            )
            self._intent_ids.add(intent.intent_id)
            self._client_order_ids.add(intent.client_order_id)
            self._orders[intent.client_order_id] = record
            self._try_fill(intent.client_order_id, market)
            return self._orders[intent.client_order_id]

    def process_market(self, market: MarketSnapshot) -> tuple[OrderRecord, ...]:
        """Apply a new snapshot to all eligible resting/partial orders."""

        with self._lock:
            updated: list[OrderRecord] = []
            for client_order_id, record in tuple(self._orders.items()):
                if (
                    record.venue != market.venue
                    or record.market_type != market.market_type
                    or record.instrument != market.instrument
                    or record.status.terminal
                ):
                    continue
                before = record
                self._try_fill(client_order_id, market)
                after = self._orders[client_order_id]
                if after != before:
                    updated.append(after)
            return tuple(updated)

    def get_order(self, client_order_id: str) -> OrderRecord | None:
        with self._lock:
            return self._orders.get(client_order_id)

    def list_orders(self) -> tuple[OrderRecord, ...]:
        with self._lock:
            return tuple(self._orders.values())

    def list_fills(self, client_order_id: str | None = None) -> tuple[Fill, ...]:
        with self._lock:
            if client_order_id is None:
                return tuple(self._fills)
            return tuple(
                fill for fill in self._fills if fill.client_order_id == client_order_id
            )

    def _validate_submission(self, intent: OrderIntent, market: MarketSnapshot) -> None:
        if not is_paper_mode(intent.source_mode):
            raise PaperModeRequired(
                "PaperBroker accepts only SourceMode.PAPER_LIVE or SourceMode.REPLAY intents"
            )
        if intent.intent_id in self._intent_ids:
            raise DuplicateBrokerIntent(f"duplicate intent_id: {intent.intent_id}")
        if intent.client_order_id in self._client_order_ids:
            raise DuplicateBrokerIntent(
                f"duplicate client_order_id: {intent.client_order_id}"
            )
        if (
            intent.venue != market.venue
            or intent.market_type != market.market_type
            or intent.instrument != market.instrument
        ):
            raise InvalidOrderError(
                "intent venue/market/instrument does not match market snapshot"
            )

    def _try_fill(self, client_order_id: str, market: MarketSnapshot) -> None:
        with localcontext() as context:
            context.prec = 50
            self._try_fill_with_context(client_order_id, market)

    def _try_fill_with_context(
        self, client_order_id: str, market: MarketSnapshot
    ) -> None:
        record = self._orders[client_order_id]
        if not self._is_marketable(record, market):
            return
        quantity = self._fillable_quantity(record, market)
        if quantity <= 0:
            return

        base_price = market.executable_price(record.side)
        slippage_rate = self.config.slippage_bps / Decimal("10000")
        if record.side is OrderSide.BUY:
            price = base_price * (Decimal("1") + slippage_rate)
            if record.limit_price is not None:
                price = min(price, record.limit_price)
        else:
            price = base_price * (Decimal("1") - slippage_rate)
            if record.limit_price is not None:
                price = max(price, record.limit_price)

        fee = price * quantity * self.config.fee_rate
        fill_sequence = self._fill_sequences.get(client_order_id, 0) + 1
        self._fill_sequences[client_order_id] = fill_sequence
        fill = Fill(
            fill_id=f"fill_{record.broker_order_id}_{fill_sequence}",
            intent_id=record.intent_id,
            client_order_id=record.client_order_id,
            broker_order_id=record.broker_order_id or "",
            instrument=record.instrument,
            side=record.side,
            quantity=quantity,
            price=price,
            fee=fee,
            filled_at=market.observed_at,
        )

        old_notional = (
            record.average_fill_price * record.filled_quantity
            if record.average_fill_price is not None
            else Decimal("0")
        )
        filled_quantity = record.filled_quantity + quantity
        average_price = (old_notional + fill.notional) / filled_quantity
        status = (
            OrderStatus.FILLED
            if filled_quantity == record.requested_quantity
            else OrderStatus.PARTIALLY_FILLED
        )
        self._fills.append(fill)
        self._orders[client_order_id] = replace(
            record,
            status=status,
            filled_quantity=filled_quantity,
            average_fill_price=average_price,
            cumulative_fee=record.cumulative_fee + fee,
            updated_at=market.observed_at,
        )

    def _is_marketable(self, record: OrderRecord, market: MarketSnapshot) -> bool:
        if record.order_type is OrderType.MARKET:
            return True
        if record.limit_price is None:
            raise InvalidOrderError("limit order is missing limit_price")
        if record.side is OrderSide.BUY:
            return record.limit_price >= market.ask
        return record.limit_price <= market.bid

    def _fillable_quantity(self, record: OrderRecord, market: MarketSnapshot) -> Decimal:
        remaining = record.remaining_quantity
        participation = self.config.liquidity_participation
        if participation is None:
            return remaining
        visible = market.available_quantity(record.side)
        if visible is None:
            return Decimal("0")
        return min(remaining, visible * participation)

    @staticmethod
    def _broker_order_id(client_order_id: str) -> str:
        digest = hashlib.sha256(f"paper:{client_order_id}".encode("utf-8")).hexdigest()
        return f"paper_{digest[:20]}"
