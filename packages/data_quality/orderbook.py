"""Decimal-safe, fail-closed L2 order-book reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

from .sequence import validate_sequence
from .verdicts import QualityCode, QualityReason, QualityVerdict

BookLevelInput = Sequence[Any] | Mapping[str, Any]


class UnsynchronizedOrderBookError(RuntimeError):
    """Raised when a consumer tries to read a book that requires resnapshot."""


@dataclass(frozen=True, slots=True)
class OrderBookView:
    sequence: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]

    @property
    def best_bid(self) -> tuple[Decimal, Decimal]:
        return self.bids[0]

    @property
    def best_ask(self) -> tuple[Decimal, Decimal]:
        return self.asks[0]


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("book level contains an invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("book level decimal must be finite")
    return result


def _parse_level(level: BookLevelInput) -> tuple[Decimal, Decimal]:
    if isinstance(level, Mapping):
        if "price" not in level or not ({"quantity", "qty"} & level.keys()):
            raise ValueError("book level object requires price and quantity")
        price = _decimal(level["price"])
        quantity = _decimal(level.get("quantity", level.get("qty")))
    elif isinstance(level, Sequence) and not isinstance(level, (str, bytes)) and len(level) == 2:
        price = _decimal(level[0])
        quantity = _decimal(level[1])
    else:
        raise ValueError("book level must be a two-item pair or object")
    if price <= 0 or quantity < 0:
        raise ValueError("book level price must be positive and quantity non-negative")
    return price, quantity


def _parse_levels(levels: Iterable[BookLevelInput]) -> list[tuple[Decimal, Decimal]]:
    return [_parse_level(level) for level in levels]


class OrderBookBuilder:
    """Reconstruct an L2 book and require a new snapshot after any gap."""

    def __init__(self) -> None:
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._sequence: int | None = None
        self._synchronized = False

    @property
    def is_synchronized(self) -> bool:
        return self._synchronized

    @property
    def synchronized(self) -> bool:
        return self._synchronized

    @property
    def last_sequence(self) -> int | None:
        return self._sequence

    @property
    def best_bid(self) -> tuple[Decimal, Decimal] | None:
        if not self._synchronized or not self._bids:
            return None
        price = max(self._bids)
        return price, self._bids[price]

    @property
    def best_ask(self) -> tuple[Decimal, Decimal] | None:
        if not self._synchronized or not self._asks:
            return None
        price = min(self._asks)
        return price, self._asks[price]

    def _invalidate(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._sequence = None
        self._synchronized = False

    @staticmethod
    def _book_verdict(bids: Mapping[Decimal, Decimal], asks: Mapping[Decimal, Decimal]) -> QualityVerdict:
        if not bids or not asks:
            return QualityVerdict.reject(
                QualityReason(QualityCode.INCOMPLETE_ORDER_BOOK, "both sides of the order book are required")
            )
        if max(bids) >= min(asks):
            return QualityVerdict.reject(
                QualityReason(QualityCode.CROSSED_ORDER_BOOK, "best bid must be below best ask")
            )
        return QualityVerdict.pass_()

    def apply_snapshot(
        self,
        *,
        sequence: int,
        bids: Iterable[BookLevelInput],
        asks: Iterable[BookLevelInput],
    ) -> QualityVerdict:
        sequence_verdict = validate_sequence(None, sequence, sequence, snapshot=True)
        if not sequence_verdict.accepted:
            self._invalidate()
            return sequence_verdict
        try:
            parsed_bids = _parse_levels(bids)
            parsed_asks = _parse_levels(asks)
        except (TypeError, ValueError) as exc:
            self._invalidate()
            return QualityVerdict.reject(
                QualityReason(QualityCode.INVALID_BOOK_LEVEL, str(exc))
            )
        next_bids = {price: quantity for price, quantity in parsed_bids if quantity > 0}
        next_asks = {price: quantity for price, quantity in parsed_asks if quantity > 0}
        book_verdict = self._book_verdict(next_bids, next_asks)
        if not book_verdict.accepted:
            self._invalidate()
            return book_verdict
        self._bids = next_bids
        self._asks = next_asks
        self._sequence = sequence
        self._synchronized = True
        return QualityVerdict.pass_()

    def apply_delta(
        self,
        *,
        sequence_start: int,
        sequence_end: int,
        bids: Iterable[BookLevelInput] = (),
        asks: Iterable[BookLevelInput] = (),
    ) -> QualityVerdict:
        if not self._synchronized or self._sequence is None:
            return QualityVerdict.reject(
                QualityReason(QualityCode.BOOK_NOT_SYNCHRONIZED, "a valid snapshot is required before deltas")
            )
        sequence_verdict = validate_sequence(self._sequence, sequence_start, sequence_end)
        if not sequence_verdict.accepted:
            if QualityCode.SEQUENCE_GAP in sequence_verdict.codes:
                self._invalidate()
            return sequence_verdict
        try:
            parsed_bids = _parse_levels(bids)
            parsed_asks = _parse_levels(asks)
        except (TypeError, ValueError) as exc:
            self._invalidate()
            return QualityVerdict.reject(QualityReason(QualityCode.INVALID_BOOK_LEVEL, str(exc)))

        next_bids = dict(self._bids)
        next_asks = dict(self._asks)
        for price, quantity in parsed_bids:
            if quantity == 0:
                next_bids.pop(price, None)
            else:
                next_bids[price] = quantity
        for price, quantity in parsed_asks:
            if quantity == 0:
                next_asks.pop(price, None)
            else:
                next_asks[price] = quantity
        book_verdict = self._book_verdict(next_bids, next_asks)
        if not book_verdict.accepted:
            self._invalidate()
            return book_verdict

        self._bids = next_bids
        self._asks = next_asks
        self._sequence = sequence_end
        return QualityVerdict.pass_()

    def view(self, *, depth: int | None = None) -> OrderBookView:
        if not self._synchronized or self._sequence is None:
            raise UnsynchronizedOrderBookError("order book is not synchronized; request a new snapshot")
        if depth is not None and depth <= 0:
            raise ValueError("depth must be positive")
        bids = sorted(self._bids.items(), reverse=True)
        asks = sorted(self._asks.items())
        if depth is not None:
            bids = bids[:depth]
            asks = asks[:depth]
        return OrderBookView(self._sequence, tuple(bids), tuple(asks))

