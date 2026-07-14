"""Reason-coded validation verdicts shared by all quality gates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class QualityCode(str, Enum):
    OK = "OK"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_NUMBER = "INVALID_NUMBER"
    NON_FINITE_NUMBER = "NON_FINITE_NUMBER"
    NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
    NEGATIVE_VOLUME = "NEGATIVE_VOLUME"
    HIGH_BELOW_PRICE = "HIGH_BELOW_PRICE"
    LOW_ABOVE_PRICE = "LOW_ABOVE_PRICE"
    INVALID_CANDLE_WINDOW = "INVALID_CANDLE_WINDOW"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    TIMESTAMP_NOT_UTC = "TIMESTAMP_NOT_UTC"
    EVENT_FROM_FUTURE = "EVENT_FROM_FUTURE"
    RECEIVED_BEFORE_EXCHANGE = "RECEIVED_BEFORE_EXCHANGE"
    EXCESSIVE_TRANSPORT_LAG = "EXCESSIVE_TRANSPORT_LAG"
    STALE_EVENT = "STALE_EVENT"
    INVALID_SEQUENCE = "INVALID_SEQUENCE"
    SEQUENCE_NOT_INITIALIZED = "SEQUENCE_NOT_INITIALIZED"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    STALE_SEQUENCE = "STALE_SEQUENCE"
    BOOK_NOT_SYNCHRONIZED = "BOOK_NOT_SYNCHRONIZED"
    INVALID_BOOK_LEVEL = "INVALID_BOOK_LEVEL"
    INCOMPLETE_ORDER_BOOK = "INCOMPLETE_ORDER_BOOK"
    CROSSED_ORDER_BOOK = "CROSSED_ORDER_BOOK"


@dataclass(frozen=True, slots=True)
class QualityReason:
    code: QualityCode
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class QualityVerdict:
    """An immutable allow/reject decision with stable machine-readable codes."""

    accepted: bool
    reasons: tuple[QualityReason, ...] = ()

    def __post_init__(self) -> None:
        if self.accepted and self.reasons:
            raise ValueError("an accepted verdict cannot contain rejection reasons")
        if not self.accepted and not self.reasons:
            raise ValueError("a rejected verdict must contain at least one reason")

    @property
    def is_valid(self) -> bool:
        return self.accepted

    @property
    def ok(self) -> bool:
        return self.accepted

    @property
    def codes(self) -> tuple[QualityCode, ...]:
        return (QualityCode.OK,) if self.accepted else tuple(reason.code for reason in self.reasons)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(code.value for code in self.codes)

    @classmethod
    def pass_(cls) -> QualityVerdict:
        return cls(accepted=True)

    @classmethod
    def reject(cls, *reasons: QualityReason) -> QualityVerdict:
        return cls(accepted=False, reasons=tuple(reasons))

    @classmethod
    def from_reasons(cls, reasons: Iterable[QualityReason]) -> QualityVerdict:
        collected = tuple(reasons)
        return cls.pass_() if not collected else cls(accepted=False, reasons=collected)

    @classmethod
    def combine(cls, *verdicts: QualityVerdict) -> QualityVerdict:
        return cls.from_reasons(reason for verdict in verdicts for reason in verdict.reasons)

