from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from packages.data_quality import (
    OrderBookBuilder,
    QualityCode,
    UnsynchronizedOrderBookError,
    validate_candle,
    validate_sequence,
    validate_timestamps,
)


NOW = datetime(2026, 7, 15, 0, 0, 5, tzinfo=UTC)


def valid_candle() -> dict[str, object]:
    return {
        "open_time": datetime(2026, 7, 15, 0, 0, tzinfo=UTC),
        "close_time": datetime(2026, 7, 15, 0, 1, tzinfo=UTC),
        "open": "100.1",
        "high": "101.2",
        "low": "99.8",
        "close": "100.8",
        "volume": "50.25",
    }


def test_valid_candle_returns_an_explicit_ok_code() -> None:
    verdict = validate_candle(valid_candle())

    assert verdict.accepted
    assert verdict.is_valid
    assert verdict.reason_codes == ("OK",)


def test_candle_validation_collects_reason_codes_without_throwing() -> None:
    data = valid_candle()
    data.update(high="99", low="102", open="0", volume="-1", close_time=data["open_time"])

    verdict = validate_candle(data)

    assert not verdict.accepted
    assert {
        QualityCode.NON_POSITIVE_PRICE,
        QualityCode.NEGATIVE_VOLUME,
        QualityCode.HIGH_BELOW_PRICE,
        QualityCode.LOW_ABOVE_PRICE,
        QualityCode.INVALID_CANDLE_WINDOW,
    } <= set(verdict.codes)


@pytest.mark.parametrize("bad", ["not-a-number", True, "NaN", None])
def test_candle_validation_rejects_bad_numeric_input(bad: object) -> None:
    data = valid_candle()
    data["open"] = bad

    verdict = validate_candle(data)

    assert not verdict.accepted
    assert set(verdict.codes) & {
        QualityCode.INVALID_NUMBER,
        QualityCode.NON_FINITE_NUMBER,
        QualityCode.MISSING_FIELD,
    }


def test_timestamp_validation_accepts_fresh_utc_event() -> None:
    verdict = validate_timestamps(
        NOW - timedelta(milliseconds=200),
        NOW - timedelta(milliseconds=100),
        now=NOW,
    )

    assert verdict.accepted


def test_timestamp_validation_codes_staleness_and_transport_lag() -> None:
    verdict = validate_timestamps(
        NOW - timedelta(seconds=10),
        NOW - timedelta(seconds=1),
        now=NOW,
        max_staleness=timedelta(seconds=5),
        max_transport_lag=timedelta(seconds=2),
    )

    assert not verdict.accepted
    assert QualityCode.STALE_EVENT in verdict.codes
    assert QualityCode.EXCESSIVE_TRANSPORT_LAG in verdict.codes


def test_timestamp_validation_rejects_non_utc_future_and_reverse_clocks() -> None:
    non_utc = timezone(timedelta(hours=8))
    offset_verdict = validate_timestamps(
        datetime(2026, 7, 15, 8, 0, 4, tzinfo=non_utc),
        datetime(2026, 7, 15, 8, 0, 4, 100000, tzinfo=non_utc),
        now=NOW,
    )
    assert QualityCode.TIMESTAMP_NOT_UTC in offset_verdict.codes

    future = validate_timestamps(NOW + timedelta(seconds=2), NOW, now=NOW)
    assert QualityCode.EVENT_FROM_FUTURE in future.codes
    assert QualityCode.RECEIVED_BEFORE_EXCHANGE in future.codes


def test_sequence_validation_accepts_contiguous_and_covering_ranges() -> None:
    assert validate_sequence(100, 101, 101).accepted
    assert validate_sequence(100, 99, 105).accepted
    assert validate_sequence(None, 100, 100, snapshot=True).accepted


def test_sequence_validation_distinguishes_gap_stale_and_uninitialized() -> None:
    assert validate_sequence(100, 102, 103).codes == (QualityCode.SEQUENCE_GAP,)
    assert validate_sequence(100, 90, 100).codes == (QualityCode.STALE_SEQUENCE,)
    assert validate_sequence(None, 1, 1).codes == (QualityCode.SEQUENCE_NOT_INITIALIZED,)
    assert validate_sequence(100, 105, 104).codes == (QualityCode.INVALID_SEQUENCE,)


def test_order_book_requires_snapshot_before_delta() -> None:
    book = OrderBookBuilder()

    verdict = book.apply_delta(sequence_start=1, sequence_end=1, bids=[["100", "1"]])

    assert verdict.codes == (QualityCode.BOOK_NOT_SYNCHRONIZED,)
    assert not book.is_synchronized
    with pytest.raises(UnsynchronizedOrderBookError):
        book.view()


def test_order_book_applies_contiguous_decimal_safe_updates_and_deletes() -> None:
    book = OrderBookBuilder()
    snapshot = book.apply_snapshot(
        sequence=100,
        bids=[["100.00", "2"], ["99.5", "3"]],
        asks=[["100.5", "4"], ["101", "5"]],
    )
    delta = book.apply_delta(
        sequence_start=101,
        sequence_end=102,
        bids=[["100", "0"], ["100.25", "1.5"]],
        asks=[{"price": "100.5", "qty": "2"}],
    )

    assert snapshot.accepted and delta.accepted
    assert book.is_synchronized
    assert book.last_sequence == 102
    assert book.best_bid == (Decimal("100.25"), Decimal("1.5"))
    assert book.best_ask == (Decimal("100.5"), Decimal("2"))
    assert book.view(depth=1).bids == ((Decimal("100.25"), Decimal("1.5")),)


def test_order_book_gap_fails_closed_and_requires_resnapshot() -> None:
    book = OrderBookBuilder()
    assert book.apply_snapshot(sequence=10, bids=[["100", "1"]], asks=[["101", "1"]]).accepted

    gap = book.apply_delta(sequence_start=12, sequence_end=12, bids=[["100", "2"]])

    assert gap.codes == (QualityCode.SEQUENCE_GAP,)
    assert not book.is_synchronized
    assert book.last_sequence is None
    assert book.best_bid is None
    assert book.apply_delta(sequence_start=13, sequence_end=13).codes == (
        QualityCode.BOOK_NOT_SYNCHRONIZED,
    )


def test_stale_duplicate_is_ignored_without_corrupting_a_synchronized_book() -> None:
    book = OrderBookBuilder()
    book.apply_snapshot(sequence=10, bids=[["100", "1"]], asks=[["101", "1"]])

    stale = book.apply_delta(sequence_start=9, sequence_end=10, bids=[["100", "999"]])

    assert stale.codes == (QualityCode.STALE_SEQUENCE,)
    assert book.is_synchronized
    assert book.best_bid == (Decimal("100"), Decimal("1"))


@pytest.mark.parametrize(
    ("bids", "asks", "code"),
    [
        ([["101", "1"]], [["101", "1"]], QualityCode.CROSSED_ORDER_BOOK),
        ([["0", "1"]], [["101", "1"]], QualityCode.INVALID_BOOK_LEVEL),
        ([], [["101", "1"]], QualityCode.INCOMPLETE_ORDER_BOOK),
    ],
)
def test_invalid_snapshot_never_becomes_readable(
    bids: list[list[str]], asks: list[list[str]], code: QualityCode
) -> None:
    book = OrderBookBuilder()

    verdict = book.apply_snapshot(sequence=1, bids=bids, asks=asks)

    assert code in verdict.codes
    assert not book.is_synchronized


def test_crossing_delta_invalidates_existing_book_atomically() -> None:
    book = OrderBookBuilder()
    book.apply_snapshot(sequence=5, bids=[["100", "1"]], asks=[["101", "1"]])

    verdict = book.apply_delta(sequence_start=6, sequence_end=6, bids=[["102", "1"]])

    assert verdict.codes == (QualityCode.CROSSED_ORDER_BOOK,)
    assert not book.is_synchronized
    with pytest.raises(UnsynchronizedOrderBookError):
        book.view()


def test_valid_resnapshot_recovers_after_gap() -> None:
    book = OrderBookBuilder()
    book.apply_snapshot(sequence=5, bids=[["100", "1"]], asks=[["101", "1"]])
    book.apply_delta(sequence_start=7, sequence_end=7)

    recovered = book.apply_snapshot(sequence=20, bids=[["200", "2"]], asks=[["201", "3"]])

    assert recovered.accepted
    assert book.is_synchronized
    assert book.last_sequence == 20

