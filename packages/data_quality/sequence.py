"""Generic inclusive sequence-range validation."""

from __future__ import annotations

from .verdicts import QualityCode, QualityReason, QualityVerdict


def _valid_sequence(value: int | None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_sequence(
    previous_end: int | None,
    sequence_start: int | None,
    sequence_end: int | None,
    *,
    snapshot: bool = False,
) -> QualityVerdict:
    """Validate an inclusive range covers exactly the next required sequence.

    A range may begin before ``previous_end + 1`` because several exchanges
    intentionally overlap diff ranges.  It is accepted only when it still
    covers the next required sequence.  Entirely old ranges are rejected as
    stale, while ranges beginning after the expected sequence are gaps.
    """

    if not _valid_sequence(sequence_start) or not _valid_sequence(sequence_end):
        return QualityVerdict.reject(
            QualityReason(QualityCode.INVALID_SEQUENCE, "sequence range must contain non-negative integers")
        )
    assert sequence_start is not None and sequence_end is not None
    if sequence_end < sequence_start:
        return QualityVerdict.reject(
            QualityReason(QualityCode.INVALID_SEQUENCE, "sequence_end is before sequence_start")
        )
    if snapshot:
        return QualityVerdict.pass_()
    if not _valid_sequence(previous_end):
        return QualityVerdict.reject(
            QualityReason(QualityCode.SEQUENCE_NOT_INITIALIZED, "a snapshot is required before deltas")
        )
    assert previous_end is not None
    expected = previous_end + 1
    if sequence_end < expected:
        return QualityVerdict.reject(
            QualityReason(QualityCode.STALE_SEQUENCE, f"delta ends at {sequence_end}; expected {expected}")
        )
    if sequence_start > expected:
        return QualityVerdict.reject(
            QualityReason(QualityCode.SEQUENCE_GAP, f"delta starts at {sequence_start}; expected {expected}")
        )
    return QualityVerdict.pass_()

