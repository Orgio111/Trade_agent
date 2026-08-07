"""Strict chronological purged and embargoed walk-forward folds."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Sequence


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("walk-forward timestamps must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    train_window: timedelta
    test_window: timedelta
    purge: timedelta
    embargo: timedelta
    min_train_samples: int = 500
    min_test_samples: int = 100
    expanding: bool = False
    include_partial_test: bool = False

    def __post_init__(self) -> None:
        if self.train_window <= timedelta(0):
            raise ValueError("train_window must be positive")
        if self.test_window <= timedelta(0):
            raise ValueError("test_window must be positive")
        if self.purge < timedelta(0) or self.embargo < timedelta(0):
            raise ValueError("purge and embargo must be non-negative")
        if self.min_train_samples < 1 or self.min_test_samples < 1:
            raise ValueError("minimum fold sample counts must be positive")

    def to_payload(self) -> dict[str, object]:
        return {
            "train_window_seconds": int(self.train_window.total_seconds()),
            "test_window_seconds": int(self.test_window.total_seconds()),
            "purge_seconds": int(self.purge.total_seconds()),
            "embargo_seconds": int(self.embargo.total_seconds()),
            "min_train_samples": self.min_train_samples,
            "min_test_samples": self.min_test_samples,
            "expanding": self.expanding,
            "include_partial_test": self.include_partial_test,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_id: str
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_end: datetime
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "embargo_end": self.embargo_end.isoformat(),
            "train_samples": len(self.train_indices),
            "test_samples": len(self.test_indices),
        }


class PurgedWalkForwardSplitter:
    """Build forward-only folds with no label overlap at the test boundary.

    ``purge`` removes the tail of the training set before each test. ``embargo``
    creates unused time between adjacent test windows. Training is always
    strictly earlier than testing; no future sample can enter a fold.
    """

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def split(
        self,
        timestamps: Sequence[datetime],
        *,
        dataset_end: datetime | None = None,
        fold_prefix: str = "fold",
    ) -> tuple[WalkForwardFold, ...]:
        normalized = tuple(_utc(value) for value in timestamps)
        if len(normalized) < 2:
            raise ValueError("walk-forward requires at least two timestamps")
        if any(
            current <= previous
            for previous, current in zip(normalized, normalized[1:], strict=False)
        ):
            raise ValueError("walk-forward timestamps must be strictly ordered")
        steps = [
            current - previous
            for previous, current in zip(normalized, normalized[1:], strict=False)
        ]
        inferred_step = sorted(steps)[len(steps) // 2]
        if inferred_step <= timedelta(0):
            raise ValueError("could not infer a positive dataset interval")
        end = (
            _utc(dataset_end)
            if dataset_end is not None
            else normalized[-1] + inferred_step
        )
        if end <= normalized[-1]:
            raise ValueError("dataset_end must be after the last timestamp")

        start = normalized[0]
        test_start = start + self.config.train_window + self.config.purge
        folds: list[WalkForwardFold] = []
        fold_number = 1
        while test_start < end:
            requested_test_end = test_start + self.config.test_window
            if requested_test_end > end and not self.config.include_partial_test:
                break
            test_end = min(requested_test_end, end)
            train_end = test_start - self.config.purge
            train_start = (
                start if self.config.expanding else train_end - self.config.train_window
            )
            train_indices = tuple(
                range(
                    bisect_left(normalized, train_start),
                    bisect_left(normalized, train_end),
                )
            )
            test_indices = tuple(
                range(
                    bisect_left(normalized, test_start),
                    bisect_left(normalized, test_end),
                )
            )
            if (
                len(train_indices) >= self.config.min_train_samples
                and len(test_indices) >= self.config.min_test_samples
            ):
                folds.append(
                    WalkForwardFold(
                        fold_id=f"{fold_prefix}-{fold_number:03d}",
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        embargo_end=test_end + self.config.embargo,
                        train_indices=train_indices,
                        test_indices=test_indices,
                    )
                )
                fold_number += 1
            test_start = requested_test_end + self.config.embargo
        if not folds:
            raise ValueError("walk-forward configuration produced no valid folds")
        return tuple(folds)
