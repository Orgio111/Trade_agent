"""Immutable historical snapshots and purged walk-forward boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest

from packages.alpha_certification.data import (
    BinanceHistoricalDownloader,
    DataGapError,
    GapPolicy,
    HistoricalRequest,
    load_snapshot,
)
from packages.alpha_certification.walk_forward import (
    PurgedWalkForwardSplitter,
    WalkForwardConfig,
)


def _row(at: datetime, price: int) -> list[object]:
    open_ms = int(at.timestamp() * 1_000)
    return [
        open_ms,
        str(price),
        str(price + 2),
        str(price - 2),
        str(price + 1),
        "10",
        open_ms + 3_599_999,
        str((price + 1) * 10),
        7,
        "0",
        "0",
        "0",
    ]


@pytest.mark.asyncio
async def test_downloader_paginates_short_pages_and_seals_half_open_snapshot(
    tmp_path,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows = [_row(start + timedelta(hours=index), 100 + index) for index in range(4)]
    calls: list[dict[str, object]] = []

    async def fetch(params):
        calls.append(dict(params))
        available = [
            row
            for row in rows
            if int(row[0]) >= int(params["startTime"])
            and int(row[0]) <= int(params["endTime"])
        ]
        return available[:2]

    request = HistoricalRequest(
        symbol="BTCUSDT",
        interval="1h",
        start=start,
        end=start + timedelta(hours=4),
    )
    snapshot = await BinanceHistoricalDownloader(fetch).download(request, tmp_path)

    assert len(calls) == 2
    assert all(call["limit"] == 1_000 for call in calls)
    assert len(snapshot.candles) == 4
    assert snapshot.candles[-1].open_time == start + timedelta(hours=3)
    assert snapshot.data_path.name.endswith(f"{snapshot.data_sha256[:16]}.jsonl")
    assert load_snapshot(snapshot.manifest_path).data_sha256 == snapshot.data_sha256

    repeated = await BinanceHistoricalDownloader(fetch).download(request, tmp_path)
    assert repeated.manifest_path == snapshot.manifest_path


@pytest.mark.asyncio
async def test_gap_policy_rejects_records_or_allowlists_explicitly(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    missing = start + timedelta(hours=1)
    rows = [_row(start, 100), _row(start + timedelta(hours=2), 102)]

    async def fetch(params):
        return [row for row in rows if int(row[0]) >= int(params["startTime"])]

    with pytest.raises(DataGapError, match="contains 1 gaps"):
        await BinanceHistoricalDownloader(fetch).download(
            HistoricalRequest(
                symbol="BTCUSDT",
                interval="1h",
                start=start,
                end=start + timedelta(hours=3),
            ),
            tmp_path / "reject",
        )

    recorded = await BinanceHistoricalDownloader(fetch).download(
        HistoricalRequest(
            symbol="BTCUSDT",
            interval="1h",
            start=start,
            end=start + timedelta(hours=3),
            gap_policy=GapPolicy.RECORD,
        ),
        tmp_path / "record",
    )
    assert recorded.gaps == (missing,)

    allowlisted = await BinanceHistoricalDownloader(fetch).download(
        HistoricalRequest(
            symbol="BTCUSDT",
            interval="1h",
            start=start,
            end=start + timedelta(hours=3),
            gap_policy=GapPolicy.ALLOWLIST,
            allowed_gaps=(missing,),
        ),
        tmp_path / "allowlist",
    )
    assert load_snapshot(allowlisted.manifest_path).gaps == (missing,)

    with pytest.raises(DataGapError, match="allowlisted gap is present"):
        await BinanceHistoricalDownloader(fetch).download(
            HistoricalRequest(
                symbol="BTCUSDT",
                interval="1h",
                start=start,
                end=start + timedelta(hours=3),
                gap_policy=GapPolicy.ALLOWLIST,
                allowed_gaps=(missing, start + timedelta(hours=2)),
            ),
            tmp_path / "stale-allowlist",
        )


@pytest.mark.asyncio
async def test_downloader_rejects_off_grid_exchange_rows(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    off_grid = _row(start + timedelta(minutes=1), 100)

    async def fetch(params):
        return [off_grid]

    with pytest.raises(ValueError, match="off the requested grid"):
        await BinanceHistoricalDownloader(fetch).download(
            HistoricalRequest(
                symbol="BTCUSDT",
                interval="1h",
                start=start,
                end=start + timedelta(hours=1),
            ),
            tmp_path,
        )


@pytest.mark.asyncio
async def test_truncated_exchange_candle_requires_exact_gap_allowlist(
    tmp_path,
) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    truncated = _row(start, 100)
    truncated[6] = int(truncated[6]) - 60_000
    rows = [truncated, _row(start + timedelta(hours=1), 101)]

    async def fetch(params):
        return [row for row in rows if int(row[0]) >= int(params["startTime"])]

    with pytest.raises(ValueError, match="close time violates"):
        await BinanceHistoricalDownloader(fetch).download(
            HistoricalRequest(
                symbol="BTCUSDT",
                interval="1h",
                start=start,
                end=start + timedelta(hours=2),
            ),
            tmp_path / "reject",
        )

    snapshot = await BinanceHistoricalDownloader(fetch).download(
        HistoricalRequest(
            symbol="BTCUSDT",
            interval="1h",
            start=start,
            end=start + timedelta(hours=2),
            gap_policy=GapPolicy.ALLOWLIST,
            allowed_gaps=(start,),
        ),
        tmp_path / "allow",
    )

    assert snapshot.gaps == (start,)
    assert tuple(candle.open_time for candle in snapshot.candles) == (
        start + timedelta(hours=1),
    )
    assert load_snapshot(snapshot.manifest_path).gaps == (start,)


@pytest.mark.asyncio
async def test_snapshot_loader_detects_content_tampering(tmp_path) -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)

    async def fetch(params):
        return [_row(start, 100)]

    snapshot = await BinanceHistoricalDownloader(fetch).download(
        HistoricalRequest(
            symbol="BTCUSDT",
            interval="1h",
            start=start,
            end=start + timedelta(hours=1),
        ),
        tmp_path,
    )
    payload = json.loads(snapshot.data_path.read_text())
    payload["close"] = "999"
    snapshot.data_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_snapshot(snapshot.manifest_path)


def test_purged_walk_forward_is_strictly_chronological_and_embargoed() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = tuple(start + timedelta(hours=index) for index in range(1_000))
    config = WalkForwardConfig(
        train_window=timedelta(hours=400),
        test_window=timedelta(hours=100),
        purge=timedelta(hours=12),
        embargo=timedelta(hours=24),
        min_train_samples=300,
        min_test_samples=90,
    )
    folds = PurgedWalkForwardSplitter(config).split(timestamps)

    assert len(folds) >= 3
    for fold in folds:
        train_times = [timestamps[index] for index in fold.train_indices]
        test_times = [timestamps[index] for index in fold.test_indices]
        assert max(train_times) < fold.train_end
        assert fold.train_end == fold.test_start - config.purge
        assert fold.train_start == fold.train_end - config.train_window
        assert max(train_times) < min(test_times)
        assert set(fold.train_indices).isdisjoint(fold.test_indices)
    for previous, current in zip(folds, folds[1:], strict=False):
        assert current.test_start >= previous.test_end + config.embargo
