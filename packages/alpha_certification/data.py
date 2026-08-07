"""Content-addressed historical OHLCV snapshots with explicit gap policy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import httpx


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_PAGE_LIMIT = 1_000
_INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 3 * 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "2h": 2 * 60 * 60,
    "4h": 4 * 60 * 60,
    "6h": 6 * 60 * 60,
    "8h": 8 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("historical timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _utc(parsed)


def _decimal(value: Any) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("OHLCV value is not a valid decimal") from exc
    if not result.is_finite():
        raise ValueError("OHLCV value must be finite")
    return result


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class GapPolicy(StrEnum):
    """How missing intervals are handled when sealing a dataset."""

    REJECT = "reject"
    RECORD = "record"
    ALLOWLIST = "allowlist"


class DataGapError(ValueError):
    """Historical data does not satisfy its declared gap policy."""


@dataclass(frozen=True, slots=True)
class HistoricalRequest:
    symbol: str
    interval: str
    start: datetime
    end: datetime
    gap_policy: GapPolicy = GapPolicy.REJECT
    allowed_gaps: tuple[datetime, ...] = ()

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if re.fullmatch(r"[A-Z0-9]{5,24}", symbol) is None:
            raise ValueError("symbol must be a compact uppercase venue symbol")
        if self.interval not in _INTERVAL_SECONDS:
            raise ValueError(f"unsupported fixed interval: {self.interval}")
        start = _utc(self.start)
        end = _utc(self.end)
        if end <= start:
            raise ValueError("historical end must be after start")
        step = _INTERVAL_SECONDS[self.interval]
        if (
            start.microsecond
            or end.microsecond
            or int(start.timestamp()) % step
            or int(end.timestamp()) % step
        ):
            raise ValueError("historical boundaries must align to the interval")
        allowed = tuple(sorted({_utc(value) for value in self.allowed_gaps}))
        if any(value < start or value >= end for value in allowed):
            raise ValueError("allowed gaps must fall inside the half-open window")
        if any(value.microsecond or int(value.timestamp()) % step for value in allowed):
            raise ValueError("allowed gaps must align to the interval")
        if self.gap_policy is not GapPolicy.ALLOWLIST and allowed:
            raise ValueError("allowed_gaps require gap_policy=allowlist")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "allowed_gaps", allowed)

    @property
    def interval_seconds(self) -> int:
        return _INTERVAL_SECONDS[self.interval]

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "start": _iso(self.start),
            "end": _iso(self.end),
            "window": "half_open",
            "gap_policy": self.gap_policy.value,
            "allowed_gaps": [_iso(value) for value in self.allowed_gaps],
        }


@dataclass(frozen=True, slots=True)
class HistoricalCandle:
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None = None
    trade_count: int | None = None

    def __post_init__(self) -> None:
        open_time = _utc(self.open_time)
        close_time = _utc(self.close_time)
        prices = tuple(
            _decimal(value) for value in (self.open, self.high, self.low, self.close)
        )
        volume = _decimal(self.volume)
        quote_volume = (
            None if self.quote_volume is None else _decimal(self.quote_volume)
        )
        if close_time <= open_time:
            raise ValueError("candle close_time must be after open_time")
        if any(value <= 0 for value in prices):
            raise ValueError("candle prices must be positive")
        if prices[1] < max(prices[0], prices[2], prices[3]):
            raise ValueError("candle high violates OHLC invariants")
        if prices[2] > min(prices[0], prices[1], prices[3]):
            raise ValueError("candle low violates OHLC invariants")
        if volume < 0 or (quote_volume is not None and quote_volume < 0):
            raise ValueError("candle volumes must be non-negative")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        object.__setattr__(self, "open_time", open_time)
        object.__setattr__(self, "close_time", close_time)
        object.__setattr__(self, "open", prices[0])
        object.__setattr__(self, "high", prices[1])
        object.__setattr__(self, "low", prices[2])
        object.__setattr__(self, "close", prices[3])
        object.__setattr__(self, "volume", volume)
        object.__setattr__(self, "quote_volume", quote_volume)

    def to_payload(self) -> dict[str, Any]:
        return {
            "open_time": _iso(self.open_time),
            "close_time": _iso(self.close_time),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
            "quote_volume": (
                None if self.quote_volume is None else str(self.quote_volume)
            ),
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> HistoricalCandle:
        return cls(
            open_time=_parse_utc(str(payload["open_time"])),
            close_time=_parse_utc(str(payload["close_time"])),
            open=_decimal(payload["open"]),
            high=_decimal(payload["high"]),
            low=_decimal(payload["low"]),
            close=_decimal(payload["close"]),
            volume=_decimal(payload["volume"]),
            quote_volume=(
                None
                if payload.get("quote_volume") is None
                else _decimal(payload["quote_volume"])
            ),
            trade_count=(
                None
                if payload.get("trade_count") is None
                else int(payload["trade_count"])
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    manifest_path: Path
    data_path: Path
    request: HistoricalRequest
    candles: tuple[HistoricalCandle, ...]
    gaps: tuple[datetime, ...]
    data_sha256: str
    manifest_sha256: str


PageFetcher = Callable[[Mapping[str, Any]], Awaitable[Sequence[Sequence[Any]]]]


class BinanceHistoricalDownloader:
    """Fetch, validate, and seal one immutable Binance kline snapshot."""

    def __init__(
        self,
        fetch_page: PageFetcher | None = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._fetch_page = fetch_page
        self._timeout = timeout_seconds

    async def download(
        self,
        request: HistoricalRequest,
        output_directory: Path,
    ) -> DatasetSnapshot:
        candles = await self._fetch_all(request)
        gaps = self._find_gaps(request, candles)
        self._enforce_gap_policy(request, gaps)
        data_bytes = b"".join(
            _canonical_json(candle.to_payload()) + b"\n" for candle in candles
        )
        data_digest = _sha256(data_bytes)
        output_directory = output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{request.symbol.lower()}-{request.interval}-"
            f"{request.start:%Y%m%dT%H%M%SZ}-{request.end:%Y%m%dT%H%M%SZ}-"
            f"{data_digest[:16]}"
        )
        data_path = output_directory / f"{stem}.jsonl"
        self._write_immutable(data_path, data_bytes)
        manifest_payload = {
            "schema_version": 1,
            "source": {
                "provider": "binance",
                "endpoint": BINANCE_KLINES_URL,
                "page_limit": BINANCE_PAGE_LIMIT,
            },
            "request": request.to_payload(),
            "data_file": data_path.name,
            "data_sha256": data_digest,
            "row_count": len(candles),
            "first_open_time": _iso(candles[0].open_time) if candles else None,
            "last_open_time": _iso(candles[-1].open_time) if candles else None,
            "gaps": [_iso(value) for value in gaps],
        }
        manifest_bytes = _canonical_json(manifest_payload) + b"\n"
        manifest_path = output_directory / f"{stem}.manifest.json"
        self._write_immutable(manifest_path, manifest_bytes)
        return DatasetSnapshot(
            manifest_path=manifest_path,
            data_path=data_path,
            request=request,
            candles=candles,
            gaps=gaps,
            data_sha256=data_digest,
            manifest_sha256=_sha256(manifest_bytes),
        )

    async def _fetch_all(
        self,
        request: HistoricalRequest,
    ) -> tuple[HistoricalCandle, ...]:
        interval_ms = request.interval_seconds * 1_000
        cursor_ms = int(request.start.timestamp() * 1_000)
        end_ms = int(request.end.timestamp() * 1_000)
        records: dict[int, HistoricalCandle] = {}
        expected_rows = max(1, (end_ms - cursor_ms) // interval_ms)
        max_pages = expected_rows // BINANCE_PAGE_LIMIT + 10

        async def collect(fetch_page: PageFetcher) -> None:
            nonlocal cursor_ms
            for _ in range(max_pages):
                if cursor_ms >= end_ms:
                    return
                page = await fetch_page(
                    {
                        "symbol": request.symbol,
                        "interval": request.interval,
                        "startTime": cursor_ms,
                        "endTime": end_ms - 1,
                        "limit": BINANCE_PAGE_LIMIT,
                    }
                )
                if not page:
                    return
                latest_ms = cursor_ms - interval_ms
                for raw in page:
                    if not raw:
                        raise ValueError("Binance kline row is empty")
                    open_ms = int(raw[0])
                    latest_ms = max(latest_ms, open_ms)
                    if open_ms < int(request.start.timestamp() * 1_000):
                        continue
                    if open_ms >= end_ms:
                        continue
                    candle = self._from_binance_row(raw, request)
                    if candle is None:
                        continue
                    existing = records.get(open_ms)
                    if existing is not None and existing != candle:
                        raise ValueError(
                            f"conflicting duplicate candle at {_iso(candle.open_time)}"
                        )
                    records[open_ms] = candle
                next_cursor = latest_ms + interval_ms
                if next_cursor <= cursor_ms:
                    raise RuntimeError("Binance pagination did not advance")
                cursor_ms = next_cursor
            raise RuntimeError("Binance pagination exceeded its bounded page count")

        if self._fetch_page is not None:
            await collect(self._fetch_page)
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:

                async def fetch(params: Mapping[str, Any]) -> Sequence[Sequence[Any]]:
                    response = await client.get(BINANCE_KLINES_URL, params=dict(params))
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, list):
                        raise ValueError("Binance kline response must be a list")
                    return payload

                await collect(fetch)
        return tuple(records[key] for key in sorted(records))

    @staticmethod
    def _from_binance_row(
        raw: Sequence[Any],
        request: HistoricalRequest,
    ) -> HistoricalCandle | None:
        if len(raw) < 9:
            raise ValueError("Binance kline row has fewer than 9 fields")
        open_ms = int(raw[0])
        interval_ms = request.interval_seconds * 1_000
        if open_ms % interval_ms:
            raise ValueError("Binance kline open time is off the requested grid")
        open_time = datetime.fromtimestamp(open_ms / 1_000, tz=UTC)
        if len(raw) > 6 and int(raw[6]) != open_ms + interval_ms - 1:
            if (
                request.gap_policy is GapPolicy.ALLOWLIST
                and open_time in request.allowed_gaps
            ):
                return None
            raise ValueError(
                "Binance kline close time violates interval semantics at "
                f"{_iso(open_time)}"
            )
        expected_close = open_time + timedelta(seconds=request.interval_seconds)
        return HistoricalCandle(
            open_time=open_time,
            close_time=expected_close,
            open=_decimal(raw[1]),
            high=_decimal(raw[2]),
            low=_decimal(raw[3]),
            close=_decimal(raw[4]),
            volume=_decimal(raw[5]),
            quote_volume=_decimal(raw[7]),
            trade_count=int(raw[8]),
        )

    @staticmethod
    def _find_gaps(
        request: HistoricalRequest,
        candles: tuple[HistoricalCandle, ...],
    ) -> tuple[datetime, ...]:
        actual = {candle.open_time for candle in candles}
        expected: list[datetime] = []
        cursor = request.start
        step = timedelta(seconds=request.interval_seconds)
        while cursor < request.end:
            if cursor not in actual:
                expected.append(cursor)
            cursor += step
        return tuple(expected)

    @staticmethod
    def _enforce_gap_policy(
        request: HistoricalRequest,
        gaps: tuple[datetime, ...],
    ) -> None:
        if request.gap_policy is GapPolicy.REJECT and gaps:
            raise DataGapError(
                f"dataset contains {len(gaps)} gaps; first={_iso(gaps[0])}"
            )
        if request.gap_policy is GapPolicy.ALLOWLIST:
            unexpected = sorted(set(gaps) - set(request.allowed_gaps))
            if unexpected:
                raise DataGapError(
                    f"dataset contains non-allowlisted gap {_iso(unexpected[0])}"
                )
            stale = sorted(set(request.allowed_gaps) - set(gaps))
            if stale:
                raise DataGapError(
                    f"allowlisted gap is present in data {_iso(stale[0])}"
                )

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise FileExistsError(f"immutable snapshot collision: {path}")
            return
        try:
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
        except FileExistsError:
            if path.read_bytes() != payload:
                raise


def load_snapshot(manifest_path: Path) -> DatasetSnapshot:
    """Load a snapshot only after manifest, content, and row validation."""

    if manifest_path.is_symlink():
        raise ValueError("dataset manifest cannot be a symlink")
    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest schema")
    request_payload = manifest["request"]
    request = HistoricalRequest(
        symbol=request_payload["symbol"],
        interval=request_payload["interval"],
        start=_parse_utc(request_payload["start"]),
        end=_parse_utc(request_payload["end"]),
        gap_policy=GapPolicy(request_payload["gap_policy"]),
        allowed_gaps=tuple(
            _parse_utc(value) for value in request_payload.get("allowed_gaps", ())
        ),
    )
    data_name = Path(str(manifest["data_file"]))
    if data_name.name != str(manifest["data_file"]):
        raise ValueError("dataset manifest data_file must be a basename")
    data_path = manifest_path.parent / data_name
    if data_path.is_symlink():
        raise ValueError("dataset data_file cannot be a symlink")
    data_bytes = data_path.read_bytes()
    data_digest = _sha256(data_bytes)
    if data_digest != manifest["data_sha256"]:
        raise ValueError("dataset content hash mismatch")
    candles = tuple(
        HistoricalCandle.from_payload(json.loads(line))
        for line in data_bytes.splitlines()
        if line.strip()
    )
    if len(candles) != int(manifest["row_count"]):
        raise ValueError("dataset row count mismatch")
    if any(
        current.open_time <= previous.open_time
        for previous, current in zip(candles, candles[1:], strict=False)
    ):
        raise ValueError("dataset candles are not strictly ordered")
    step = timedelta(seconds=request.interval_seconds)
    if any(
        candle.open_time < request.start
        or candle.open_time >= request.end
        or candle.open_time.microsecond
        or int(candle.open_time.timestamp()) % request.interval_seconds
        or candle.close_time != candle.open_time + step
        for candle in candles
    ):
        raise ValueError("dataset candle falls outside the exact interval grid")
    gaps = tuple(_parse_utc(value) for value in manifest.get("gaps", ()))
    actual_gaps = BinanceHistoricalDownloader._find_gaps(request, candles)
    if gaps != actual_gaps:
        raise ValueError("dataset gap manifest does not match content")
    BinanceHistoricalDownloader._enforce_gap_policy(request, gaps)
    return DatasetSnapshot(
        manifest_path=manifest_path,
        data_path=data_path,
        request=request,
        candles=candles,
        gaps=gaps,
        data_sha256=data_digest,
        manifest_sha256=_sha256(manifest_bytes),
    )
