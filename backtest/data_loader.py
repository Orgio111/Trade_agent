"""Historical OHLCV data loader for backtesting.

Supports fetching from an exchange via ccxt or loading from a local cache.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import ccxt.async_support as ccxt  # type: ignore[import]

from backtest.models import BacktestConfig, BarRecord

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(".backtest_cache")


def _cache_path(exchange_id: str, symbol: str, timeframe: str) -> Path:
    return _CACHE_DIR / f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}.pkl"


async def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    since: str | None = None,
    limit: int | None = None,
    use_cache: bool = True,
) -> list[BarRecord]:
    """Fetch historical OHLCV bars from an exchange (with optional disk cache).

    Parameters
    ----------
    exchange_id : str
        ccxt exchange ID (e.g. ``"binance"``).
    symbol : str
        Trading pair (e.g. ``"BTC/USDT"``).
    timeframe : str
        Candle interval (``"1h"``, ``"1m"``, etc.).
    since : str, optional
        ISO-formatted start date (e.g. ``"2024-01-01"``).
    limit : int, optional
        Max bars.  If unset, fetches everything from *since* to now.
    use_cache : bool
        If ``True``, saves/loads from a local pickle cache to avoid
        re-fetching on repeated runs.

    Returns
    -------
    list[BarRecord]
        Sorted oldest-first.
    """
    if use_cache:
        cache = _cache_path(exchange_id, symbol, timeframe)
        if cache.exists():
            logger.info("Loading cached OHLCV: %s", cache)
            with cache.open("rb") as f:
                return pickle.load(f)

    ExchangeClass = getattr(ccxt, exchange_id, None)
    if ExchangeClass is None:
        raise ValueError(f"Unknown exchange: {exchange_id}")

    exchange = ExchangeClass({"enableRateLimit": True})
    since_ms: int | None = None
    if since:
        try:
            since_dt = dt.datetime.fromisoformat(since)
            since_ms = int(since_dt.timestamp() * 1000)
        except ValueError:
            pass

    try:
        all_bars: list[BarRecord] = []
        while True:
            raw = await exchange.fetch_ohlcv(
                symbol, timeframe, since=since_ms, limit=limit or 1000
            )
            if not raw:
                break
            for row in raw:
                ts, o, h, l, c, v = row
                bar = BarRecord(
                    timestamp=dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc),
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                    volume=v,
                )
                all_bars.append(bar)
            if limit and len(raw) < (limit or 1000):
                break
            since_ms = raw[-1][0] + 1  # next page
            if len(raw) < 1000:
                break

        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as f:
                pickle.dump(all_bars, f)
            logger.info("Cached %d bars to %s", len(all_bars), cache)

        return all_bars
    finally:
        await exchange.close()


async def load_data(
    cfg: BacktestConfig,
) -> dict[str, list[BarRecord]]:
    """Load OHLCV data for all configured symbols.

    Returns a dict mapping symbol → sorted list of BarRecord.
    All symbol lists are guaranteed to share the same timestamp grid
    (missing points are forward-filled).
    """
    all_data: dict[str, list[BarRecord]] = {}
    for symbol in cfg.symbols:
        bars = await fetch_ohlcv(
            exchange_id=cfg.exchange_id,
            symbol=symbol,
            timeframe=cfg.timeframe,
            since=cfg.start,
            use_cache=True,
        )
        # Filter by end date
        if cfg.end:
            end_dt = dt.datetime.fromisoformat(cfg.end).replace(
                tzinfo=dt.timezone.utc
            )
            bars = [b for b in bars if b.timestamp <= end_dt]
        all_data[symbol] = bars
        logger.info(
            "Loaded %d bars for %s [%s .. %s]",
            len(bars),
            symbol,
            bars[0].timestamp.isoformat() if bars else "N/A",
            bars[-1].timestamp.isoformat() if bars else "N/A",
        )

    # ── Align all symbols to a common timestamp grid (intersection) ──────────
    if not all_data:
        return all_data

    # Only one symbol — no alignment needed
    if len(all_data) == 1:
        return all_data

    # Find timestamps common to all symbols
    ts_sets = [set(b.timestamp for b in bars) for bars in all_data.values()]
    common_ts = sorted(set.intersection(*ts_sets))

    aligned: dict[str, list[BarRecord]] = {}
    for symbol, bars in all_data.items():
        ts_map = {b.timestamp: b for b in bars}
        aligned_bars = [ts_map[ts] for ts in common_ts]
        aligned[symbol] = aligned_bars

    logger.info("Aligned %d symbols — %d common bars", len(aligned), len(common_ts))
    return aligned
