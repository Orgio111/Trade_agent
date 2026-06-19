"""
Normalize trade identifiers between CCXT-style `BTC/USDT` and
Binance API-style `BTCUSDT`/interval codes.

Rules:
- Canonical internal format: `BASE/QUOTE` with slash.
- Binance REST/WS format: no slash, uppercase, e.g. `BTCUSDT`.
- Timeframe canonical: CCXT style `1m/5m/15m/1h/4h/1d`.
- Binance interval: `1m/5m/15m/1h/4h/1d` compatible with most endpoints.
"""
from __future__ import annotations

from typing import Iterable, Literal

BinanceInterval = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
CcxtInterval = Literal["1m", "5m", "15m", "1h", "4h", "1d"]


def to_binance_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return f"{base}{quote}"
    return symbol


def to_ccxt_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if "/" in symbol:
        return symbol
    if len(symbol) <= 6:
        return symbol
    if len(symbol) == 8 and symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT"
    return symbol


def to_binance_interval(interval: str) -> str:
    return interval.strip().lower()


def to_ccxt_interval(interval: str) -> str:
    return interval.strip().lower()


def normalize_symbols(symbols: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in symbols:
        ccxt = to_ccxt_symbol(s)
        if ccxt not in seen:
            seen.add(ccxt)
            out.append(ccxt)
    return out


def normalize_timeframes(tfs: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tfs:
        it = to_ccxt_interval(t)
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
