"""Feature Extraction Engine: order-flow imbalance, CVD, funding rate, OI delta.

Architecture
────────────
For each configured symbol the engine spawns three background tasks:

  1. Trade collector  – ``watch_trades()`` feed that accumulates buy/sell volume
                       per bar window so we can compute OFI and CVD.
  2. Funding poller   – periodic ``fetch_funding_rate()`` (REST fallback) to
                       track funding rate deltas.
  3. OI poller        – periodic ``fetch_open_interest()`` to track OI deltas
                       and OI–price correlation.

When ``compute(symbol)`` is called the engine snapshots the current rolling
state and returns a ``FeatureSignal``.  The extractor is resilient: if an
exchange stream fails it rolls over to the next configured exchange or
returns ``None`` for the affected fields.

Graceful degradation
────────────────────
- Trades unavailable  → OFI / CVD / trade_strength = None
- Funding unavailable → funding_rate / funding_rate_delta = None
- OI unavailable      → open_interest / OI delta / corr = None
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

import ccxt.pro as ccxtpro  # type: ignore[import]
import numpy as np

from core.config import get_settings
from core.models import FeatureSignal, Side

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
_MAX_WS_FAILURES = 3
_WS_TIMEOUT = 30           # seconds before watch_trades is considered timed out
# _BAR_SECONDS is now set from config in FeatureExtractor.__init__


# ═════════════════════════════════════════════════════════════════════════════
#  Per-symbol rolling state
# ═════════════════════════════════════════════════════════════════════════════

class _TradeBar:
    """Aggregated trade data for one bar window (default 60 seconds)."""

    __slots__ = ("bar_ts", "buy_volume", "sell_volume", "n_trades")

    def __init__(self, bar_ts: float) -> None:
        self.bar_ts = bar_ts
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.n_trades = 0

    def record(self, side: str, amount: float) -> None:
        if side == "buy":
            self.buy_volume += amount
        else:
            self.sell_volume += amount
        self.n_trades += 1

    @property
    def delta(self) -> float:
        """Net aggressive volume (positive = buying pressure)."""
        return self.buy_volume - self.sell_volume

    @property
    def total(self) -> float:
        return self.buy_volume + self.sell_volume

    def ofi(self) -> float | None:
        """Order Flow Imbalance in [-1, 1]."""
        if self.total < 1e-10:
            return None
        return (self.buy_volume - self.sell_volume) / self.total


class _FeatureState:
    """All rolling state for a single symbol.

    Thread safety
    -------------
    Mutations happen from background tasks (trade / funding / OI streams)
    while ``compute_*()`` reads are performed synchronously from the supervisor
    cycle.  Under CPython's GIL, individual deque & float operations are
    atomic, but multi-step operations (``_rotate_bar``, OI-price enrichment)
    use a per-state lock to guarantee internal consistency.
    """

    def __init__(self, symbol: str, cvd_lookback: int, bar_seconds: int = 60) -> None:
        self.symbol = symbol
        self._bar_seconds = bar_seconds
        self._lock = threading.Lock()
        self.bar_queue: deque[_TradeBar] = deque(maxlen=100)
        self._current_bar: _TradeBar | None = None
        self._last_bar_ts = 0.0

        # CVD rolling window
        self._cvd_deltas: deque[float] = deque(maxlen=cvd_lookback)
        self._cvd_cumulative = 0.0

        # OI history for delta + correlation  (price, oi)
        self._oi_values: deque[tuple[float, float]] = deque(maxlen=cvd_lookback)

        # Funding history (annualised %)
        self._funding_values: deque[float] = deque(maxlen=cvd_lookback)

        # Trade intensity baseline (EMA of bars/min)
        self._avg_trades_per_bar: float = 50.0

        # Exchange being used for trade stream
        self.trade_exchange_idx: int = 0
        self.trade_ws_failures: int = 0

    # ── Trade recording ─────────────────────────────────────────────────────

    def record_trade(self, side: str, amount: float, timestamp: int) -> None:
        """Record a single trade from watch_trades()."""
        bar_ts = (timestamp // (self._bar_seconds * 1000)) * self._bar_seconds
        with self._lock:
            if bar_ts != self._last_bar_ts:
                self._rotate_bar_locked(bar_ts)
            if self._current_bar is not None:
                self._current_bar.record(side, amount)

    def _rotate_bar_locked(self, bar_ts: float) -> None:
        """Finalise the previous bar and start a new one (lock held)."""
        if self._current_bar is not None:
            self.bar_queue.append(self._current_bar)
            delta = self._current_bar.delta
            self._cvd_deltas.append(delta)
            self._cvd_cumulative += delta
            self._avg_trades_per_bar = (
                0.9 * self._avg_trades_per_bar + 0.1 * self._current_bar.n_trades
            )
        self._current_bar = _TradeBar(bar_ts)
        self._last_bar_ts = bar_ts

    # ── Feature computation ─────────────────────────────────────────────────

    def compute_ofi(self) -> tuple[float | None, float | None, float | None]:
        """Return (ofi, cvd, cvd_delta) for the latest bar."""
        if not self.bar_queue:
            return None, None, None
        latest = self.bar_queue[-1]
        return latest.ofi(), self._cvd_cumulative, (self._cvd_deltas[-1] if self._cvd_deltas else None)

    def record_oi(self, oi: float, price: float) -> None:
        with self._lock:
            self._oi_values.append((price, oi))

    def enrich_oi_price(self, price: float) -> None:
        """Replace the last OI entry's price (set by the supervisor cycle)."""
        with self._lock:
            if self._oi_values:
                last_oi = self._oi_values[-1]
                self._oi_values.pop()
                self._oi_values.append((price, last_oi[1]))

    def compute_oi(self) -> tuple[float | None, float | None, float | None]:
        """Return (open_interest, oi_delta, oi_price_corr).

        Acquires the per-state lock because it reads ``_oi_values`` which may
        be concurrently mutated by the background OI poller.
        """
        with self._lock:
            if len(self._oi_values) < 2:
                return None, None, None

            latest = self._oi_values[-1][1]
            prev = self._oi_values[0][1]
            delta = latest - prev

            corr: float | None = None
            if len(self._oi_values) >= 5:
                prices = np.array([p for p, _ in self._oi_values])
                ois = np.array([o for _, o in self._oi_values])
                dp = np.diff(prices)
                do = np.diff(ois)
                if len(dp) >= 3:
                    c = np.corrcoef(dp, do)[0, 1]
                    corr = float(c) if not np.isnan(c) else None

            return latest, delta, corr

    def record_funding(self, rate: float) -> None:
        self._funding_values.append(rate)

    def compute_funding(self) -> tuple[float | None, float | None]:
        """Return (funding_rate, funding_rate_delta).

        Acquires the per-state lock because it reads ``_funding_values`` which
        may be concurrently mutated by the background funding poller.
        """
        with self._lock:
            if not self._funding_values:
                return None, None
            latest = self._funding_values[-1]
            delta = latest - self._funding_values[-2] if len(self._funding_values) >= 2 else None
            return latest, delta

    def compute_trade_strength(self) -> float | None:
        """Normalised trade intensity vs rolling average in [0, 2]."""
        if not self.bar_queue or self._avg_trades_per_bar < 1:
            return None
        return min(self.bar_queue[-1].n_trades / self._avg_trades_per_bar, 2.0)


# ═════════════════════════════════════════════════════════════════════════════
#  Perp-symbol helper
# ═════════════════════════════════════════════════════════════════════════════

def _perp_symbol(symbol: str) -> str:
    """Convert a spot symbol (``BTC/USDT``) to the perp trading pair format
    used by most exchanges (``BTC/USDT:USDT``).

    This is a best-effort heuristic.  If the symbol already looks like a
    perp/linear swap (contains ``:``) it is returned as-is.
    """
    if ":" in symbol:
        return symbol
    # BTC/USDT → BTC/USDT:USDT
    base, quote = symbol.split("/", 1)
    return f"{base}/{quote}:{quote}"


# ═════════════════════════════════════════════════════════════════════════════
#  FeatureExtractor — manages background streams per symbol
# ═════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Long-lived feature extraction engine.

    Call ``start()`` to begin background trade / funding / OI streams for each
    configured symbol.  Call ``compute(symbol)`` from the supervisor cycle to
    get a ``FeatureSignal`` snapshot.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.symbols = cfg.symbols
        self.exchange_ids: list[str] = cfg.exchanges
        self._bar_seconds: int = cfg.feature_trade_window_seconds
        self._running = False
        self._states: dict[str, _FeatureState] = {
            s: _FeatureState(s, cfg.feature_cvd_lookback, self._bar_seconds)
            for s in self.symbols
        }
        self._tasks: list[asyncio.Task] = []

        # Exchange connections – separate instances for trades (spot) vs
        # funding/OI (swap), since the two categories use different market types.
        self._trade_exchanges: dict[str, ccxtpro.Exchange] = {}
        self._perp_exchanges: dict[str, ccxtpro.Exchange] = {}

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch background streams for all symbols."""
        if self._running:
            return
        self._running = True

        for symbol in self.symbols:
            self._tasks.append(
                asyncio.create_task(self._stream_trades(symbol), name=f"feat-trades-{symbol}")
            )
            self._tasks.append(
                asyncio.create_task(self._poll_funding(symbol), name=f"feat-funding-{symbol}")
            )
            self._tasks.append(
                asyncio.create_task(self._poll_oi(symbol), name=f"feat-oi-{symbol}")
            )

        logger.info(
            "FeatureExtractor started — %d symbols × 3 streams each",
            len(self.symbols),
        )

    async def stop(self) -> None:
        """Cancel all background tasks and close exchange connections."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for pool in (self._trade_exchanges, self._perp_exchanges):
            for exchange in pool.values():
                try:
                    await exchange.close()
                except Exception:
                    pass

        logger.info("FeatureExtractor stopped")

    # ── Public feature computation ──────────────────────────────────────────

    def compute(self, symbol: str, current_price: float | None = None) -> FeatureSignal:
        """Snapshot current features for the given symbol.

        Parameters
        ----------
        symbol : str
            Trading pair symbol (e.g. ``"BTC/USDT"``).
        current_price : float, optional
            Latest market price from the OHLCV feed.  Used for OI–price
            correlation.  If omitted the correlation will be ``None``.

        This is a synchronous call — it reads rolling state without awaiting
        any network I/O.  Call it from the supervisor cycle.
        """
        now = datetime.now(timezone.utc)
        state = self._states.get(symbol)
        if state is None:
            return FeatureSignal(symbol=symbol, timestamp=now)

        # ── Enrich OI with current market price for correlation ──────────────
        if current_price is not None:
            state.enrich_oi_price(current_price)

        ofi, cvd, cvd_delta = state.compute_ofi()
        oi, oi_delta, oi_corr = state.compute_oi()
        funding_rate, funding_delta = state.compute_funding()
        trade_strength = state.compute_trade_strength()

        # ── Composite feature trend ──────────────────────────────────────────
        score = 0.0
        if ofi is not None:
            score += ofi
        if cvd_delta is not None:
            score += math.tanh(cvd_delta / (abs(cvd_delta) + 1e-6)) * 0.5
        if funding_rate is not None and abs(funding_rate) > 0.0001:
            # Negative funding → perp pays shorts → bullish; positive → bearish
            score -= math.tanh(funding_rate * 100) * 0.3
        if oi_corr is not None:
            score += oi_corr * 0.3

        norm_score = max(-1.0, min(1.0, score))
        trend = Side.BUY if norm_score > 0.15 else (Side.SELL if norm_score < -0.15 else Side.HOLD)
        confidence = min(abs(norm_score), 1.0)

        return FeatureSignal(
            symbol=symbol,
            timestamp=now,
            ofi=ofi,
            cvd=cvd,
            cvd_delta=cvd_delta,
            funding_rate=funding_rate,
            funding_rate_delta=funding_delta,
            open_interest=oi,
            open_interest_delta=oi_delta,
            oi_price_delta_corr=oi_corr,
            trade_strength=trade_strength,
            trend=trend,
            confidence=confidence,
        )

    # ── Exchange helpers ────────────────────────────────────────────────────

    async def _ensure_trade_exchange(self, exchange_id: str) -> ccxtpro.Exchange:
        """Lazy-init a spot-market exchange for trade streaming."""
        if exchange_id not in self._trade_exchanges:
            ExchangeClass = getattr(ccxtpro, exchange_id, None)
            if ExchangeClass is None:
                raise ValueError(f"Unknown exchange: {exchange_id}")
            ex = ExchangeClass({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            try:
                async with asyncio.timeout(15):
                    await ex.load_markets()
            except Exception:
                await ex.close()
                raise
            self._trade_exchanges[exchange_id] = ex
        return self._trade_exchanges[exchange_id]

    async def _ensure_perp_exchange(self, exchange_id: str) -> ccxtpro.Exchange:
        """Lazy-init a swap-market exchange for funding / OI polling."""
        if exchange_id not in self._perp_exchanges:
            ExchangeClass = getattr(ccxtpro, exchange_id, None)
            if ExchangeClass is None:
                raise ValueError(f"Unknown exchange: {exchange_id}")
            ex = ExchangeClass({
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
            try:
                async with asyncio.timeout(15):
                    await ex.load_markets()
            except Exception:
                await ex.close()
                raise
            self._perp_exchanges[exchange_id] = ex
        return self._perp_exchanges[exchange_id]

    # ── Trade stream ────────────────────────────────────────────────────────

    async def _stream_trades(self, symbol: str) -> None:
        """Continuously stream watch_trades() for a symbol, failover across
        exchanges, and record each trade into the rolling state."""
        cfg = get_settings()
        state = self._states[symbol]

        while self._running:
            idx = state.trade_exchange_idx
            if idx >= len(self.exchange_ids):
                logger.warning("All trade exchanges exhausted for %s — idle", symbol)
                await asyncio.sleep(60)
                continue

            exchange_id = self.exchange_ids[idx]

            try:
                exchange = await self._ensure_trade_exchange(exchange_id)
                async with asyncio.timeout(_WS_TIMEOUT):
                    trades = await exchange.watch_trades(symbol)
                for t in trades:
                    if not self._running:
                        return
                    state.record_trade(
                        side=t.get("side", "buy"),
                        amount=float(t.get("amount", 0)),
                        timestamp=int(t.get("timestamp", 0)),
                    )
                state.trade_ws_failures = 0

            except asyncio.TimeoutError:
                state.trade_ws_failures += 1
                logger.debug(
                    "Trade stream timeout %s/%s (%d/%d)",
                    symbol, exchange_id, state.trade_ws_failures, _MAX_WS_FAILURES,
                )
                if state.trade_ws_failures >= _MAX_WS_FAILURES:
                    state.trade_exchange_idx += 1
                    state.trade_ws_failures = 0
                    logger.info("Failing over trade stream %s from %s after timeout", symbol, exchange_id)
                continue
            except Exception as exc:
                state.trade_ws_failures += 1
                logger.warning(
                    "Trade stream error %s/%s (%d/%d): %s",
                    symbol, exchange_id, state.trade_ws_failures, _MAX_WS_FAILURES, exc,
                )
                if state.trade_ws_failures >= _MAX_WS_FAILURES:
                    state.trade_exchange_idx += 1
                    state.trade_ws_failures = 0
                    logger.info("Failing over trade stream %s from %s", symbol, exchange_id)
                await asyncio.sleep(cfg.ws_reconnect_delay)

    # ── Funding rate poller ─────────────────────────────────────────────────

    async def _poll_funding(self, symbol: str) -> None:
        """Periodically fetch funding rate.  Tries the perp symbol first,
        falling back to the configured (spot) symbol."""
        cfg = get_settings()
        state = self._states[symbol]
        poll_s = cfg.feature_funding_poll_seconds
        perp_sym = _perp_symbol(symbol)

        while self._running:
            for exchange_id in self.exchange_ids:
                try:
                    exchange = await self._ensure_perp_exchange(exchange_id)

                    if not exchange.has.get("fetchFundingRate"):
                        continue

                    # Try perp symbol first, fall back to configured symbol
                    syms_to_try = [perp_sym, symbol]
                    for sym in syms_to_try:
                        try:
                            result = await exchange.fetch_funding_rate(sym)
                            rate = float(result.get("fundingRate", 0))
                            # Annualise (approximate — assumes 8h funding cycle
                            # common on Bybit / Binance; Hyperliquid uses 1h so
                            # the annualisation will be ~×3 lower than real).
                            state.record_funding(rate * 1095)  # approx annualised %
                            break
                        except Exception:
                            continue
                    break  # succeeded on one exchange
                except Exception:
                    continue

            await asyncio.sleep(poll_s)

    # ── Open interest poller ────────────────────────────────────────────────

    async def _poll_oi(self, symbol: str) -> None:
        """Periodically fetch open interest.  Tries perp symbol first."""
        cfg = get_settings()
        state = self._states[symbol]
        poll_s = cfg.feature_oi_poll_seconds
        perp_sym = _perp_symbol(symbol)

        while self._running:
            for exchange_id in self.exchange_ids:
                try:
                    exchange = await self._ensure_perp_exchange(exchange_id)

                    if not exchange.has.get("fetchOpenInterest"):
                        continue

                    syms_to_try = [perp_sym, symbol]
                    for sym in syms_to_try:
                        try:
                            result = await exchange.fetch_open_interest(sym)
                            oi = float(result.get("openInterestAmount", 0)
                                       or result.get("openInterest", 0))
                            state.record_oi(oi, price=0.0)
                            break
                        except Exception:
                            continue
                    break
                except Exception:
                    continue

            await asyncio.sleep(poll_s)


# ── Global singleton ─────────────────────────────────────────────────────────
_extractor: FeatureExtractor | None = None


def get_feature_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor
