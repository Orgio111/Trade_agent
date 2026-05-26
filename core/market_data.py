"""WebSocket market data feed with multi-exchange failover and simulated fallback.

Exchange priority flow per symbol
─────────────────────────────────
   (start)
      │
      ▼
  exch[0] WS ──error──→ exch[0] REST ──error──→ exch[1] WS ── ... ──→ simulated
      │                                                    │
      └── periodic recovery ───────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from datetime import datetime, timezone

import ccxt.pro as ccxtpro  # type: ignore[import]
import numpy as np

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import TickData

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────
_MAX_WS_FAILURES = 3           # WS failures before trying REST on same exchange
_WS_RECOVERY_INTERVAL = 300    # seconds between WS recovery attempts while in REST/simulated
_REST_POLL_SECONDS = 60        # OHLCV polling interval in REST mode
_WS_TIMEOUT = 10               # seconds before watch_ohlcv is considered timed out

# ── Simulated data defaults ──────────────────────────────────────────────────
_SIM_BASE_PRICES: dict[str, float] = {
    "BTC/USDT": 67_500.0,
    "ETH/USDT": 3_450.0,
    "SOL/USDT": 145.0,
    "ADA/USDT": 0.45,
}
_SIM_VOLATILITY: dict[str, float] = {
    "BTC/USDT": 0.006,
    "ETH/USDT": 0.008,
    "SOL/USDT": 0.012,
    "ADA/USDT": 0.015,
}


class ExchangeNotAvailable(Exception):
    """Raised when an exchange cannot be reached at all."""


class MarketDataFeed:
    """Real-time OHLCV feed across multiple exchanges with automatic failover.

    For each symbol the feed walks through the priority-ordered exchange list
    trying WebSocket → REST polling before advancing to the next exchange.
    If all exchanges are exhausted the feed falls back to a geometric
    random-walk simulator and periodically retries the primary exchange.

    Public API
    ──────────
    start()       — launch per-symbol streaming coroutines
    stop()        — close all exchange connections
    get_closes()  — numpy array of close prices for a symbol
    get_highs()   — numpy array of high prices
    get_lows()    — numpy array of low prices
    get_volumes() — numpy array of volumes
    latest_tick() — most recent TickData for a symbol
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self.symbols = cfg.symbols
        self.exchange_ids: list[str] = cfg.exchanges
        self._exchanges: dict[str, ccxtpro.Exchange] = {}
        self._running = False
        self._simulated_symbols: set[str] = set()  # symbols currently in simulated mode

        # Per-symbol ring buffer (shared across exchanges)
        self._tick_buffer: dict[str, deque[TickData]] = {
            s: deque(maxlen=500) for s in self.symbols
        }

        # Per-symbol exchange progression
        self._active_idx: dict[str, int] = {}          # symbol → index in exchange_ids
        self._ws_failures: dict[str, int] = {}          # symbol → consecutive WS failures on current exchange
        self._rest_mode: dict[str, bool] = {}           # symbol → True → do REST poll
        self._last_recovery_ts: dict[str, float] = {}   # symbol → last recovery attempt timestamp

        # Simulator state (geometric random walk)
        self._sim_prices: dict[str, float] = {}
        self._sim_seed: dict[str, np.random.Generator] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        tasks = [self._stream_symbol(s) for s in self.symbols]
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        self._running = False
        for exchange in self._exchanges.values():
            try:
                await exchange.close()
            except Exception:
                pass

    # ── Exchange helpers ─────────────────────────────────────────────────────

    async def _ensure_exchange(self, exchange_id: str) -> ccxtpro.Exchange:
        """Lazy-init a ccxt.pro exchange instance and load its markets."""
        if exchange_id not in self._exchanges:
            ExchangeClass = getattr(ccxtpro, exchange_id, None)
            if ExchangeClass is None:
                raise ExchangeNotAvailable(f"Unknown exchange: {exchange_id}")
            exchange = ExchangeClass({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            try:
                await exchange.load_markets()
                logger.info(
                    "Loaded %d markets from %s",
                    len(exchange.markets), exchange_id,
                )
            except Exception as exc:
                logger.warning("Failed to load markets from %s: %s", exchange_id, exc)
                raise ExchangeNotAvailable(str(exc)) from exc
            self._exchanges[exchange_id] = exchange
        return self._exchanges[exchange_id]

    # ── Simulator ────────────────────────────────────────────────────────────

    def _ensure_sim_seed(self, symbol: str) -> None:
        """Lazy-init simulator state for a single symbol if not already seeded."""
        if symbol in self._sim_prices:
            return
        base = _SIM_BASE_PRICES.get(symbol, 50_000.0)
        self._sim_prices[symbol] = base
        self._sim_seed[symbol] = np.random.default_rng(hash(symbol) & 0xFFFF_FFFF)
        # Pre-warm 100 historical ticks so the supervisor can trade immediately
        for _ in range(100):
            tick = self._generate_sim_tick(symbol)
            self._tick_buffer[symbol].append(tick)

    def _generate_sim_tick(self, symbol: str) -> TickData:
        rng = self._sim_seed[symbol]
        vol = _SIM_VOLATILITY.get(symbol, 0.01)
        prev_close = self._sim_prices[symbol]

        drift = 0.0001
        log_ret = rng.normal(drift * vol, vol)
        close = prev_close * math.exp(log_ret)
        close = max(close, prev_close * 0.95)

        intra_vol = vol * 0.4
        open_ = prev_close
        high = max(open_, close) * (1 + abs(rng.normal(0, intra_vol)))
        low = min(open_, close) * (1 - abs(rng.normal(0, intra_vol)))
        volume = rng.exponential(scale=200) + 50

        self._sim_prices[symbol] = close

        return TickData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            open=round(open_, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close, 2),
            volume=round(volume, 2),
        )

    # ── Per-symbol stream ────────────────────────────────────────────────────

    async def _stream_symbol(self, symbol: str) -> None:
        """Main loop for a symbol — walks exchange priority until one works,
        then streams from it, failing over to the next exchange on errors."""
        cfg = get_settings()
        bus = await get_bus()

        while self._running:
            # ── Simulated mode (all exchanges exhausted) ────────────────────
            if symbol in self._simulated_symbols:
                try:
                    tick = self._generate_sim_tick(symbol)
                    self._tick_buffer[symbol].append(tick)
                    await bus.publish(
                        cfg.stream_market_data,
                        MsgType.TICK,
                        tick.model_dump(mode="json"),
                    )
                except Exception as exc:
                    logger.error("Sim tick error for %s: %s", symbol, exc)

                # Periodically try to recover a live exchange
                last_ts = self._last_recovery_ts.get(symbol, 0.0)
                if time.monotonic() - last_ts >= _WS_RECOVERY_INTERVAL:
                    self._last_recovery_ts[symbol] = time.monotonic()
                    if await self._try_recover(symbol):
                        self._simulated_symbols.discard(symbol)
                await asyncio.sleep(60)
                continue

            # ── Determine active exchange ───────────────────────────────────
            idx = self._active_idx.get(symbol, 0)
            if idx >= len(self.exchange_ids):
                logger.warning(
                    "All exchanges exhausted for %s — entering simulated mode", symbol
                )
                self._simulated_symbols.add(symbol)
                if symbol not in self._sim_prices:
                    self._ensure_sim_seed(symbol)
                continue

            exchange_id = self.exchange_ids[idx]

            # Try to initialise this exchange (lazy, may fail)
            try:
                exchange = await self._ensure_exchange(exchange_id)
            except ExchangeNotAvailable:
                logger.warning("Exchange %s unavailable for %s — advancing", exchange_id, symbol)
                self._advance_exchange(symbol)
                continue

            # ── REST fallback for current exchange ──────────────────────────
            if self._rest_mode.get(symbol, False):
                try:
                    ohlcv_list = await exchange.fetch_ohlcv(symbol, "1m", limit=1)
                    if ohlcv_list:
                        await self._publish_tick(symbol, ohlcv_list[-1], bus, cfg)
                    await asyncio.sleep(_REST_POLL_SECONDS)

                    # Periodically try to recover WS on this exchange
                    last_ts = self._last_recovery_ts.get(symbol, 0.0)
                    if time.monotonic() - last_ts >= _WS_RECOVERY_INTERVAL:
                        logger.info("Attempting WS recovery on %s for %s", exchange_id, symbol)
                        self._rest_mode[symbol] = False
                        self._ws_failures[symbol] = 0
                        self._last_recovery_ts[symbol] = time.monotonic()
                    continue
                except Exception as exc:
                    logger.warning("REST error %s/%s: %s", symbol, exchange_id, exc)
                    self._advance_exchange(symbol)
                    continue

            # ── WebSocket primary path ──────────────────────────────────────
            reconnect_delay = cfg.ws_reconnect_delay
            try:
                async with asyncio.timeout(_WS_TIMEOUT):
                    ohlcv = await exchange.watch_ohlcv(symbol, "1m")
                if ohlcv:
                    await self._publish_tick(symbol, ohlcv[-1], bus, cfg)
                    self._ws_failures[symbol] = 0
            except asyncio.TimeoutError:
                pass  # No new candle yet — loop and try again
            except Exception as exc:
                self._ws_failures[symbol] = self._ws_failures.get(symbol, 0) + 1
                logger.warning(
                    "WS error %s/%s (%d/%d): %s",
                    symbol, exchange_id,
                    self._ws_failures[symbol], _MAX_WS_FAILURES,
                    exc,
                )

                if self._ws_failures[symbol] >= _MAX_WS_FAILURES:
                    # Try REST before advancing to next exchange
                    try:
                        ohlcv_list = await exchange.fetch_ohlcv(symbol, "1m", limit=1)
                        if ohlcv_list:
                            await self._publish_tick(symbol, ohlcv_list[-1], bus, cfg)
                            # REST worked — switch to REST mode on this exchange
                            self._rest_mode[symbol] = True
                            self._ws_failures[symbol] = 0
                            continue
                    except Exception:
                        pass
                    # Both WS and REST failed — advance to next exchange
                    logger.info("Failing over %s from %s", symbol, exchange_id)
                    self._advance_exchange(symbol)
                else:
                    await asyncio.sleep(reconnect_delay)

    def _advance_exchange(self, symbol: str) -> None:
        """Move a symbol to the next exchange in priority order (or wrap to -1)."""
        current = self._active_idx.get(symbol, 0)
        next_idx = current + 1
        if next_idx >= len(self.exchange_ids):
            self._active_idx[symbol] = len(self.exchange_ids)  # will trigger simulated
        else:
            self._active_idx[symbol] = next_idx
        # Reset per-symbol failure state for the new exchange
        self._ws_failures.pop(symbol, None)
        self._rest_mode.pop(symbol, None)

    async def _try_recover(self, symbol: str) -> bool:
        """Try to re-establish a live connection to the primary exchange.
        Returns True if successful."""
        for idx, exchange_id in enumerate(self.exchange_ids):
            try:
                exchange = await self._ensure_exchange(exchange_id)
                # Quick connectivity check
                await exchange.fetch_ohlcv(symbol, "1m", limit=1)
                logger.info("Recovered live connection via %s for %s", exchange_id, symbol)
                self._active_idx[symbol] = idx
                self._ws_failures.pop(symbol, None)
                self._rest_mode.pop(symbol, None)
                return True
            except Exception:
                continue
        return False

    # ── Tick publishing ──────────────────────────────────────────────────────

    async def _publish_tick(
        self,
        symbol: str,
        bar: list,
        bus,
        cfg,
    ) -> None:
        """Convert an OHLCV bar to TickData, buffer it, and publish."""
        tick = TickData(
            symbol=symbol,
            timestamp=datetime.utcfromtimestamp(bar[0] / 1000),
            open=bar[1],
            high=bar[2],
            low=bar[3],
            close=bar[4],
            volume=bar[5],
        )
        self._tick_buffer[symbol].append(tick)
        await bus.publish(
            cfg.stream_market_data,
            MsgType.TICK,
            tick.model_dump(mode="json"),
        )

    # ── Public helpers (called by the supervisor) ────────────────────────────

    def get_closes(self, symbol: str) -> np.ndarray:
        return np.array([t.close for t in self._tick_buffer[symbol]])

    def get_highs(self, symbol: str) -> np.ndarray:
        return np.array([t.high for t in self._tick_buffer[symbol]])

    def get_lows(self, symbol: str) -> np.ndarray:
        return np.array([t.low for t in self._tick_buffer[symbol]])

    def get_volumes(self, symbol: str) -> np.ndarray:
        return np.array([t.volume for t in self._tick_buffer[symbol]])

    def latest_tick(self, symbol: str) -> TickData | None:
        buf = self._tick_buffer[symbol]
        return buf[-1] if buf else None
