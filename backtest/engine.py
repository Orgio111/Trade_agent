"""Backtesting engine: replays historical OHLCV bars through the full supervisor
pipeline and returns a BacktestResult."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from backtest.data_loader import load_data
from backtest.models import BacktestConfig, BacktestResult, BacktestTrade
from backtest.report import compute_performance
from agents.supervisor import PortfolioSupervisor

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Replay historical OHLCV bars through the supervisor trading pipeline.

    Typical usage::

        engine = BacktestEngine(cfg)
        result = await engine.run()

    The engine loads historical data, creates a PortfolioSupervisor, and feeds
    bars one at a time through ``run_cycle()``, tracking portfolio state,
    trades, and PnL as it goes.

    Notes on symbol processing
    --------------------------
    For each bar the engine runs the supervisor cycle for **one** symbol
    (round-robin across all configured symbols).  This mirrors the live
    architecture where each symbol's cycle is independent.  For a 2-symbol
    portfolio on a 1h timeframe, each symbol gets processed every 2 hours.
    """

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self._data: dict[str, list] | None = None  # symbol -> list[BarRecord]

    async def run(self) -> BacktestResult:
        """Run the backtest and return the result."""
        logger.info(
            "Backtest starting: %s %s..%s capital=%.0f timeframe=%s",
            self.config.symbols,
            self.config.start,
            self.config.end,
            self.config.initial_capital,
            self.config.timeframe,
        )

        # ── 1. Load historical data ──────────────────────────────────────────
        data = await load_data(self.config)
        self._data = data
        if not data:
            return BacktestResult(
                config=self.config,
                errors=["No historical data loaded -- check exchange/date range."],
            )

        n_bars = min(len(bars) for bars in data.values())
        if self.config.max_cycles > 0:
            n_bars = min(n_bars, self.config.max_cycles)

        logger.info("Replaying %d bars", n_bars)

        # ── 2. Create & start supervisor ─────────────────────────────────────
        supervisor = PortfolioSupervisor(initial_equity=self.config.initial_capital)
        try:
            await supervisor.start()
        except Exception as exc:
            logger.warning("Supervisor start failed (continuing): %s", exc)

        # ── 3. Bar-by-bar replay ─────────────────────────────────────────────
        trades: list[BacktestTrade] = []
        equity_curve: list[float] = []
        equity_timestamps: list[datetime] = []
        open_trades: dict[str, BacktestTrade] = {}
        errors: list[str] = []

        equity = self.config.initial_capital
        peak_equity = equity

        for i in range(n_bars):
            # Build per-symbol snapshots for this bar
            all_closes: list[float] = []
            bar_ts: datetime | None = None

            for symbol in self.config.symbols:
                symbol_bars = data[symbol][: i + 1]
                closes = [b.close for b in symbol_bars]
                all_closes.append(closes[-1])
                if bar_ts is None:
                    bar_ts = symbol_bars[-1].timestamp

            # Select one symbol this cycle (round-robin)
            symbol_idx = i % len(self.config.symbols)
            symbol = self.config.symbols[symbol_idx]
            symbol_bars = data[symbol][: i + 1]
            closes = [b.close for b in symbol_bars]
            highs = [b.high for b in symbol_bars]
            lows = [b.low for b in symbol_bars]
            volumes = [b.volume for b in symbol_bars]
            current_price = closes[-1]

            # Check for stop-loss on open positions before running cycle
            if symbol in open_trades:
                ot = open_trades[symbol]
                if ot.side == "BUY":
                    stop_price = ot.entry_price * (1.0 - self.config.stop_loss_pct)
                    if current_price <= stop_price:
                        ot.exit_time = bar_ts or datetime.now(timezone.utc)
                        ot.exit_price = current_price
                        pnl = (current_price - ot.entry_price) * ot.quantity
                        ot.pnl = pnl
                        ot.pnl_pct = (current_price - ot.entry_price) / ot.entry_price
                        ot.exit_reason = "stop_loss"
                        equity += pnl
                        trades.append(ot)
                        del open_trades[symbol]
                elif ot.side == "SELL":
                    # Short position: stop-loss triggered when price rises
                    stop_price = ot.entry_price * (1.0 + self.config.stop_loss_pct)
                    if current_price >= stop_price:
                        ot.exit_time = bar_ts or datetime.now(timezone.utc)
                        ot.exit_price = current_price
                        pnl = (ot.entry_price - current_price) * ot.quantity
                        ot.pnl = pnl
                        ot.pnl_pct = (ot.entry_price - current_price) / ot.entry_price
                        ot.exit_reason = "stop_loss"
                        equity += pnl
                        trades.append(ot)
                        del open_trades[symbol]

            # Run the supervisor cycle
            if len(closes) >= 50:
                try:
                    order = await supervisor.run_cycle(
                        symbol=symbol,
                        closes=closes,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                    )

                    # Track trades from order events
                    if order and symbol not in open_trades:
                        entry_price = order.avg_fill_price or current_price
                        qty = order.quantity
                        cost = entry_price * qty
                        if cost > equity:
                            qty = equity / entry_price if entry_price > 0 else 0.0

                        trade = BacktestTrade(
                            symbol=symbol,
                            side=order.side.value,
                            entry_time=bar_ts or datetime.now(timezone.utc),
                            entry_price=entry_price,
                            quantity=qty,
                            rationale=order.supervisor_rationale,
                        )
                        open_trades[symbol] = trade

                except Exception as exc:
                    errors.append(f"Cycle error at bar {i} ({symbol}): {exc}")
                    logger.warning("Backtest cycle error at bar %d: %s", i, exc)

            # ── Mark-to-market PnL ────────────────────────────────────────────
            mtm_equity = equity
            for sym, ot in open_trades.items():
                sym_idx = self.config.symbols.index(sym)
                price = all_closes[sym_idx]
                if ot.side == "BUY":
                    mtm_equity += (price - ot.entry_price) * ot.quantity
                else:
                    mtm_equity += (ot.entry_price - price) * ot.quantity

            equity = mtm_equity
            peak_equity = max(peak_equity, equity)
            equity_curve.append(equity)
            equity_timestamps.append(bar_ts or datetime.now(timezone.utc))

            if (i + 1) % 1000 == 0:
                logger.info(
                    "  Processed %d/%d bars -- equity=%.2f %s",
                    i + 1, n_bars, equity,
                    f"({(equity / self.config.initial_capital - 1) * 100:+.2f}%)",
                )

        # ── 4. Close any remaining open trades at last price ──────────────────
        for sym, ot in list(open_trades.items()):
            sym_idx = self.config.symbols.index(sym)
            last_price = all_closes[sym_idx] if all_closes else 0.0
            ot.exit_time = (
                equity_timestamps[-1] if equity_timestamps
                else datetime.now(timezone.utc)
            )
            ot.exit_price = last_price
            if ot.side == "BUY":
                pnl = (last_price - ot.entry_price) * ot.quantity
            else:
                pnl = (ot.entry_price - last_price) * ot.quantity
            ot.pnl = pnl
            ot.pnl_pct = pnl / (ot.entry_price * ot.quantity + 1e-10)
            trades.append(ot)

        # ── 5. Compute performance metrics ────────────────────────────────────
        result = compute_performance(
            config=self.config,
            trades=trades,
            equity_curve=equity_curve,
            equity_timestamps=equity_timestamps,
            errors=errors,
        )
        result.peak_equity = peak_equity

        logger.info(
            "Backtest complete: return=%.2f%% trades=%d sharpe=%.2f maxdd=%.2f%%",
            result.total_return_pct,
            result.total_trades,
            result.sharpe_ratio,
            result.max_drawdown_pct,
        )
        return result
