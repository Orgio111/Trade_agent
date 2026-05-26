"""
PaperAccount — tracks simulated positions, P&L, equity curve, and trade journal
for paper trading mode.

Usage::

    account = PaperAccount(initial_capital=100_000.0)
    account.open_position("BTC/USDT", Side.BUY, 0.5, 67500.0)
    account.mark_to_market("BTC/USDT", 68000.0)   # daily MTM update
    result = account.close_position("BTC/USDT", 69000.0)
    print(account.total_pnl, account.win_rate, account.sharpe_ratio)
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from core.models import Side

logger = logging.getLogger(__name__)

# ── Position model ────────────────────────────────────────────────────────────


@dataclass
class PaperPosition:
    """An open position in the paper account."""
    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClosedTrade:
    """A closed trade with realised P&L."""
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    win: bool
    holding_period_s: float
    entry_time: datetime
    exit_time: datetime = field(default_factory=datetime.utcnow)


# ── PaperAccount ──────────────────────────────────────────────────────────────


class PaperAccount:
    """Tracks simulated trading activity — positions, P&L, equity curve.

    The account maintains its own :attr:`cash` and :attr:`equity` independently
    of the live :class:`PortfolioState`.  It is designed to be used alongside
    :class:`~agents.paper_execution.PaperExecutionEngine` for paper trading.

    Parameters
    ----------
    initial_capital:
        Starting cash balance.
    max_history:
        Maximum number of equity history points to retain.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_history: int = 500,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._positions: dict[str, PaperPosition] = {}
        self._closed_trades: list[ClosedTrade] = []

        # Rolling metrics
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._daily_start_equity = initial_capital
        self._equity_history: deque[dict] = deque(maxlen=max_history)
        self._equity_history.append({
            "t": datetime.utcnow().isoformat(),
            "v": round(initial_capital, 2),
        })

        # Per-symbol mark-to-market prices
        self._mtm_prices: dict[str, float] = {}

        # Track dates for daily P&L
        self._last_date: str | None = None

        logger.info("PaperAccount created — initial_capital=%.2f", initial_capital)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        """Current mark-to-market equity = cash + market value of open positions."""
        mtm_value = 0.0
        for symbol, pos in self._positions.items():
            mtm = self._mtm_prices.get(symbol, pos.entry_price)
            mtm_value += mtm * pos.quantity
        return self._cash + mtm_value

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def total_pnl(self) -> float:
        """Realised P&L from all closed trades."""
        return sum(t.pnl for t in self._closed_trades)

    @property
    def unrealised_pnl(self) -> float:
        return self._unrealised_pnl()

    @property
    def daily_pnl(self) -> float:
        return self.equity - self._daily_start_equity

    @property
    def daily_pnl_pct(self) -> float:
        if self._daily_start_equity == 0:
            return 0.0
        return self.daily_pnl / self._daily_start_equity

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.equity) / self._peak_equity

    @property
    def win_rate(self) -> float:
        """Fraction of closed trades that were profitable."""
        closed = len(self._closed_trades)
        if closed == 0:
            return 0.0
        wins = sum(1 for t in self._closed_trades if t.win)
        return wins / closed

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio based on per-trade returns."""
        if len(self._closed_trades) < 2:
            return 0.0
        returns = np.array([t.pnl_pct for t in self._closed_trades])
        excess = returns - 0.0  # assume risk-free rate = 0
        if excess.std() < 1e-10:
            return 0.0
        # Approximate annualisation: assume ~252 trading periods
        return float(excess.mean() / excess.std() * math.sqrt(252))

    @property
    def total_trades(self) -> int:
        return len(self._closed_trades)

    @property
    def open_positions(self) -> dict[str, PaperPosition]:
        return dict(self._positions)

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._closed_trades)

    @property
    def average_holding_period_s(self) -> float:
        if not self._closed_trades:
            return 0.0
        return float(np.mean([t.holding_period_s for t in self._closed_trades]))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _unrealised_pnl(self) -> float:
        total = 0.0
        for pos in self._positions.values():
            mtm = self._mtm_prices.get(pos.symbol, pos.entry_price)
            if pos.side == Side.BUY:
                total += (mtm - pos.entry_price) * pos.quantity
            else:
                total += (pos.entry_price - mtm) * pos.quantity
        return total

    def _update_equity_tracking(self) -> None:
        eq = self.equity
        if eq > self._peak_equity:
            self._peak_equity = eq
        self._equity_history.append({
            "t": datetime.utcnow().isoformat(),
            "v": round(eq, 2),
        })

        # Daily P&L boundary
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._last_date:
            self._daily_start_equity = eq
            self._last_date = today

    # ── Public API ────────────────────────────────────────────────────────────

    def mark_to_market(self, symbol: str, price: float) -> None:
        """Update the mark-to-market price for a symbol.

        This updates unrealised P&L for open positions and the equity curve.
        Call this on every bar to keep equity tracking accurate.
        """
        if self._mtm_prices.get(symbol) == price:
            return  # no change — skip redundant equity tracking update
        self._mtm_prices[symbol] = price
        self._update_equity_tracking()

    def open_position(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Open a new position.  Deducts cost from cash."""
        if symbol in self._positions:
            logger.warning(
                "PaperAccount: position already open for %s — "
                "close it before opening a new one", symbol,
            )
            return

        cost = quantity * price
        if cost > self._cash:
            logger.warning(
                "PaperAccount: insufficient cash for %s %.6f @ %.2f "
                "(need %.2f, have %.2f)",
                symbol, quantity, price, cost, self._cash,
            )
            return

        self._cash -= cost
        self._positions[symbol] = PaperPosition(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp or datetime.utcnow(),
        )
        self._mtm_prices[symbol] = price
        self._update_equity_tracking()

        logger.debug("PaperAccount OPEN %s %s %.6f @ %.2f", symbol, side.value, quantity, price)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime | None = None,
    ) -> ClosedTrade | None:
        """Close an open position.  Credits cash and records the trade."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            logger.debug("PaperAccount: no open position for %s to close", symbol)
            return None

        if pos.side == Side.BUY:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = pnl / (pos.entry_price * pos.quantity + 1e-10)

        # Credit proceeds
        proceeds = pos.quantity * exit_price
        self._cash += proceeds

        exit_ts = timestamp or datetime.utcnow()
        holding_s = (exit_ts - pos.entry_time).total_seconds()

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            win=pnl > 0,
            holding_period_s=max(holding_s, 0.0),
            entry_time=pos.entry_time,
            exit_time=exit_ts,
        )
        self._closed_trades.append(trade)
        self._update_equity_tracking()

        logger.info(
            "PaperAccount CLOSE %s %s pnl=%.2f (%s)",
            symbol, pos.side.value, pnl, "WIN" if pnl > 0 else "LOSS",
        )
        return trade

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable snapshot for the dashboard.

        Note: this does **not** append a new equity history entry — call
        :meth:`mark_to_market` first if you want the latest equity recorded.
        """
        return {
            "initial_capital": round(self._initial_capital, 2),
            "cash": round(self._cash, 2),
            "equity": round(self.equity, 2),
            "unrealised_pnl": round(self.unrealised_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl_pct * 100, 4),
            "drawdown_pct": round(self.current_drawdown_pct * 100, 4),
            "peak_equity": round(self.peak_equity, 2),
            "total_pnl": round(self.total_pnl, 2),
            "win_rate": round(self.win_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "total_trades": self.total_trades,
            "open_positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "quantity": round(p.quantity, 6),
                    "entry_price": round(p.entry_price, 4),
                    "entry_time": p.entry_time.isoformat(),
                }
                for p in self._positions.values()
            ],
            "equity_history": list(self._equity_history),
            "mode": "paper",
        }

    def reset(self) -> None:
        """Reset the account to initial state (e.g. for a fresh simulation)."""
        self._cash = self._initial_capital
        self._positions.clear()
        self._closed_trades.clear()
        self._equity = self._initial_capital
        self._peak_equity = self._initial_capital
        self._daily_start_equity = self._initial_capital
        self._equity_history.clear()
        self._equity_history.append({
            "t": datetime.utcnow().isoformat(),
            "v": round(self._initial_capital, 2),
        })
        self._mtm_prices.clear()
        self._last_date = None
        logger.info("PaperAccount reset to initial capital %.2f", self._initial_capital)
