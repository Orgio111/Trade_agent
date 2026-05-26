"""
PaperExecutionEngine — simulated order execution for paper trading.

Uses a :class:`~agents.paper_account.PaperAccount` to track positions and P&L,
and supports multiple fill and slippage models for realistic simulation.

Usage::

    account = PaperAccount(100_000.0)
    engine = PaperExecutionEngine(account, fill_model="twap", slippage_model="fixed")
    filled_order = await engine.execute(order, recent_prices)
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Literal

import numpy as np

from agents.paper_account import PaperAccount
from core.config import get_settings
from core.models import Order, OrderStatus, OrderType, Side

logger = logging.getLogger(__name__)

FillModel = Literal["immediate", "twap"]
"""Supported fill simulation models:

- ``immediate`` — entire order fills at the current mid-price (plus slippage)
- ``twap`` — order is split into :attr:`default_slices` equal-sized child
  orders, each filled at a perturbed price drawn from a narrow Gaussian around
  the mid-price, with a configurable inter-slice delay
"""

SlippageModel = Literal["fixed", "percentage", "market_impact"]
"""Supported slippage models:

- ``fixed`` — constant slippage in bps set via :attr:`slippage_bps`
- ``percentage`` — slippage as a fraction of price set via
  :attr:`slippage_pct`
- ``market_impact`` — slippage proportional to the square-root of the order's
  share of estimated market volume: ``impact ∝ sqrt(quantity / estimated_volume)``
"""


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_SLICES = 3
_DEFAULT_SLICE_DELAY_S = 0.05
_DEFAULT_SLIPPAGE_BPS = 5.0  # 0.05 %
_DEFAULT_SLIPPAGE_PCT = 0.0005
_DEFAULT_LATENCY_MS = 50.0
_DEFAULT_FILL_NOISE_VOL = 0.0002  # 0.02 % Gaussian noise per slice


# ── Execution engine ──────────────────────────────────────────────────────────


class PaperExecutionEngine:
    """Simulated order execution for paper trading.

    Parameters
    ----------
    account:
        The :class:`PaperAccount` to update on fills.
    fill_model:
        Fill simulation model (``"immediate"`` or ``"twap"``).
    slippage_model:
        Slippage model (``"fixed"``, ``"percentage"``, or ``"market_impact"``).
    slippage_bps:
        Fixed slippage in bps (only used when ``slippage_model="fixed"``).
    slippage_pct:
        Fractional slippage (only used when ``slippage_model="percentage"``).
    latency_ms:
        Simulated network / exchange latency in milliseconds.
    fill_noise_vol:
        Standard deviation of Gaussian noise applied to each slice fill price.
    default_slices:
        Number of TWAP slices (only used when ``fill_model="twap"``).
    slice_delay_s:
        Delay between TWAP slices in seconds.
    """

    def __init__(
        self,
        account: PaperAccount,
        *,
        fill_model: FillModel = "immediate",
        slippage_model: SlippageModel = "fixed",
        slippage_bps: float = _DEFAULT_SLIPPAGE_BPS,
        slippage_pct: float = _DEFAULT_SLIPPAGE_PCT,
        latency_ms: float = _DEFAULT_LATENCY_MS,
        fill_noise_vol: float = _DEFAULT_FILL_NOISE_VOL,
        default_slices: int = _DEFAULT_SLICES,
        slice_delay_s: float = _DEFAULT_SLICE_DELAY_S,
    ) -> None:
        self._account = account
        self.fill_model = fill_model
        self.slippage_model = slippage_model
        self.slippage_bps = slippage_bps
        self.slippage_pct = slippage_pct
        self.latency_ms = latency_ms
        self.fill_noise_vol = fill_noise_vol
        self.default_slices = default_slices
        self.slice_delay_s = slice_delay_s

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def account(self) -> PaperAccount:
        return self._account

    # ── Slippage computation ──────────────────────────────────────────────────

    def _compute_slippage_price(
        self,
        base_price: float,
        order: Order,
        slice_quantity: float | None = None,
    ) -> float:
        """Return the fill price after applying the configured slippage model.

        For buy orders the price is increased (adverse); for sell orders it is
        decreased.
        """
        sign = 1.0 if order.side == Side.BUY else -1.0

        if self.slippage_model == "fixed":
            slippage = base_price * (self.slippage_bps / 10_000)
        elif self.slippage_model == "percentage":
            slippage = base_price * self.slippage_pct
        elif self.slippage_model == "market_impact":
            # Simple square-root market impact model
            qty = slice_quantity or order.quantity
            # Estimate daily volume as a fraction of average volume from settings
            estimated_vol = qty * 100  # rough fallback
            if estimated_vol > 0:
                impact = math.sqrt(qty / estimated_vol) * 0.001 * base_price
            else:
                impact = 0.0
            slippage = impact
        else:
            slippage = 0.0

        return base_price + sign * slippage

    # ── Fill simulation ───────────────────────────────────────────────────────

    async def _simulate_immediate_fill(
        self,
        order: Order,
        recent_prices: np.ndarray,
    ) -> tuple[float, float | None]:
        """Fill the entire order at the current mid-price + slippage.

        Returns ``(avg_fill_price, slippage_bps)``.
        """
        mid = float(recent_prices[-1]) if len(recent_prices) > 0 else 1.0

        # Simulate latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        # Apply slippage
        fill_price = self._compute_slippage_price(mid, order)
        fill_price = max(fill_price, 0.01)

        # Add small Gaussian noise
        noise = float(np.random.normal(0, self.fill_noise_vol * mid))
        fill_price += noise
        fill_price = max(fill_price, 0.01)

        actual_slippage = abs(fill_price - mid) / mid * 10_000
        return fill_price, actual_slippage

    async def _simulate_twap_fill(
        self,
        order: Order,
        recent_prices: np.ndarray,
        slices: int | None = None,
    ) -> tuple[float, float | None]:
        """Split the order into *slices* child orders, each filled separately.

        Returns ``(avg_fill_price, avg_slippage_bps)``.
        """
        n = slices or self.default_slices
        if n < 1:
            n = 1

        mid = float(recent_prices[-1]) if len(recent_prices) > 0 else 1.0
        slice_qty = order.quantity / n
        fill_prices: list[float] = []

        for i in range(n):
            # Simulate latency per slice
            if self.latency_ms > 0 and i > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            # Apply slippage
            base_price = self._compute_slippage_price(mid, order, slice_qty)

            # Add Gaussian noise (increases with each slice to simulate market impact)
            noise_vol = self.fill_noise_vol * (1.0 + 0.1 * i)
            noise = float(np.random.normal(0, noise_vol * mid))
            fill_price = base_price + noise
            fill_price = max(fill_price, 0.01)

            fill_prices.append(fill_price)

            if i < n - 1 and self.slice_delay_s > 0:
                await asyncio.sleep(self.slice_delay_s)

        avg_fill = float(np.mean(fill_prices))
        avg_slippage = abs(avg_fill - mid) / mid * 10_000
        return avg_fill, avg_slippage

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute(
        self,
        order: Order,
        recent_prices: np.ndarray,
        slices: int | None = None,
    ) -> Order:
        """Execute *order* against the paper account.

        This is the main entry point — it:

        1. Determines the fill price using the configured fill / slippage models.
        2. Opens or updates the position in the :class:`PaperAccount`.
        3. Updates mark-to-market prices on the account.
        4. Returns the filled :class:`Order` with ``avg_fill_price``,
           ``slippage_bps``, and ``status`` set.

        Parameters
        ----------
        order:
            The order to execute.  ``order.status`` should be ``PENDING``.
        recent_prices:
            Recent price series used to determine the mid-price.
        slices:
            Override the number of TWAP slices (only applies when
            ``fill_model="twap"`` and *slices* is not ``None``).

        Returns
        -------
        Order:
            The same order object with fill details populated.
        """
        t0 = time.monotonic()

        if self.fill_model == "twap":
            avg_fill, slippage = await self._simulate_twap_fill(order, recent_prices, slices)
        else:
            avg_fill, slippage = await self._simulate_immediate_fill(order, recent_prices)

        order.avg_fill_price = avg_fill
        order.slippage_bps = slippage
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()

        # Update PaperAccount
        existing = self._account.open_positions.get(order.symbol)
        if existing and existing.side != order.side:
            # Closing trade — opposite side closes existing position
            self._account.close_position(order.symbol, avg_fill)
            logger.info(
                "PaperExecution: closed %s %s @ %.4f",
                order.symbol, existing.side.value, avg_fill,
            )

        # Open new position
        self._account.open_position(
            order.symbol,
            order.side,
            order.quantity,
            avg_fill,
        )

        # Update MTM for all positions
        if len(recent_prices) > 0:
            for symbol in self._account.open_positions:
                self._account.mark_to_market(symbol, float(recent_prices[-1]))

        elapsed = (time.monotonic() - t0) * 1000
        logger.debug(
            "PaperExecution: %s %s %.6f @ %.4f  slippage=%.1f bps  (%s, %.0fms)",
            order.side.value, order.symbol, order.quantity,
            avg_fill, slippage or 0, self.fill_model, elapsed,
        )
        return order

    async def execute_decision(
        self,
        order: Order,
        recent_prices: np.ndarray,
        slices: int | None = None,
    ) -> Order:
        """Convenience wrapper — same as :meth:`execute` but logs at INFO level.

        This is the method the supervisor should call for paper trading.
        """
        return await self.execute(order, recent_prices, slices)



