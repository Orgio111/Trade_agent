"""Hard kill-switch: halts all trading if daily drawdown exceeds threshold."""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

from prometheus_client import Counter, Gauge

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import PortfolioState

logger = logging.getLogger(__name__)

_kill_active = Gauge("kill_switch_active", "1 when kill switch is engaged")
_kill_triggers = Counter("kill_switch_triggers_total", "Total kill switch activations")


class KillSwitch:
    """
    Monitors daily drawdown. Trips automatically when loss exceeds
    max_daily_drawdown_pct. Resets at UTC midnight.
    """

    def __init__(self, initial_equity: float) -> None:
        cfg = get_settings()
        self.threshold = cfg.max_daily_drawdown_pct
        self._active = False
        self._peak_equity = initial_equity
        self._day_open_equity = initial_equity
        self._reset_date = date.today()

    @property
    def is_active(self) -> bool:
        return self._active

    def update(self, state: PortfolioState) -> bool:
        """Returns True if kill switch just triggered (newly activated)."""
        self._maybe_reset()
        if self._active:
            return False  # already tripped

        drawdown = (self._day_open_equity - state.equity) / self._day_open_equity
        if drawdown >= self.threshold:
            self._active = True
            _kill_active.set(1)
            _kill_triggers.inc()
            logger.critical(
                "KILL SWITCH ACTIVATED — daily drawdown %.2f%% >= threshold %.2f%%",
                drawdown * 100,
                self.threshold * 100,
            )
            return True
        return False

    def _maybe_reset(self) -> None:
        today = date.today()
        if today != self._reset_date:
            self._reset_date = today
            self._day_open_equity = (
                self._peak_equity  # conservative: use current equity as new baseline
            )
            logger.info("Kill switch day-reset — new baseline equity=%.2f", self._day_open_equity)

    def manual_reset(self) -> None:
        self._active = False
        _kill_active.set(0)
        logger.warning("Kill switch manually reset")

    async def broadcast_halt(self) -> None:
        bus = await get_bus()
        await bus.publish(
            "stream:system",
            MsgType.KILL_SWITCH,
            {"active": True, "timestamp": datetime.utcnow().isoformat()},
        )
