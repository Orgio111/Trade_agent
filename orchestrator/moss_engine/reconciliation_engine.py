"""Safe-by-Default Infrastructure — Krypt-Trader Disaster Recovery.

Boot-time reconciliation:
  - reconcile_ledger_with_exchange() compares bot internal DB cache
    against Binance account snapshot (open positions, cash balance, margin)
  - Generates fixing trades if orphan states detected

Hardware guardrails:
  - Hard daily trailing stop-loss (5% absolute)
  - Max take-profit (50% absolute)
  - Master kill-switch: boolean toggle → immediately zero-initialized outputs

Built atop SkillRegistry's arm_kill_switch() for instant kill.

Security: credentials exclusively via python-dotenv / .env.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ccxt
import numpy as np

from .schemas import (
    GuardrailStatus,
    MOSS_FACTORY_VERSION,
    ReconciliationReport,
    SkillMeta,
)
from .skill_registry import SkillRegistry

logger = logging.getLogger(__name__)

# ── .env auto-load ───────────────────────
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parents[2]
    _env_file = _project_root / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════
# MODULE 3:  Safe Infrastructure
# ══════════════════════════════════════════════════════════════════════

class ReconciliationEngine:
    """Boot-time book reconciliation machine."""

    def __init__(self,
                 exchange_config: Optional[Dict[str, Any]] = None,
                 registry: Optional[SkillRegistry] = None) -> None:
        """Exchange: ccxt.Binance() | ccxt.Binance.usdm_testnet.

        Registry: updates guardrail state after reconciliation.
        """
        apikey = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_SECRET", "")
        exchange_id = "binance"
        defaults = {
            "apiKey": apikey,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"}
        }

        self.exchange = ccxt.binance({**defaults, **(exchange_config or {})})
        self._registry = registry

        # Default guardrail thresholds
        self.max_daily_loss_pct: float = 5.0     # absolute
        self.max_take_profit_pct: float = 50.0  # absolute

    async def reconcile_ledger_with_exchange(
        self,
        internal_cache: Dict[str, Any],
        ignore_threshold_pct: float = 0.01
    ) -> ReconciliationReport:
        """Reconcile bot internal DB cache ↔ Binance account snapshot.

        Generates fixing trades to close orphan positions.
        Returns ReconciliationReport.
        """
        if not internal_cache:
            logger.warning("Empty internal_cache — assuming boot virgin state")
            return ReconciliationReport(
                matched_positions=0,
                orphan_positions=0,
                generated_fixing_trades=0,
            )

        # Fetch Binance snapshot
        account = await self.exchange.fetch_balance()
        position_snapshot: Dict[str, Any] = account["info"]["positions"]
        cash_balance = float(account["free"])

        # Check cache
        bot_positions: Dict[str, float] = {p["symbol"]: p["notional"] for p in internal_cache.get("positions", [])}
        binance_positions: Dict[str, float] = {}
        for pos in position_snapshot:
            if str(pos["symbol"]).startswith("BTC"):
                qty = float(pos["positionAmt"])
                if abs(qty) > ignore_threshold_pct * cash_balance:
                    binance_positions[pos["symbol"]] = qty

        # Match bolt-on logic
        matched = []
        orphan_entries = []
        fixing_trades = []

        all_symbols = list(set(list(bot_positions.keys()) + list(binance_positions.keys())))

        for symbol in all_symbols:
            bot_pos = bot_positions.get(symbol, 0.0)
            exch_pos = binance_positions.get(symbol, 0.0)
            
            # Tolerance check
            if abs(bot_pos - exch_pos) <= ignore_threshold_pct:
                matched.append(symbol)
            else:
                notional_diff = bot_pos - exch_pos
                orphan_entries.append({
                    "symbol": symbol,
                    "bot_pos": bot_pos,
                    "exch_pos": exch_pos,
                    "notional_diff": notional_diff,
                })
                # Generate fixing trade
                order_side = "sell" if notional_diff > 0 else "buy"
                order_qty = abs(notional_diff)
                fixing_trades.append((symbol, order_side, order_qty))

        # Update guardrail state
        if self._registry:
            total_pnl = sum(state.daily_pnl for state in self._registry._states.values())
            daily_drawdown = -min(0.0, total_pnl)
            composite_state = self._registry.get_state("moss_composite")
            if composite_state is not None:
                self._registry.update_state(
                    "moss_composite",
                    daily_pnl=total_pnl,
                    daily_drawdown_pct=daily_drawdown,
                    consecutive_losses=max(0, composite_state.consecutive_losses - 1),
                )

        report = ReconciliationReport(
            matched_positions=len(matched),
            orphan_positions=len(orphan_entries),
            generated_fixing_trades=len(fixing_trades),
            discrepancies=orphan_entries,
        )
        logger.info("[Reconciliation] %s matched, %s orphans → %s fixing trades",
                   len(matched), len(orphan_entries), len(fixing_trades))
        return report

    async def check_and_trigger_guardrails(
        self,
        registry: SkillRegistry
    ) -> GuardrailStatus:
        """Evaluate guardrail thresholds and flip master kill-switch if breached."""
        total_pnl = 0.0
        daily_trailing_high = 0.0
        consecutive_losses = 0
        daily_drawdown_pct = 0.0

        # Aggregate across all skills
        for name, state in registry._states.items():
            total_pnl += state.daily_pnl
            daily_trailing_high = max(daily_trailing_high, state.daily_pnl)
            consecutive_losses += state.consecutive_losses
            
        # Percentage drawdown
        daily_trailing_high = daily_trailing_high or 1.0
        daily_drawdown_pct = (daily_trailing_high - total_pnl) / daily_trailing_high * 100.0
        
        # Check thresholds
        max_daily_breach = daily_drawdown_pct >= self.max_daily_loss_pct
        max_tp_breach = total_pnl > self.max_take_profit_pct
        max_loss_streak = consecutive_losses >= 4

        # Decision
        if max_daily_breach or max_tp_breach or max_loss_streak:
            if not registry.master_kill_switch:
                registry.arm_kill_switch(True)
                logger.critical("GUARDRAIL TRIPPED — MASTER KILL-SWITCH ARMED")
            reason = []
            if max_daily_breach:
                reason.append(f"daily_drawdown={daily_drawdown_pct:.2f}% >= max={self.max_daily_loss_pct:.2f}%")
            if max_tp_breach:
                reason.append(f"pnl={total_pnl:.2f} > max_take_profit={self.max_take_profit_pct:.2f}")
            if max_loss_streak:
                reason.append(f"loss streak={consecutive_losses} >= max_consecutive_losses=4")
        else:
            if registry.master_kill_switch:
                registry.arm_kill_switch(False)
                logger.info("Guardrails armed=false — resumed")
            reason = ""

        # Repack state
        status = GuardrailStatus(
            master_kill_switch=registry.master_kill_switch,
            daily_pnl=total_pnl,
            daily_trailing_high=daily_trailing_high,
            daily_drawdown_pct=daily_drawdown_pct,
            max_daily_loss_pct=self.max_daily_loss_pct,
            max_take_profit_pct=self.max_take_profit_pct,
            consecutive_losses=consecutive_losses,
            max_consecutive_losses=4,
            is_triggered=max_daily_breach or max_tp_breach or max_loss_streak,
            trigger_reason=", ".join(reason)
        )
        return status

    @property
    def version(self) -> str:
        return MOSS_FACTORY_VERSION


# ══════════════════════════════════════════════════════════════════════
# __main__ Simulation Runner
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_reconciliation():
        """Mock reconciliation."""
        registry = SkillRegistry()
        engine = ReconciliationEngine(registry=registry)
        
        # Mock internal cache
        internal_cache = {
            "positions": [
                {"symbol": "BTCUSDT", "notional": 10_000},
                {"symbol": "ETHUSDT", "notional": 5_000},
            ]
        }
        
        # Mock Binance → simulate mismatch
        report = await engine.reconcile_ledger_with_exchange(internal_cache)
        status = await engine.check_and_trigger_guardrails(registry)
        
        print("Reconciliation Report:")
        print(json.dumps(asdict(report), indent=2))
        print("Guardrail Status:")
        print(json.dumps(asdict(status), indent=2))
        print(f"Kill-switch armed: {status.master_kill_switch}")
    
    asyncio.run(demo_reconciliation())
