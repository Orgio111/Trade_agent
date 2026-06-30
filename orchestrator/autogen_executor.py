"""AutoGen Signal Executor — wires autogen_team debate → risk engine → execution.

Pipeline:
  1. AutoGen GroupChat produces raw trading signal (dict)
  2. autogen_to_langgraph() converts → LangGraph-compatible signal
  3. EnhancedRiskEngine validates (hard filter, ATR, Kelly, circuit breaker)
  4. SignalExecutor dispatches to PaperClient or BinanceClient
  5. Result logged + state updated

Usage:
    from orchestrator.autogen_executor import AutoGenSignalExecutor

    executor = AutoGenSignalExecutor(paper_mode=True)
    result = executor.execute_cycle(candle_data={...}, vlm_output={...})
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models import (
    AccountState,
    Candle,
    ExecutionResult,
    OrderSide,
    OrderType,
    Signal,
    SignalAction,
)
from execution import BinanceClient, OrderRequest, OrderResponse, PaperClient
from risk import RiskEngine

from orchestrator.autogen_team import TradingTeam, autogen_to_langgraph
from orchestrator.model_loader import ModelLoader

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────


@dataclass
class CycleResult:
    """Complete result of one trading cycle."""

    cycle_id: str = ""
    timestamp: str = ""
    raw_signal: dict = field(default_factory=dict)
    langgraph_signal: dict = field(default_factory=dict)
    risk_decision: dict = field(default_factory=dict)
    execution_result: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0
    autogen_latency_ms: float = 0.0
    risk_latency_ms: float = 0.0
    exec_latency_ms: float = 0.0
    status: str = "pending"  # pending | completed | rejected | error
    error: str = ""


# ── Signal conversion ──────────────────────────────────────


def _langgraph_signal_to_model_signal(lg_signal: dict, symbol: str = "BTCUSDT") -> Signal:
    """Convert langgraph signal dict → pydantic Signal model.

    The autogen_to_langgraph output has: action, confidence (0 if risk blocked),
    entry_reason (list of str), risk_notes (str).
    """
    action_str = lg_signal.get("action", "HOLD")
    confidence = lg_signal.get("confidence", 0.0)

    # Size calculation: confidence-based, capped at 5% equity
    size_pct = min(confidence * 0.05, 0.05) if action_str != "HOLD" else 0.0

    return Signal(
        action=action_str,
        confidence=confidence,
        size_pct=size_pct,
        reasoning="; ".join(lg_signal.get("entry_reason", [])),
        model_used="autogen_team",
    )


# ── Main executor ──────────────────────────────────────────


class AutoGenSignalExecutor:
    """Full pipeline: AutoGen debate → Risk → Execution.

    Args:
        paper_mode: True = PaperClient, False = BinanceClient (live!)
        config_path: Path to config.yaml for risk engine.
        initial_balance: Paper trading starting balance (ignored for live).
        max_round: AutoGen GroupChat rounds (6 per agents.md).
        model_warmup: Pre-warm Ollama models before first cycle.
    """

    def __init__(
        self,
        paper_mode: bool = True,
        config_path: str = "config.yaml",
        initial_balance: float = 10000.0,
        max_round: int = 6,
        model_warmup: bool = True,
    ):
        self.paper_mode = paper_mode
        self.config_path = config_path
        self.max_round = max_round
        self.model_warmup = model_warmup

        # Components
        self.team = TradingTeam(max_round=max_round)
        self.risk_engine = RiskEngine(config_path=config_path)
        self.model_loader = ModelLoader()

        # Execution client
        if paper_mode:
            self.executor = PaperClient(initial_balance=initial_balance)
        else:
            # BinanceClient requires async context — store config, init on first use
            self._binance_config: dict | None = None
            self.executor = None

        # Account state
        self.account = AccountState(
            equity=initial_balance,
            balance=initial_balance,
            open_positions=0,
            daily_pnl_pct=0.0,
            loss_streak=0,
            max_drawdown_pct=0.0,
        )

        # Cycle counter
        self._cycle_count = 0
        self._warmed_up = False

    # ── Warmup ──────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-warm all Ollama models into GPU cache (sequential)."""
        if self._warmed_up:
            return

        logger.info("Pre-warming Ollama models (sequential GPU load)...")
        from orchestrator.autogen_team import (
            MODEL_COORDINATOR,
            MODEL_MACRO_RISK,
            MODEL_PATTERN,
            MODEL_QUANT,
        )

        for model_name in [MODEL_QUANT, MODEL_PATTERN, MODEL_MACRO_RISK, MODEL_COORDINATOR]:
            t0 = time.perf_counter()
            try:
                self.model_loader.warmup_model(model_name)
                dt = (time.perf_counter() - t0) * 1000
                logger.info(f"  {model_name}: warmed in {dt:.0f}ms")
            except Exception as e:
                logger.warning(f"  {model_name}: warmup failed — {e}")

        self._warmed_up = True

    # ── Main cycle ──────────────────────────────────────

    def execute_cycle(
        self,
        candle_data: dict[str, Any],
        vlm_output: dict[str, Any] | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> CycleResult:
        """Execute one complete trading cycle.

        Returns CycleResult with full trace (autogen → risk → execution).
        """
        self._cycle_count += 1
        cycle_id = f"cycle_{self._cycle_count:04d}"
        result = CycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        try:
            # ── Step 1: AutoGen debate ────────────────────
            if self.model_warmup and not self._warmed_up:
                self.warmup()

            t0 = time.perf_counter()
            raw_signal = self.team.run_trading_cycle(
                candle_data=candle_data,
                vlm_output=vlm_output,
                market_context=market_context,
            )
            result.autogen_latency_ms = (time.perf_counter() - t0) * 1000
            result.raw_signal = raw_signal

            # ── Step 2: Convert to LangGraph signal ───────
            lg_signal = autogen_to_langgraph(raw_signal)
            result.langgraph_signal = lg_signal

            # ── Step 3: Risk engine (hard filter) ──────────
            model_signal = _langgraph_signal_to_model_signal(
                lg_signal, symbol=candle_data.get("symbol", "BTCUSDT")
            )

            t1 = time.perf_counter()
            risk_decision = self.risk_engine.validate(model_signal, self.account)
            result.risk_latency_ms = (time.perf_counter() - t1) * 1000
            result.risk_decision = {
                "allow": risk_decision.allow,
                "reason": risk_decision.reason,
                "max_size": risk_decision.max_size,
            }

            # ── Step 4: Execute if approved ────────────────
            if not risk_decision.allow:
                result.status = "rejected"
                result.execution_result = {"skipped": True, "reason": risk_decision.reason}
                logger.info(
                    f"[{cycle_id}] REJECTED by risk: {risk_decision.reason}"
                )
            else:
                t2 = time.perf_counter()
                exec_result = self._execute_signal(model_signal, risk_decision)
                result.exec_latency_ms = (time.perf_counter() - t2) * 1000
                result.execution_result = exec_result
                result.status = "completed"
                self._update_account(exec_result)

            result.total_latency_ms = (
                result.autogen_latency_ms
                + result.risk_latency_ms
                + result.exec_latency_ms
            )

        except Exception as e:
            result.status = "error"
            result.error = str(e)
            logger.error(f"[{cycle_id}] Cycle error: {e}", exc_info=True)

        return result

    # ── Execution dispatch ─────────────────────────────

    def _execute_signal(self, signal: Signal, risk_decision) -> dict:
        """Dispatch signal to paper or live execution client."""
        symbol = "BTCUSDT"  # could be dynamic
        side = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL
        quantity = self._calc_quantity(signal, risk_decision.max_size)

        exec_dict: dict = {
            "action": signal.action,
            "side": side.value,
            "quantity": quantity,
            "confidence": signal.confidence,
            "symbol": symbol,
        }

        if self.paper_mode:
            # Paper execution (synchronous)
            order_req = OrderRequest(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )
            if isinstance(self.executor, PaperClient):
                order_resp = self.executor.place_order(order_req)
                exec_dict.update({
                    "order_id": order_resp.order_id,
                    "filled_qty": order_resp.filled_qty,
                    "avg_price": order_resp.avg_price,
                    "status": order_resp.status.value,
                    "mode": "paper",
                })
        else:
            # Live Binance execution (would need async wrapper in production)
            exec_dict.update({
                "mode": "live",
                "status": "deferred",
                "note": "Use async execute_cycle_live() for Binance execution",
            })

        return exec_dict

    def _calc_quantity(self, signal: Signal, max_size: float) -> float:
        """Calculate order quantity from signal and risk limits."""
        equity = self.account.equity
        if equity <= 0 or signal.action == "HOLD":
            return 0.0

        notional = min(equity * signal.size_pct, max_size)
        # Assume price from model signal or reasonable default
        price = signal.entry_price or 60000.0  # BTC fallback
        if price <= 0:
            return 0.0

        qty = notional / price
        return round(qty, 4)

    def _update_account(self, exec_result: dict) -> None:
        """Update internal account state after execution."""
        if self.paper_mode and isinstance(self.executor, PaperClient):
            self.account.balance = self.executor.balance
            self.account.equity = self.executor.balance + sum(
                p.get("unrealized_pnl", 0) for p in self.executor.positions.values()
            )
            self.account.open_positions = len(
                [p for p in self.executor.positions.values() if p.get("size", 0) > 0]
            )

    # ── Async Binance execution ─────────────────────────

    async def execute_cycle_live(
        self,
        candle_data: dict[str, Any],
        vlm_output: dict[str, Any] | None = None,
        market_context: dict[str, Any] | None = None,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
    ) -> CycleResult:
        """Async cycle for live Binance execution."""
        # Run sync parts in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.execute_cycle(candle_data, vlm_output, market_context),
        )

        # If approved and live mode, execute via BinanceClient
        if result.status == "completed" and not self.paper_mode:
            async with BinanceClient(api_key, api_secret, testnet=testnet) as client:
                signal = _langgraph_signal_to_model_signal(
                    result.langgraph_signal,
                    symbol=candle_data.get("symbol", "BTCUSDT"),
                )
                side = OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL
                qty = self._calc_quantity(signal, result.risk_decision.get("max_size", 0))
                order_req = OrderRequest(
                    symbol=candle_data.get("symbol", "BTCUSDT"),
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=qty,
                    price=candle_data.get("price"),
                    post_only=True,
                )
                try:
                    order_resp = await client.place_order(order_req)
                    result.execution_result.update({
                        "order_id": order_resp.order_id,
                        "status": order_resp.status.value,
                        "mode": "live",
                    })
                except Exception as e:
                    result.execution_result.update({"mode": "live", "error": str(e)})
                    result.status = "error"

        return result

    # ── Status ───────────────────────────────────────────

    def get_status(self) -> dict:
        """Current executor state."""
        return {
            "cycle_count": self._cycle_count,
            "paper_mode": self.paper_mode,
            "account": {
                "equity": self.account.equity,
                "balance": self.account.balance,
                "open_positions": self.account.open_positions,
                "daily_pnl_pct": self.account.daily_pnl_pct,
                "loss_streak": self.account.loss_streak,
            },
            "risk": self.risk_engine.get_status(),
            "warmed_up": self._warmed_up,
        }
