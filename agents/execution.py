"""
Execution Agent: PPO-based RL agent optimizes order timing to minimize slippage.
Falls back to async TWAP/VWAP when the RL model is not loaded.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Protocol

import numpy as np

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import (
    CouncilDecision,
    Order,
    OrderStatus,
    OrderType,
    RiskReport,
    Side,
)
from core.observability import AGENT_LATENCY, ORDER_COUNTER, SLIPPAGE_HIST

logger = logging.getLogger(__name__)


class PPOPolicy(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]: ...


def _build_obs(
    decision: CouncilDecision,
    risk: RiskReport,
    recent_prices: np.ndarray,
) -> np.ndarray:
    """Construct normalized observation vector for the PPO agent."""
    price_returns = np.diff(recent_prices[-30:]) / (recent_prices[-30:-1] + 1e-10) if len(recent_prices) >= 31 else np.zeros(29)
    features = [
        decision.consensus_score,
        decision.bull_score,
        decision.bear_score,
        risk.var_99,
        risk.kelly_fractional,
        float(decision.final_side == Side.BUY),
        float(decision.technical.rsi_14 / 100.0) if decision.technical else 0.5,
        float(decision.technical.volume_ratio / 5.0) if decision.technical else 0.2,
        float(decision.sentiment.news_score) if decision.sentiment else 0.0,
        float(decision.sentiment.fear_greed_index / 100.0) if decision.sentiment else 0.5,
    ]
    obs = np.concatenate([np.array(features, dtype=np.float32), price_returns.astype(np.float32)])
    # Pad or truncate to fixed size 40
    if len(obs) < 40:
        obs = np.pad(obs, (0, 40 - len(obs)))
    else:
        obs = obs[:40]
    return obs


class ExecutionAgent:
    """
    Wraps a PPO model (stable-baselines3) for optimal execution timing.
    Sends orders to the exchange via ccxt (paper or live).
    """

    def __init__(self) -> None:
        self._ppo: PPOPolicy | None = None
        self._load_ppo()

    def _load_ppo(self) -> None:
        cfg = get_settings()
        if os.path.exists(cfg.ppo_model_path):
            try:
                from stable_baselines3 import PPO  # type: ignore[import]
                self._ppo = PPO.load(cfg.ppo_model_path)
                logger.info("PPO execution model loaded from %s", cfg.ppo_model_path)
            except Exception as exc:
                logger.warning("Failed to load PPO model: %s — using TWAP fallback", exc)

    async def execute(
        self,
        decision: CouncilDecision,
        risk: RiskReport,
        recent_prices: np.ndarray,
    ) -> Order:
        cfg = get_settings()

        with AGENT_LATENCY.labels(agent="execution").time():
            order_type, slice_count = self._decide_algo(decision, risk, recent_prices)

            order = Order(
                session_id=decision.session_id,
                symbol=decision.symbol,
                side=decision.final_side,
                order_type=order_type,
                quantity=risk.position_size_units,
                stop_price=risk.stop_loss_price,
                supervisor_rationale=decision.rationale,
            )

            if cfg.paper_trading:
                order = await self._paper_fill(order, recent_prices, slice_count)
            else:
                order = await self._live_fill(order, slice_count)

        bus = await get_bus()
        await bus.publish(cfg.stream_orders, MsgType.ORDER, order.model_dump(mode="json"))
        ORDER_COUNTER.labels(
            symbol=decision.symbol, side=decision.final_side.value, status=order.status.value
        ).inc()
        if order.slippage_bps is not None:
            SLIPPAGE_HIST.labels(symbol=decision.symbol).observe(order.slippage_bps)

        logger.info(
            "Execution %s %s %.6f @ %.4f  slippage=%.1f bps  [%s]",
            order.side.value,
            order.symbol,
            order.quantity,
            order.avg_fill_price or 0,
            order.slippage_bps or 0,
            order.order_type.value,
        )
        return order

    def _decide_algo(
        self, decision: CouncilDecision, risk: RiskReport, prices: np.ndarray
    ) -> tuple[OrderType, int]:
        if self._ppo is not None:
            obs = _build_obs(decision, risk, prices)
            action, _ = self._ppo.predict(obs, deterministic=True)
            # Action 0→MARKET, 1→TWAP/3, 2→TWAP/5, 3→VWAP/5
            algo_map = {
                0: (OrderType.MARKET, 1),
                1: (OrderType.TWAP, 3),
                2: (OrderType.TWAP, 5),
                3: (OrderType.VWAP, 5),
            }
            return algo_map.get(int(action[0] if hasattr(action, "__len__") else action), (OrderType.TWAP, 3))

        # Fallback heuristic: large orders → TWAP, small → MARKET
        size_usd = risk.position_size_usd
        if size_usd > 10_000:
            return OrderType.TWAP, 5
        return OrderType.MARKET, 1

    async def _paper_fill(self, order: Order, prices: np.ndarray, slices: int) -> Order:
        cfg = get_settings()
        mid_price = float(prices[-1]) if len(prices) > 0 else 1.0

        # Simulate TWAP slicing
        fill_prices = []
        for i in range(slices):
            noise = mid_price * np.random.normal(0, 0.0002)
            spread = mid_price * (cfg.max_slippage_bps / 10000) * 0.5
            fill = mid_price + noise + (spread if order.side == Side.BUY else -spread)
            fill_prices.append(max(fill, 0.01))
            if slices > 1:
                await asyncio.sleep(0.05)  # simulate time between slices

        avg_fill = float(np.mean(fill_prices))
        slippage_bps = abs(avg_fill - mid_price) / mid_price * 10_000

        order.avg_fill_price = avg_fill
        order.slippage_bps = slippage_bps
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.utcnow()
        return order

    async def _live_fill(self, order: Order, slices: int) -> Order:
        """Real exchange order submission via ccxt."""
        import ccxt.async_support as ccxt  # type: ignore[import]
        cfg = get_settings()
        ExchangeClass = getattr(ccxt, cfg.exchange)
        exchange = ExchangeClass()
        try:
            slice_qty = order.quantity / slices
            fills = []
            for _ in range(slices):
                result = await asyncio.wait_for(
                    exchange.create_order(
                        order.symbol,
                        "market",
                        order.side.value.lower(),
                        slice_qty,
                    ),
                    timeout=cfg.order_timeout_s,
                )
                fills.append(result["average"] or result["price"])
                if slices > 1:
                    await asyncio.sleep(1.0)
            order.avg_fill_price = float(np.mean(fills))
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
        except Exception as exc:
            logger.error("Live order failed: %s", exc)
            order.status = OrderStatus.REJECTED
        finally:
            await exchange.close()
        return order
