"""Brain #6: FinRL PPO Position Sizer via Kelly Criterion.

Uses reinforcement learning (PPO) to determine optimal position sizing.
Falls back to Kelly Criterion from win rate + avg win/loss ratio.
Weight in Go orchestrator: 0.10.
"""

from __future__ import annotations

import logging
import os
from collections import deque

import numpy as np

from .base_brain import BaseBrain, BrainSignal

logger = logging.getLogger(__name__)


class FinRLBrain(BaseBrain):
    """FinRL PPO + Kelly Criterion position sizing brain.

    Primary: Trained PPO agent for position size recommendation.
    Fallback: Kelly Criterion = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss.

    This brain's score represents the SIZING conviction, not direction.
    It amplifies or dampens the directional consensus from other brains.
    """

    @property
    def brain_id(self) -> str:
        return "finrl_kelly"

    def __init__(self) -> None:
        self._ppo_agent = None
        self._trade_history: deque[dict] = deque(maxlen=500)
        self._max_risk_pct = float(os.getenv("MAX_RISK_PCT", "0.015"))  # 1.5%
        self.risk_pct = float(os.getenv("RISK_PCT_PER_TRADE", "0.015"))

    async def warmup(self) -> None:
        """Attempt to load trained PPO agent."""
        model_path = os.getenv("FINRL_MODEL_PATH", "")
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import PPO
                self._ppo_agent = PPO.load(model_path)
                logger.info("[finrl_kelly] PPO agent loaded from %s", model_path)
            except ImportError:
                logger.warning("[finrl_kelly] stable_baselines3 not installed, using Kelly only")

    async def compute_score(self, symbol: str) -> BrainSignal:
        """Compute position sizing recommendation.

        Score interpretation:
          Positive (0..1) → supports LONG, magnitude = sizing fraction
          Negative (-1..0) → supports SHORT, magnitude = sizing fraction
          Near 0 → reduce position / no edge
        """
        kelly_frac = self._compute_kelly()
        confidence = self._kelly_confidence()

        # If PPO agent available, blend with Kelly
        if self._ppo_agent is not None:
            try:
                obs = self._build_observation(symbol)
                action, _ = self._ppo_agent.predict(obs, deterministic=True)
                ppo_size = float(np.clip(action, -1.0, 1.0))
                # Blend: 50% PPO, 50% Kelly
                combined = 0.5 * ppo_size + 0.5 * kelly_frac
                confidence = min(0.95, confidence + 0.1)
                return BrainSignal(
                    brain_id=self.brain_id,
                    symbol=symbol,
                    score=float(np.clip(combined, -1.0, 1.0)),
                    confidence=confidence,
                    metadata={"kelly_frac": round(kelly_frac, 4), "ppo_size": round(ppo_size, 4)},
                )
            except Exception as e:
                logger.warning("[finrl_kelly] PPO predict failed: %s", e)

        # Pure Kelly
        return BrainSignal(
            brain_id=self.brain_id,
            symbol=symbol,
            score=float(np.clip(kelly_frac, -1.0, 1.0)),
            confidence=confidence,
            metadata={"kelly_frac": round(kelly_frac, 4), "method": "kelly_only"},
        )

    def push_trade_result(self, pnl_pct: float) -> None:
        """Record trade P&L for Kelly calculation."""
        self._trade_history.append({
            "pnl_pct": pnl_pct,
            "win": pnl_pct > 0,
        })

    def _compute_kelly(self) -> float:
        """Compute Kelly Criterion fraction from trade history."""
        if len(self._trade_history) < 20:
            return 0.0  # insufficient data → flat

        wins = [t["pnl_pct"] for t in self._trade_history if t["win"]]
        losses = [abs(t["pnl_pct"]) for t in self._trade_history if not t["win"]]

        if not wins or not losses:
            return 0.0

        p = len(wins) / len(self._trade_history)  # win rate
        q = 1.0 - p
        avg_win = float(np.mean(wins))
        avg_loss = float(np.mean(losses))

        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss  # reward/risk ratio
        kelly = (p * b - q) / b  # Kelly fraction

        # Apply half-Kelly for safety (standard practice)
        half_kelly = kelly * 0.5

        # Cap at max risk
        return float(np.clip(half_kelly, -self._max_risk_pct / self.risk_pct, self._max_risk_pct / self.risk_pct))

    def _kelly_confidence(self) -> float:
        """Confidence based on number of trades and win rate stability."""
        n = len(self._trade_history)
        if n < 20:
            return 0.15
        if n < 50:
            return 0.4
        if n < 100:
            return 0.6
        return 0.8

    def _build_observation(self, symbol: str) -> np.ndarray:
        """Build observation vector for PPO agent."""
        # Simple state: [kelly_frac, win_rate, avg_win, avg_loss, trade_count_normalized]
        kelly = self._compute_kelly()
        wins = [t for t in self._trade_history if t["win"]]
        win_rate = len(wins) / max(1, len(self._trade_history))
        avg_win = float(np.mean([t["pnl_pct"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([abs(t["pnl_pct"]) for t in self._trade_history if not t["win"]])) if len(self._trade_history) > len(wins) else 0.01
        count_norm = min(1.0, len(self._trade_history) / 500)

        return np.array([kelly, win_rate, avg_win, avg_loss, count_norm], dtype=np.float32)
