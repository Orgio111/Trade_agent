"""
Reward Function Module — computes RL rewards from trade outcomes.

Core concept: Every trade becomes training data. Reward = f(pnl, risk, confidence, regime).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Import trade outcome from memory store
import sys
sys.path.append(r"C:\Users\Mns Mns\OneDrive\Documents\GitHub\Trade_agent")
from orchestrator.rl_memory.memory_store import TradeOutcome


# ── Reward Components ──────────────────────────────────────────


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward computation."""

    pnl_reward: float = 0.0
    risk_penalty: float = 0.0
    confidence_bonus: float = 0.0
    regime_bonus: float = 0.0
    latency_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pnl_reward": self.pnl_reward,
            "risk_penalty": self.risk_penalty,
            "confidence_bonus": self.confidence_bonus,
            "regime_bonus": self.regime_bonus,
            "latency_penalty": self.latency_penalty,
            "total": self.total,
        }


# ── Reward Configuration ───────────────────────────────────────


@dataclass
class RewardConfig:
    """Configurable weights for reward function."""

    # PnL weights
    pnl_weight: float = 1.0           # Base PnL multiplier
    pnl_cap: float = 5.0              # Max absolute PnL reward

    # Risk weights
    drawdown_weight: float = 0.7      # Max drawdown penalty
    risk_penalty_cap: float = 3.0     # Max risk penalty

    # Confidence
    confidence_weight: float = 0.2    # Bonus for high confidence
    confidence_threshold: float = 0.7 # Minimum for bonus

    # Regime alignment
    regime_bonus: float = 0.1         # Small bonus for correct regime

    # Latency
    latency_penalty_per_100ms: float = 0.01
    max_latency_penalty: float = 0.1

    # Trade quality
    min_trade_size_pct: float = 0.001  # Ignore dust trades
    hold_penalty: float = -0.01        # Small penalty for HOLD (no action)

    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


# ── Main Reward Function ───────────────────────────────────────


def compute_reward(
    trade: TradeOutcome,
    config: RewardConfig | None = None,
    market_context: dict | None = None,
) -> tuple[float, RewardBreakdown]:
    """
    Compute RL reward from trade outcome.

    Args:
        trade: TradeOutcome with PnL, confidence, indicators, etc.
        config: RewardConfig with weights (uses defaults if None)
        market_context: Optional dict with regime, volatility, etc.

    Returns:
        (total_reward, RewardBreakdown)
    """
    config = config or RewardConfig()
    market_context = market_context or {}
    breakdown = RewardBreakdown()

    # Skip reward for dust trades
    if abs(trade.pnl_pct) < config.min_trade_size_pct and trade.action != "HOLD":
        return 0.0, breakdown

    # ── 1. PnL Reward (primary signal) ─────────────────────
    # Scale PnL to reward space: 1% PnL ≈ 1.0 reward
    raw_pnl_reward = trade.pnl_pct * 100  # 1% = 1.0
    breakdown.pnl_reward = max(min(raw_pnl_reward * config.pnl_weight, config.pnl_cap), -config.pnl_cap)

    # ── 2. Risk Penalty (max drawdown during trade) ────────
    if trade.max_drawdown_pct > 0:
        # Penalize trades that had large adverse excursion
        drawdown_pct = trade.max_drawdown_pct * 100  # Convert to %
        breakdown.risk_penalty = -min(drawdown_pct * config.drawdown_weight, config.risk_penalty_cap)
    else:
        breakdown.risk_penalty = 0.0

    # ── 3. Confidence Bonus ─────────────────────────────────
    if trade.confidence >= config.confidence_threshold:
        # Bonus scales with confidence above threshold
        bonus = (trade.confidence - config.confidence_threshold) * config.confidence_weight * 10
        breakdown.confidence_bonus = min(bonus, 0.5)  # Cap at 0.5
    else:
        # Small penalty for low confidence trades that lost
        if trade.pnl_pct < 0 and trade.confidence < 0.5:
            breakdown.confidence_bonus = -0.1

    # ── 4. Regime Alignment Bonus ───────────────────────────
    # Reward trades that align with market regime
    # Skip for HOLD (non-action)
    if trade.action != "HOLD":
        expected_action = _expected_action_for_regime(trade.regime)
        if expected_action and trade.action == expected_action:
            breakdown.regime_bonus = config.regime_bonus
        elif expected_action and trade.action != expected_action:
            breakdown.regime_bonus = -config.regime_bonus  # Penalty for counter-regime
    else:
        breakdown.regime_bonus = 0.0

    # ── 5. Latency Penalty ──────────────────────────────────
    latency_penalty = (trade.execution_latency_ms / 100.0) * config.latency_penalty_per_100ms
    breakdown.latency_penalty = -min(latency_penalty, config.max_latency_penalty)

    # ── 6. HOLD Penalty ─────────────────────────────────────
    if trade.action == "HOLD":
        breakdown.hold_penalty = config.hold_penalty
    else:
        breakdown.hold_penalty = 0.0

    # ── Total ───────────────────────────────────────────────
    breakdown.total = (
        breakdown.pnl_reward
        + breakdown.risk_penalty
        + breakdown.confidence_bonus
        + breakdown.regime_bonus
        + breakdown.latency_penalty
        + breakdown.hold_penalty
    )

    # Clamp total to reasonable range
    breakdown.total = max(min(breakdown.total, 10.0), -10.0)

    return breakdown.total, breakdown


def _expected_action_for_regime(regime: str) -> str | None:
    """Map market regime to expected directional bias."""
    mapping = {
        "trending_up": "BUY",
        "trending_down": "SELL",
        "ranging": "HOLD",
        "high_volatility": "HOLD",
        "low_volatility": "HOLD",
        "breakout_up": "BUY",
        "breakout_down": "SELL",
    }
    return mapping.get(regime)


# ── Batch Reward Computation ──────────────────────────────────


def compute_batch_rewards(
    trades: list[TradeOutcome],
    config: RewardConfig | None = None,
) -> list[tuple[float, RewardBreakdown]]:
    """Compute rewards for multiple trades."""
    return [compute_reward(t, config) for t in trades]


def get_reward_stats(
    trades: list[TradeOutcome],
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    """Get aggregate statistics on rewards."""
    rewards = [compute_reward(t, config)[0] for t in trades]
    if not rewards:
        return {"count": 0}

    return {
        "count": len(rewards),
        "mean": sum(rewards) / len(rewards),
        "std": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5,
        "min": min(rewards),
        "max": max(rewards),
        "positive_count": sum(1 for r in rewards if r > 0),
        "negative_count": sum(1 for r in rewards if r < 0),
        "sharpe": (sum(rewards) / len(rewards)) / (max(0.001, (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5)),
    }


# ── Policy Update Prompt Generation ────────────────────────────


def generate_policy_update_prompt(
    winning_trades: list[TradeOutcome],
    losing_trades: list[TradeOutcome],
    config: RewardConfig | None = None,
) -> str:
    """
    Generate prompt for LLM to improve trading policy.

    Args:
        winning_trades: List of profitable trades
        losing_trades: List of losing trades
        config: Reward config for context

    Returns:
        Prompt string for LLM
    """
    config = config or RewardConfig()

    def format_trade(t: TradeOutcome) -> str:
        r, bd = compute_reward(t, config)
        return (
            f"  {t.action} {t.symbol} @ {t.entry_price:.2f} → {t.exit_price:.2f} "
            f"| PnL: {t.pnl_pct*100:.2f}% | Conf: {t.confidence:.2f} | Regime: {t.regime} "
            f"| Reward: {r:.3f} | Reason: {t.agent_reasoning[:2] if t.agent_reasoning else 'N/A'}"
        )

    wins_text = "\n".join(format_trade(t) for t in winning_trades[:10]) or "  (none)"
    losses_text = "\n".join(format_trade(t) for t in losing_trades[:10]) or "  (none)"

    return f"""You are a trading strategy optimizer. Analyze recent trade outcomes and suggest improvements.

## Winning Trades (Reward > 0):
{wins_text}

## Losing Trades (Reward ≤ 0):
{losses_text}

## Task:
Identify patterns in wins vs losses. Output improved trading rules in this format:

```json
{{
  "entry_rules": [
    "Rule 1: ...",
    "Rule 2: ..."
  ],
  "exit_rules": [
    "Rule 1: ..."
  ],
  "risk_rules": [
    "Rule 1: ..."
  ],
  "regime_filters": [
    "Only trade BUY in trending_up",
    "Avoid SELL in ranging"
  ],
  "confidence_thresholds": {{
    "BUY": 0.75,
    "SELL": 0.75
  }},
  "notes": "Key observations..."
}}
```

Focus on actionable, specific rules. No generic advice."""


# ── Export ────────────────────────────────────────────────────


__all__ = [
    "RewardConfig",
    "RewardBreakdown",
    "compute_reward",
    "compute_batch_rewards",
    "get_reward_stats",
    "generate_policy_update_prompt",
]