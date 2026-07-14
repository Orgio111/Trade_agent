"""
Streaming Reward Engine — ms-level reward computation from trade execution events.

Core concept: Every trade execution triggers immediate reward computation
and pushes to online RL buffer for policy update.

Architecture:
  Trade Executed → RewardEvent → StreamingRewardEngine → Online Policy Update
                      ↓
              RL Buffer (ring buffer, max 10k samples)
                      ↓
              Light PPO / Bandit Update (async, non-blocking)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Deque, Optional

import numpy as np

# Import existing trade outcome and memory
from orchestrator.rl_memory.memory_store import MarketState, TradeOutcome
from orchestrator.rl_memory.reward_function import RewardBreakdown, RewardConfig, compute_reward


class RewardEventType(str, Enum):
    """Type of reward event."""

    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    TRADE_UPDATE = "trade_update"  # Partial fill, trailing stop update
    POSITION_ADJUST = "position_adjust"  # Size increase/decrease
    RISK_EVENT = "risk_event"  # Stop loss hit, margin call, etc.


@dataclass
class RewardEvent:
    """Single reward event from trading system."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: RewardEventType = RewardEventType.TRADE_CLOSE
    timestamp: datetime = field(default_factory=datetime.now)

    # Trade identification
    trade_id: str = ""
    symbol: str = ""

    # Execution data
    action: str = ""  # BUY/SELL/HOLD
    entry_price: float = 0.0
    exit_price: float | None = None
    quantity: float = 0.0
    filled_qty: float = 0.0
    avg_price: float = 0.0

    # PnL
    unrealized_pnl_pct: float = 0.0
    realized_pnl_pct: float = 0.0
    unrealized_pnl_abs: float = 0.0
    realized_pnl_abs: float = 0.0

    # Risk context
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    position_size_pct: float = 0.0
    leverage: float = 1.0

    # Agent context
    confidence: float = 0.5
    regime: str = "unknown"
    indicators: dict[str, float] = field(default_factory=dict)
    agent_reasoning: list[str] = field(default_factory=list)
    risk_decision: dict[str, Any] = field(default_factory=dict)

    # Execution quality
    execution_latency_ms: float = 0.0
    slippage_bps: float = 0.0
    fees_bps: float = 0.0

    # Status
    status: str = "OPEN"  # OPEN, CLOSED, PARTIAL, REJECTED

    def to_trade_outcome(self) -> TradeOutcome:
        """Convert to TradeOutcome for memory storage."""
        pnl_pct = self.realized_pnl_pct if self.event_type == RewardEventType.TRADE_CLOSE else self.unrealized_pnl_pct
        pnl_abs = self.realized_pnl_abs if self.event_type == RewardEventType.TRADE_CLOSE else self.unrealized_pnl_abs

        return TradeOutcome(
            trade_id=self.trade_id or self.event_id,
            timestamp=self.timestamp,
            symbol=self.symbol,
            action=self.action,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            quantity=self.quantity,
            pnl_pct=pnl_pct,
            pnl_abs=pnl_abs,
            confidence=self.confidence,
            regime=self.regime,
            indicators=self.indicators,
            agent_reasoning=self.agent_reasoning,
            risk_decision=self.risk_decision,
            execution_latency_ms=self.execution_latency_ms,
            status=self.status,
            max_drawdown_pct=self.max_drawdown_pct,
        )


@dataclass
class StreamingRewardConfig:
    """Configuration for streaming reward engine."""

    # Reward computation
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Buffer settings
    buffer_max_size: int = 10000
    min_buffer_for_update: int = 32
    update_batch_size: int = 64

    # Update frequency
    update_interval_ms: int = 100  # Policy update every N ms
    max_updates_per_second: int = 50

    # Reward shaping
    enable_streaming_shaping: bool = True
    shaping_gamma: float = 0.99  # Discount for future rewards
    shaping_lambda: float = 0.95  # GAE lambda

    # Online learning
    learning_rate: float = 0.01
    baseline_decay: float = 0.99  # Exponential moving average baseline

    # Risk-aware scaling
    risk_scaling: bool = True
    max_risk_multiplier: float = 2.0

    # Async processing
    async_updates: bool = True
    update_timeout_ms: int = 50


class StreamingRewardEngine:
    """
    Real-time reward computation engine.

    Processes trade execution events → computes rewards → pushes to RL buffer
    → triggers online policy updates at ms-level latency.
    """

    def __init__(
        self,
        config: StreamingRewardConfig | None = None,
        policy_update_callback: Callable[[list[RewardEvent], list[float]], None] | None = None,
        memory_manager=None,
    ):
        self.config = config or StreamingRewardConfig()
        self.policy_update_callback = policy_update_callback
        self.memory_manager = memory_manager

        # Ring buffer for streaming rewards
        self._reward_buffer: Deque[tuple[RewardEvent, float, RewardBreakdown]] = deque(
            maxlen=self.config.buffer_max_size
        )

        # Baseline for advantage computation (EMA)
        self._reward_baseline: float = 0.0
        self._baseline_initialized: bool = False

        # Update throttling
        self._last_update_time: float = 0.0
        self._updates_this_second: int = 0
        self._second_window_start: float = time.time()

        # Stats
        self.stats = {
            "events_processed": 0,
            "rewards_computed": 0,
            "policy_updates_triggered": 0,
            "avg_reward": 0.0,
            "avg_latency_ms": 0.0,
        }

    def process_event(self, event: RewardEvent) -> tuple[float, RewardBreakdown]:
        """
        Process a single reward event, compute reward, buffer it.

        Returns: (reward, breakdown)
        """
        start_time = time.perf_counter()

        # Convert to TradeOutcome for reward computation
        trade = event.to_trade_outcome()

        # Compute reward using existing reward function
        reward, breakdown = compute_reward(trade, self.config.reward_config)

        # Apply streaming reward shaping if enabled
        if self.config.enable_streaming_shaping and event.event_type != RewardEventType.TRADE_CLOSE:
            reward = self._apply_streaming_shaping(event, reward, breakdown)

        # Update baseline (EMA)
        self._update_baseline(reward)

        # Compute advantage
        advantage = reward - self._reward_baseline

        # Buffer the event
        self._reward_buffer.append((event, advantage, breakdown))

        # Update stats
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._update_stats(reward, latency_ms)

        # Trigger policy update if conditions met
        self._maybe_trigger_update()

        return reward, breakdown

    def _apply_streaming_shaping(
        self, event: RewardEvent, base_reward: float, breakdown: RewardBreakdown
    ) -> float:
        """Apply temporal difference shaping for open positions."""
        # For open positions, add potential future reward estimate
        if event.event_type == RewardEventType.TRADE_OPEN:
            # Small positive shaping for taking action aligned with signal
            if event.action in ("BUY", "SELL") and event.confidence > 0.6:
                return base_reward + 0.1 * event.confidence
        elif event.event_type == RewardEventType.TRADE_UPDATE:
            # Shape based on unrealized PnL trajectory
            if event.unrealized_pnl_pct > 0:
                return base_reward + 0.05 * min(event.unrealized_pnl_pct * 100, 1.0)

        return base_reward

    def _update_baseline(self, reward: float):
        """Update exponential moving average baseline."""
        if not self._baseline_initialized:
            self._reward_baseline = reward
            self._baseline_initialized = True
        else:
            self._reward_baseline = (
                self.config.baseline_decay * self._reward_baseline + (1 - self.config.baseline_decay) * reward
            )

    def _update_stats(self, reward: float, latency_ms: float):
        """Update running statistics."""
        self.stats["events_processed"] += 1
        self.stats["rewards_computed"] += 1

        # Running average
        n = self.stats["rewards_computed"]
        self.stats["avg_reward"] = ((n - 1) * self.stats["avg_reward"] + reward) / n
        self.stats["avg_latency_ms"] = ((n - 1) * self.stats["avg_latency_ms"] + latency_ms) / n

    def _maybe_trigger_update(self):
        """Check if policy update should be triggered."""
        now = time.time()

        # Throttle updates per second
        if now - self._second_window_start >= 1.0:
            self._updates_this_second = 0
            self._second_window_start = now

        if self._updates_this_second >= self.config.max_updates_per_second:
            return

        # Minimum interval between updates
        if now - self._last_update_time < self.config.update_interval_ms / 1000:
            return

        # Minimum buffer size
        if len(self._reward_buffer) < self.config.min_buffer_for_update:
            return

        # Trigger update
        self._trigger_policy_update()
        self._last_update_time = now
        self._updates_this_second += 1

    def _trigger_policy_update(self):
        """Sample batch from buffer and call policy update callback."""
        if not self.policy_update_callback or len(self._reward_buffer) < self.config.update_batch_size:
            return

        # Sample recent batch (prioritize recent)
        batch_size = min(self.config.update_batch_size, len(self._reward_buffer))
        indices = np.random.choice(len(self._reward_buffer), batch_size, replace=False)
        batch = [self._reward_buffer[i] for i in indices]

        events = [item[0] for item in batch]
        advantages = [item[1] for item in batch]

        try:
            if self.config.async_updates:
                # Fire and forget
                asyncio.create_task(self._async_policy_update(events, advantages))
            else:
                self.policy_update_callback(events, advantages)
            self.stats["policy_updates_triggered"] += 1
        except Exception as e:
            # Log but don't crash the reward engine
            print(f"⚠️ Policy update callback failed: {e}")

    async def _async_policy_update(self, events: list[RewardEvent], advantages: list[float]):
        """Async policy update with timeout."""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.policy_update_callback, events, advantages),
                timeout=self.config.update_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            print(f"⚠️ Policy update timed out after {self.config.update_timeout_ms}ms")
        except Exception as e:
            print(f"⚠️ Async policy update failed: {e}")

    def get_buffer_stats(self) -> dict:
        """Get current buffer statistics."""
        if not self._reward_buffer:
            return {"size": 0}

        rewards = [item[1] for item in self._reward_buffer]
        return {
            "size": len(self._reward_buffer),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "baseline": self._reward_baseline,
        }

    def get_recent_events(self, limit: int = 100) -> list[RewardEvent]:
        """Get most recent events from buffer."""
        return [item[0] for item in list(self._reward_buffer)[-limit:]]

    def clear_buffer(self):
        """Clear the reward buffer."""
        self._reward_buffer.clear()
        self._baseline_initialized = False
        self._reward_baseline = 0.0


def compute_streaming_reward(
    event: RewardEvent,
    config: StreamingRewardConfig | None = None,
) -> tuple[float, RewardBreakdown]:
    """
    Standalone function to compute streaming reward from event.

    Useful for testing or direct integration without full engine.
    """
    config = config or StreamingRewardConfig()
    trade = event.to_trade_outcome()
    reward, breakdown = compute_reward(trade, config.reward_config)

    if config.enable_streaming_shaping and event.event_type != RewardEventType.TRADE_CLOSE:
        reward = _apply_streaming_shaping_static(event, reward, breakdown)

    return reward, breakdown


def _apply_streaming_shaping_static(
    event: RewardEvent, base_reward: float, breakdown: RewardBreakdown
) -> float:
    """Static version of streaming shaping."""
    if event.event_type == RewardEventType.TRADE_OPEN:
        if event.action in ("BUY", "SELL") and event.confidence > 0.6:
            return base_reward + 0.1 * event.confidence
    elif event.event_type == RewardEventType.TRADE_UPDATE:
        if event.unrealized_pnl_pct > 0:
            return base_reward + 0.05 * min(event.unrealized_pnl_pct * 100, 1.0)
    return base_reward


# Factory function
def create_streaming_reward_engine(
    config: StreamingRewardConfig | None = None,
    policy_update_callback: Callable[[list[RewardEvent], list[float]], None] | None = None,
    memory_manager=None,
) -> StreamingRewardEngine:
    """Factory for creating streaming reward engine."""
    return StreamingRewardEngine(config, policy_update_callback, memory_manager)