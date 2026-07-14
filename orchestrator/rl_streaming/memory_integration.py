"""
Streaming Memory Integration — Unified interface for all three memory stores.

Extends existing RLMemoryManager with streaming-specific functionality:
- Ring buffers for recent rewards/advantages
- Vector similarity search for meta-learning context
- Trade outcome persistence with streaming rewards
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np

from orchestrator.rl_memory.memory_store import (
    MarketState,
    RLMemoryManager,
    TradeOutcome,
    create_memory_manager,
)
from orchestrator.rl_streaming.streaming_reward import (
    RewardEvent,
    RewardEventType,
    StreamingRewardConfig,
    StreamingRewardEngine,
    compute_streaming_reward,
)
from orchestrator.rl_streaming.online_policy import OnlinePolicy, OnlinePolicyConfig, create_online_policy

logger = logging.getLogger(__name__)


@dataclass
class StreamingMemoryConfig:
    """Configuration for streaming memory integration."""

    # Base memory config (passed to RLMemoryManager)
    redis_url: str | None = None
    postgres_dsn: str | None = None
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "trade_memory_streaming"

    # Streaming reward config
    streaming_reward_config: StreamingRewardConfig | None = None

    # Online policy config
    online_policy_config: OnlinePolicyConfig | None = None

    # Buffer sizes
    recent_rewards_buffer: int = 10000
    recent_states_buffer: int = 5000
    advantage_buffer: int = 10000

    # Vector search
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dim: int = 384
    similarity_threshold: float = 0.7
    max_similar_results: int = 10

    # Persistence
    save_interval_seconds: int = 60
    flush_on_shutdown: bool = True


class StreamingMemoryManager:
    """
    Unified memory manager for streaming RL + meta-learning.

    Combines:
    - RLMemoryManager (Redis/Postgres/Qdrant)
    - StreamingRewardEngine (real-time rewards)
    - OnlinePolicy (light PPO/bandit updates)
    - Ring buffers for recent data
    """

    def __init__(self, config: StreamingMemoryConfig | None = None):
        self.config = config or StreamingMemoryConfig()

        # Initialize base memory manager
        self.memory_manager = create_memory_manager({
            "redis_url": self.config.redis_url,
            "postgres_dsn": self.config.postgres_dsn,
            "qdrant_url": self.config.qdrant_url,
            "qdrant_api_key": self.config.qdrant_api_key,
            "collection": self.config.qdrant_collection,
        })

        # Streaming reward engine
        self.reward_engine = StreamingRewardEngine(
            config=self.config.streaming_reward_config or StreamingRewardConfig(),
            policy_update_callback=self._on_policy_update,
            memory_manager=self.memory_manager,
        )

        # Online policy
        self.online_policy = create_online_policy(self.config.online_policy_config or OnlinePolicyConfig())

        # Ring buffers for streaming data
        self._recent_rewards: deque = deque(maxlen=self.config.recent_rewards_buffer)
        self._recent_states: deque = deque(maxlen=self.config.recent_states_buffer)
        self._advantages: deque = deque(maxlen=self.config.advantage_buffer)
        self._reward_events: deque = deque(maxlen=self.config.recent_rewards_buffer)

        # State tracking
        self._last_save_time = time.time()
        self._total_events = 0
        self._total_updates = 0

        # Callback for external policy updates
        self.external_update_callback: Callable[[dict], None] | None = None

        logger.info("[StreamingMemoryManager] Initialized")

    def _on_policy_update(self, events: list[RewardEvent], advantages: list[float]):
        """Internal callback when reward engine triggers policy update."""
        # Convert events to state vectors for online policy
        for event, advantage in zip(events, advantages):
            state_vector = self._event_to_state_vector(event)
            if state_vector is not None:
                # Map event action to policy action
                action_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
                action = action_map.get(event.action, 0)

                # Update online policy
                self.online_policy.update(state_vector, action, advantage)

        self._total_updates += 1

        # Call external callback if set
        if self.external_update_callback:
            try:
                self.external_update_callback({
                    "events_count": len(events),
                    "avg_advantage": np.mean(advantages) if advantages else 0,
                    "policy_stats": self.online_policy.stats,
                })
            except Exception as e:
                logger.warning(f"External update callback failed: {e}")

    def _event_to_state_vector(self, event: RewardEvent) -> np.ndarray | None:
        """Convert RewardEvent to state vector for policy."""
        try:
            # Build feature vector from event
            features = []

            # Price/action features
            features.append(1.0 if event.action == "BUY" else (-1.0 if event.action == "SELL" else 0.0))
            features.append(event.confidence)
            features.append(event.quantity)
            features.append(event.leverage)

            # PnL features
            features.append(event.unrealized_pnl_pct)
            features.append(event.realized_pnl_pct)
            features.append(event.max_drawdown_pct)
            features.append(event.current_drawdown_pct)

            # Risk features
            features.append(event.position_size_pct)

            # Execution features
            features.append(min(event.execution_latency_ms / 1000, 1.0))  # Normalized latency
            features.append(event.slippage_bps / 100)  # Normalized slippage
            features.append(event.fees_bps / 100)

            # Indicators (top 10)
            indicator_names = sorted(event.indicators.keys())[:10]
            for name in indicator_names:
                features.append(event.indicators[name])

            # Pad/truncate to state_dim
            state_dim = self.online_policy.config.state_dim
            if len(features) < state_dim:
                features.extend([0.0] * (state_dim - len(features)))
            else:
                features = features[:state_dim]

            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to convert event to state vector: {e}")
            return None

    def process_reward_event(self, event: RewardEvent) -> tuple[float, any]:
        """
        Process a reward event through the full pipeline:
        1. Compute streaming reward
        2. Persist to memory stores
        3. Update online policy (via reward engine callback)
        4. Store in ring buffers
        """
        # Process through reward engine
        reward, breakdown = self.reward_engine.process_event(event)

        # Persist to memory if trade is closed
        if event.event_type == RewardEventType.TRADE_CLOSE:
            trade = event.to_trade_outcome()
            market_state = MarketState(
                timestamp=event.timestamp,
                symbol=event.symbol,
                price=event.exit_price or event.entry_price,
                regime=event.regime,
                indicators=event.indicators,
                recent_pnl=event.realized_pnl_pct,
                open_positions=0,  # Would need portfolio state
                confidence=event.confidence,
            )
            self.memory_manager.save_trade(trade, market_state)

        # Store in ring buffers
        self._recent_rewards.append(reward)
        self._advantages.append(reward - self.reward_engine._reward_baseline)
        self._reward_events.append(event)

        # Store state for similarity search
        market_state = MarketState(
            timestamp=event.timestamp,
            symbol=event.symbol,
            price=event.avg_price or event.entry_price,
            regime=event.regime,
            indicators=event.indicators,
            recent_pnl=event.realized_pnl_pct,
            open_positions=0,
            confidence=event.confidence,
        )
        self._recent_states.append(market_state)

        self._total_events += 1

        # Periodic save
        if time.time() - self._last_save_time > self.config.save_interval_seconds:
            self._save_periodic()
            self._last_save_time = time.time()

        return reward, breakdown

    def _save_periodic(self):
        """Periodic save of online policy."""
        self.online_policy.save()
        logger.debug(f"[StreamingMemoryManager] Periodic save (events={self._total_events}, updates={self._total_updates})")

    def get_recent_rewards(self, limit: int = 100) -> list[float]:
        """Get recent rewards from ring buffer."""
        return list(self._recent_rewards)[-limit:]

    def get_recent_advantages(self, limit: int = 100) -> list[float]:
        """Get recent advantages from ring buffer."""
        return list(self._advantages)[-limit:]

    def get_recent_events(self, limit: int = 100) -> list[RewardEvent]:
        """Get recent reward events."""
        return list(self._reward_events)[-limit:]

    def get_similar_states(
        self,
        query_state: MarketState,
        limit: int | None = None,
        threshold: float | None = None,
    ) -> list[dict]:
        """
        Find similar market states from Qdrant vector store.
        Used by meta-learner for context retrieval.
        """
        limit = limit or self.config.max_similar_results
        threshold = threshold or self.config.similarity_threshold

        if not self.memory_manager.is_available("qdrant"):
            # Fallback: search in recent states buffer
            return self._search_local_states(query_state, limit)

        return self.memory_manager.find_similar_states(query_state, limit)

    def _search_local_states(self, query_state: MarketState, limit: int) -> list[dict]:
        """Search recent states buffer for similarity (cosine on embeddings)."""
        # This is a simplified fallback - would need embedder for true similarity
        results = []
        for state in list(self._recent_states)[-1000:]:
            if state.symbol == query_state.symbol:
                # Simple regime match
                score = 1.0 if state.regime == query_state.regime else 0.5
                results.append({"score": score, "payload": {"regime": state.regime, "symbol": state.symbol}})
        return sorted(results, key=lambda x: -x["score"])[:limit]

    def get_state_for_similarity(self, symbol: str, indicators: dict, regime: str) -> MarketState:
        """Create MarketState for similarity search."""
        return MarketState(
            timestamp=datetime.now(),
            symbol=symbol,
            price=indicators.get("close", 0),
            regime=regime,
            indicators=indicators,
            recent_pnl=0.0,
            open_positions=0,
            confidence=0.5,
        )

    def get_performance_stats(self, days: int = 30) -> dict:
        """Get aggregate performance from persistent memory."""
        return self.memory_manager.get_performance_stats(days)

    def get_online_policy_stats(self) -> dict:
        """Get online policy statistics."""
        return self.online_policy.stats.copy()

    def get_reward_engine_stats(self) -> dict:
        """Get streaming reward engine statistics."""
        return {
            **self.reward_engine.stats,
            "buffer_stats": self.reward_engine.get_buffer_stats(),
            "total_events": self._total_events,
            "total_updates": self._total_updates,
        }

    def set_external_update_callback(self, callback: Callable[[dict], None]):
        """Set callback for external systems when policy updates."""
        self.external_update_callback = callback

    def save_all(self):
        """Save all components."""
        self.online_policy.save()
        self._save_periodic()
        logger.info("[StreamingMemoryManager] All components saved")

    def shutdown(self):
        """Graceful shutdown - save everything."""
        if self.config.flush_on_shutdown:
            self.save_all()
        logger.info("[StreamingMemoryManager] Shutdown complete")

    def get_status(self) -> dict:
        """Get full status."""
        return {
            "memory_manager": {
                "redis": self.memory_manager.is_available("redis"),
                "postgres": self.memory_manager.is_available("postgres"),
                "qdrant": self.memory_manager.is_available("qdrant"),
            },
            "reward_engine": self.get_reward_engine_stats(),
            "online_policy": self.get_online_policy_stats(),
            "buffers": {
                "recent_rewards": len(self._recent_rewards),
                "recent_states": len(self._recent_states),
                "advantages": len(self._advantages),
                "events": len(self._reward_events),
            },
            "total_events": self._total_events,
            "total_updates": self._total_updates,
        }


# Factory function
def create_streaming_memory_manager(
    config: StreamingMemoryConfig | None = None,
) -> StreamingMemoryManager:
    """Factory for StreamingMemoryManager."""
    return StreamingMemoryManager(config)


__all__ = [
    "StreamingMemoryConfig",
    "StreamingMemoryManager",
    "create_streaming_memory_manager",
]