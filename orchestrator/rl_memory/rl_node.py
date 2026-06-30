"""
LangGraph RL Node — integrates RL reward computation and memory storage into the trading graph.

Adds a node after execution that:
1. Computes reward from trade outcome
2. Persists trade to memory stores (Redis, Postgres, Qdrant)
3. Updates session state for online learning
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Import from our modules
import sys
sys.path.append(r"C:\Users\Mns Mns\OneDrive\Documents\GitHub\Trade_agent")
from orchestrator.rl_memory.memory_store import (
    TradeOutcome,
    MarketState,
    RLMemoryManager,
    create_memory_manager,
)
from orchestrator.rl_memory.reward_function import (
    compute_reward,
    RewardConfig,
    get_reward_stats,
)


# ── RL State Extension ────────────────────────────────────────


@dataclass
class RLState:
    """RL-specific state additions to the trading graph."""

    # Memory manager (initialized once)
    memory_manager: RLMemoryManager | None = None
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    # Current trade tracking
    current_trade_id: str | None = None
    entry_price: float | None = None
    entry_time: datetime | None = None
    max_drawdown_pct: float = 0.0

    # Episode tracking
    episode_trades: list[TradeOutcome] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)

    # Session stats
    session_wins: int = 0
    session_losses: int = 0
    session_total_pnl: float = 0.0

    def add_trade(self, trade: TradeOutcome, reward: float) -> None:
        self.episode_trades.append(trade)
        self.episode_rewards.append(reward)
        if trade.pnl_pct > 0:
            self.session_wins += 1
        else:
            self.session_losses += 1
        self.session_total_pnl += trade.pnl_abs

    def get_session_stats(self) -> dict[str, Any]:
        return {
            "trades": len(self.episode_trades),
            "wins": self.session_wins,
            "losses": self.session_losses,
            "win_rate": self.session_wins / max(1, self.session_wins + self.session_losses),
            "total_pnl": self.session_total_pnl,
            "avg_reward": sum(self.episode_rewards) / max(1, len(self.episode_rewards)),
        }


# ── Initialize Memory Manager ────────────────────────────────


def init_memory_manager(config: dict | None = None) -> RLMemoryManager:
    """Initialize memory manager from config."""
    return create_memory_manager(config)


# ── RL Node Functions ────────────────────────────────────────


def rl_update_node(state: dict) -> dict:
    """
    LangGraph node: Compute reward and persist trade outcome.

    Expected state keys:
    - autogen_signal: dict with action, confidence, reasoning
    - execution_result: dict with order_id, filled_qty, avg_price, status
    - candle_data: dict with OHLCV, symbol, indicators
    - risk_decision: dict with allow, reason, max_size
    - account: AccountState with equity, balance, etc.

    Adds to state:
    - rl_reward: float
    - rl_breakdown: RewardBreakdown
    - trade_outcome: TradeOutcome
    """
    from orchestrator.rl_memory.reward_function import RewardBreakdown

    # Extract required data from state
    autogen_signal = state.get("autogen_signal", {})
    execution = state.get("execution_result", {})
    candle = state.get("candle_data", {})
    risk_dec = state.get("risk_decision", {})
    account = state.get("account", {})
    indicators = state.get("indicators", {})

    # Skip if trade was rejected or error
    if not execution or execution.get("status") != "FILLED":
        state["rl_reward"] = 0.0
        state["rl_skipped"] = True
        return state

    # Create trade outcome
    trade_id = execution.get("order_id", str(uuid.uuid4())[:8])
    symbol = candle.get("symbol", "BTCUSDT")
    action = autogen_signal.get("action", "HOLD")
    confidence = autogen_signal.get("confidence", 0.5)
    entry_price = execution.get("avg_price", candle.get("close", 0))
    quantity = execution.get("filled_qty", 0)

    # Calculate PnL (simplified - in real system would track from entry to exit)
    # For now, use 0 as we don't have exit price yet
    # In practice, this would be called when position is closed
    pnl_pct = 0.0
    pnl_abs = 0.0

    # Build market state for embedding
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol=symbol,
        price=entry_price,
        regime=candle.get("regime", "unknown"),
        indicators=indicators,
        recent_pnl=pnl_pct,
        open_positions=account.get("open_positions", 0),
        confidence=confidence,
    )

    trade = TradeOutcome(
        trade_id=trade_id,
        timestamp=datetime.now(),
        symbol=symbol,
        action=action,
        entry_price=entry_price,
        exit_price=None,  # Will be updated on close
        quantity=quantity,
        pnl_pct=pnl_pct,
        pnl_abs=pnl_abs,
        confidence=confidence,
        regime=candle.get("regime"),
        indicators=indicators,
        agent_reasoning=autogen_signal.get("entry_reason", []),
        risk_decision=risk_dec,
        execution_latency_ms=state.get("exec_latency_ms", 0),
        status=execution.get("status", "FILLED"),
    )

    # Compute reward (using current PnL, will be updated on close)
    reward, breakdown = compute_reward(trade, RewardConfig())

    # Persist to memory
    memory_manager = state.get("memory_manager")
    if memory_manager:
        memory_manager.save_trade(trade, market_state)

    # Update RL state in graph
    rl_state = state.get("rl_state")
    if rl_state and isinstance(rl_state, RLState):
        rl_state.add_trade(trade, reward)

    # Add to state for downstream nodes
    state["rl_reward"] = reward
    state["rl_breakdown"] = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown
    state["trade_outcome"] = trade.to_dict()

    return state


def rl_on_position_close_node(state: dict) -> dict:
    """
    LangGraph node: Called when a position is closed.
    Updates trade with exit price, computes final reward, persists.
    """
    execution = state.get("execution_result", {})
    if not execution or execution.get("status") != "FILLED":
        return state

    # Find the open trade for this symbol
    # In real system, would look up from memory manager
    trade_id = execution.get("order_id")
    if not trade_id:
        return state

    # For now, this is a placeholder - real implementation would:
    # 1. Look up open trade from Redis/Postgres
    # 2. Update with exit_price, pnl_pct, pnl_abs
    # 3. Recompute reward with final PnL
    # 4. Update memory stores
    # 5. Update RL state

    return state


def rl_session_summary_node(state: dict) -> dict:
    """
    LangGraph node: Emit session summary stats for monitoring.
    """
    rl_state = state.get("rl_state")
    if rl_state and isinstance(rl_state, RLState):
        state["rl_session_stats"] = rl_state.get_session_stats()
    return state


def rl_retrieve_similar_node(state: dict) -> dict:
    """
    LangGraph node: Retrieve similar market states from Qdrant for context.

    Adds to state:
    - rl_similar_states: list of similar historical trades
    """
    candle = state.get("candle_data", {})
    indicators = state.get("indicators", {})

    market_state = MarketState(
        timestamp=datetime.now(),
        symbol=candle.get("symbol", "BTCUSDT"),
        price=candle.get("close", 0),
        regime=candle.get("regime", "unknown"),
        indicators=indicators,
        recent_pnl=0.0,
        open_positions=state.get("account", {}).get("open_positions", 0),
        confidence=state.get("autogen_signal", {}).get("confidence", 0.5),
    )

    memory_manager = state.get("memory_manager")
    similar = []
    if memory_manager and memory_manager.is_available("qdrant"):
        similar = memory_manager.find_similar_states(market_state, limit=5)

    state["rl_similar_states"] = similar
    return state


# ── Graph Integration Helper ─────────────────────────────────


def add_rl_nodes_to_graph(graph, config: dict | None = None):
    """
    Add RL nodes to existing LangGraph trading graph.

    Graph edges:
    ... → execution → rl_update → rl_session_summary → END
    ... → execution → rl_retrieve_similar → (parallel)
    """
    config = config or {}

    # Initialize memory manager
    memory_manager = init_memory_manager(config.get("memory_config"))

    # Add nodes
    graph.add_node("rl_update", rl_update_node)
    graph.add_node("rl_on_close", rl_on_position_close_node)
    graph.add_node("rl_session_summary", rl_session_summary_node)
    graph.add_node("rl_retrieve_similar", rl_retrieve_similar_node)

    # Add edges (assuming existing graph has "execution" node)
    # execution → rl_update → rl_session_summary
    graph.add_edge("execution", "rl_update")
    graph.add_edge("rl_update", "rl_session_summary")

    # Optional: parallel retrieval for next decision
    graph.add_edge("execution", "rl_retrieve_similar")

    # Store memory_manager in graph config for node access
    graph.config = {**getattr(graph, "config", {}), "memory_manager": memory_manager}

    return graph


# ── Factory for RL State ─────────────────────────────────────


def create_initial_rl_state(config: dict | None = None) -> dict:
    """Create initial RL state for graph invocation."""
    config = config or {}
    return {
        "rl_state": RLState(
            memory_manager=init_memory_manager(config.get("memory_config")),
            reward_config=RewardConfig.from_dict(config.get("reward_config", {})),
        ),
        "memory_manager": init_memory_manager(config.get("memory_config")),
    }


# ── Export ────────────────────────────────────────────────────


__all__ = [
    "RLState",
    "RLMemoryManager",
    "init_memory_manager",
    "rl_update_node",
    "rl_on_position_close_node",
    "rl_session_summary_node",
    "rl_retrieve_similar_node",
    "add_rl_nodes_to_graph",
    "create_initial_rl_state",
]