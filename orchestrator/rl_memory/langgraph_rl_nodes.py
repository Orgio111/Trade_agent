"""
RL Integration for LangGraph Pipeline — adds RL nodes to existing trading graph.

Adds nodes:
- rl_update: Compute reward, persist trade outcome to memory stores
- rl_on_close: Update trade with exit price when position closes
- rl_session_summary: Emit session stats
- rl_retrieve_similar: Retrieve similar states from Qdrant for context
"""

from __future__ import annotations

import sys
sys.path.append(r"C:\Users\Mns Mns\OneDrive\Documents\GitHub\Trade_agent")

from langgraph.graph import END, StateGraph
from typing import Any, Callable

# Import our RL modules
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


# ─── RL Nodes ───────────────────────────────────────────────────


def create_rl_update_node(memory_manager: RLMemoryManager | None = None) -> Callable:
    """Factory for rl_update node with injected memory manager."""

    def rl_update_node(state: dict) -> dict:
        """
        Compute reward and persist trade outcome after execution.

        Expected state keys:
        - execution: dict with order_id, filled_qty, avg_price, status
        - autogen_signal: dict with action, confidence, entry_reason
        - candle_data: dict with OHLCV, symbol, indicators, regime
        - risk_decision: dict with allow, reason, max_size
        - account: AccountState or dict with equity, balance, open_positions
        """
        execution = state.get("execution", {})
        if not execution or execution.get("status") != "FILLED":
            state["rl_reward"] = 0.0
            state["rl_skipped"] = True
            return state

        autogen_signal = state.get("autogen_signal", {})
        candle = state.get("candle_data", {})
        risk_dec = state.get("risk_decision", {})
        account = state.get("account", {})
        indicators = state.get("indicators", {})

        # Create trade outcome
        trade_id = execution.get("order_id", f"trade_{hash(str(state.get('trace_id', '')))}")
        symbol = candle.get("symbol", "BTCUSDT")
        action = autogen_signal.get("action", "HOLD")
        confidence = autogen_signal.get("confidence", 0.5)
        entry_price = execution.get("avg_price", candle.get("close", 0))
        quantity = execution.get("filled_qty", 0)

        # PnL is 0 at entry - will be updated on close
        pnl_pct = 0.0
        pnl_abs = 0.0

        market_state = MarketState(
            timestamp=state.get("timestamp", __import__("datetime").datetime.now()),
            symbol=symbol,
            price=entry_price,
            regime=candle.get("regime", "unknown"),
            indicators=indicators,
            recent_pnl=pnl_pct,
            open_positions=account.get("open_positions", 0) if isinstance(account, dict) else getattr(account, "open_positions", 0),
            confidence=confidence,
        )

        trade = TradeOutcome(
            trade_id=trade_id,
            timestamp=__import__("datetime").datetime.now(),
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=None,
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

        # Compute reward (initial entry reward)
        reward, breakdown = compute_reward(trade, RewardConfig())

        # Persist to memory
        if memory_manager:
            memory_manager.save_trade(trade, market_state)

        # Update state
        state["rl_reward"] = reward
        state["rl_breakdown"] = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown.__dict__
        state["trade_outcome"] = trade.to_dict()

        return state

    return rl_update_node


def create_rl_on_close_node(memory_manager: RLMemoryManager | None = None) -> Callable:
    """Factory for rl_on_close node - updates trade when position closes."""

    def rl_on_close_node(state: dict) -> dict:
        execution = state.get("execution", {})
        if not execution or execution.get("status") != "FILLED":
            return state

        # In real implementation, would:
        # 1. Look up open trade for this symbol from Redis/Postgres
        # 2. Update with exit_price, pnl_pct, pnl_abs
        # 3. Recompute reward with final PnL
        # 4. Update memory stores
        # 5. Update RL session state

        # Placeholder for now
        return state

    return rl_on_close_node


def create_rl_session_summary_node() -> Callable:
    """Factory for rl_session_summary node - emits session stats."""

    def rl_session_summary_node(state: dict) -> dict:
        # Could aggregate from memory_manager.get_performance_stats()
        state["rl_session_stats"] = {
            "trades_this_session": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
        }
        return state

    return rl_session_summary_node


def create_rl_retrieve_similar_node(memory_manager: RLMemoryManager | None = None) -> Callable:
    """Factory for rl_retrieve_similar node - gets similar states from Qdrant."""

    def rl_retrieve_similar_node(state: dict) -> dict:
        candle = state.get("candle_data", {})
        indicators = state.get("indicators", {})
        account = state.get("account", {})
        autogen_signal = state.get("autogen_signal", {})

        # Import here to avoid circular
        from orchestrator.rl_memory.memory_store import MarketState

        ms = MarketState(
            timestamp=__import__("datetime").datetime.now(),
            symbol=candle.get("symbol", "BTCUSDT"),
            price=candle.get("close", 0),
            regime=candle.get("regime", "unknown"),
            indicators=indicators,
            recent_pnl=0.0,
            open_positions=account.get("open_positions", 0) if isinstance(account, dict) else getattr(account, "open_positions", 0),
            confidence=autogen_signal.get("confidence", 0.5),
        )

        similar = []
        if memory_manager and hasattr(memory_manager, "is_available") and memory_manager.is_available("qdrant"):
            similar = memory_manager.find_similar_states(ms, limit=5)

        state["rl_similar_states"] = similar
        return state

    return rl_retrieve_similar_node


# ─── Graph Integration ──────────────────────────────────────────


def add_rl_nodes_to_graph(
    graph: StateGraph,
    memory_config: dict | None = None,
) -> StateGraph:
    """
    Add RL nodes to existing LangGraph trading pipeline.

    Modifies graph in-place:
    - Adds rl_update after execution
    - Adds rl_on_close after execution (for position closes)
    - Adds rl_session_summary after rl_update
    - Adds rl_retrieve_similar as parallel branch from execution

    Returns modified graph.
    """
    memory_manager = None
    try:
        memory_manager = create_memory_manager(memory_config)
    except Exception as e:
        # Memory stores not available, continue without persistence
        print(f"⚠️  RL Memory stores not available: {e}")
        memory_manager = None

    # Add nodes
    graph.add_node("rl_update", create_rl_update_node(memory_manager))
    graph.add_node("rl_on_close", create_rl_on_close_node(memory_manager))
    graph.add_node("rl_session_summary", create_rl_session_summary_node())
    graph.add_node("rl_retrieve_similar", create_rl_retrieve_similar_node(memory_manager))

    # Add edges: execution → rl_update → rl_session_summary → log_and_publish
    # execution → rl_on_close (parallel for position closes)
    # execution → rl_retrieve_similar (parallel for next decision context)

    graph.add_edge("execution", "rl_update")
    graph.add_edge("rl_update", "rl_session_summary")
    graph.add_edge("rl_session_summary", "log_and_publish")

    # Optional: also route to rl_on_close for position management
    # graph.add_edge("execution", "rl_on_close")

    # Parallel: retrieve similar states for next cycle context
    graph.add_edge("execution", "rl_retrieve_similar")

    return graph


def create_rl_enabled_pipeline(memory_config: dict | None = None) -> StateGraph:
    """
    Create a new pipeline graph with RL nodes integrated.

    This creates a complete graph from scratch with RL nodes.
    """
    from orchestrator.langgraph_pipeline import (
        TradingState,
        PipelineStage,
        data_ingest_node,
        feature_engine_node,
        scalping_node,
        _route_after_scalping,
        vlm_agent_node,
        rag_agent_node,
        swarm_strategy_node,
        risk_gate_node,
        execution_node,
        log_and_publish_node,
    )

    graph = StateGraph(TradingState)

    # Add all original nodes
    graph.add_node("data_ingest", data_ingest_node)
    graph.add_node("feature_engine", feature_engine_node)
    graph.add_node("scalping_node", scalping_node)
    graph.add_node("vlm_agent", vlm_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("swarm_strategy", swarm_strategy_node)
    graph.add_node("risk_gate", risk_gate_node)
    graph.add_node("execution", execution_node)
    graph.add_node("log_and_publish", log_and_publish_node)

    # Add RL nodes
    try:
        memory_manager = create_memory_manager(memory_config)
    except Exception:
        memory_manager = None

    graph.add_node("rl_update", create_rl_update_node(memory_manager))
    graph.add_node("rl_on_close", create_rl_on_close_node(memory_manager))
    graph.add_node("rl_session_summary", create_rl_session_summary_node())
    graph.add_node("rl_retrieve_similar", create_rl_retrieve_similar_node(memory_manager))

    # Entry point
    graph.set_entry_point("data_ingest")

    # Original edges
    graph.add_edge("data_ingest", "feature_engine")
    graph.add_edge("feature_engine", "scalping_node")

    graph.add_conditional_edges(
        "scalping_node",
        _route_after_scalping,
        {
            "execution": "execution",
            "vlm_agent": "vlm_agent",
        },
    )

    graph.add_edge("vlm_agent", "rag_agent")
    graph.add_edge("rag_agent", "swarm_strategy")
    graph.add_edge("swarm_strategy", "risk_gate")

    # Risk → Execution → RL → Log
    graph.add_edge("risk_gate", "execution")
    graph.add_edge("execution", "rl_update")
    graph.add_edge("rl_update", "rl_session_summary")
    graph.add_edge("rl_session_summary", "log_and_publish")
    graph.add_edge("log_and_publish", END)

    # Parallel retrieval for next decision
    graph.add_edge("execution", "rl_retrieve_similar")

    return graph


# ─── Export ────────────────────────────────────────────────────


__all__ = [
    "create_rl_update_node",
    "create_rl_on_close_node",
    "create_rl_session_summary_node",
    "create_rl_retrieve_similar_node",
    "add_rl_nodes_to_graph",
    "create_rl_enabled_pipeline",
]