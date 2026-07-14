"""
LangGraph Integration Nodes — RL Streaming + Meta-Learning + SEG Evolution.

Adds nodes to the existing LangGraph trading pipeline:
- rl_streaming_node: Process trade execution → streaming reward → online policy update
- meta_learning_node: Periodic meta-analysis → improved strategy rules
- seg_evolution_node: Strategy generation → backtest → evolution → deploy best
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph

from orchestrator.rl_streaming.streaming_reward import (
    RewardEvent,
    RewardEventType,
    StreamingRewardEngine,
    StreamingRewardConfig,
    create_streaming_reward_engine,
)
from orchestrator.rl_streaming.meta_learner import MetaLearner, MetaLearningConfig, create_meta_learner
from orchestrator.rl_streaming.memory_integration import (
    StreamingMemoryManager,
    StreamingMemoryConfig,
    create_streaming_memory_manager,
)
from orchestrator.seg.seg_generator import StrategyGenerator, GeneratorConfig, create_generator
from orchestrator.seg.backtest_engine import BacktestEngine, BacktestConfig, create_backtest_engine
from orchestrator.seg.evolution_engine import EvolutionEngine, EvolutionConfig, create_evolution_engine

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# RL STREAMING NODE
# ═══════════════════════════════════════════════════════════════════

def create_rl_streaming_node(
    streaming_memory: StreamingMemoryManager | None = None,
    config: StreamingRewardConfig | None = None,
) -> Callable:
    """
    Factory for RL streaming node.

    Expected state keys:
    - execution: dict with order_id, filled_qty, avg_price, status, symbol
    - autogen_signal: dict with action, confidence, entry_reason
    - candle_data: dict with OHLCV, symbol, indicators, regime
    - risk_decision: dict with allow, reason, max_size
    - account: dict with equity, balance, open_positions
    - indicators: dict of technical indicators
    - trace_id: string for tracing
    """
    memory = streaming_memory
    reward_engine = None

    if memory:
        reward_engine = memory.reward_engine
    elif config:
        reward_engine = create_streaming_reward_engine(config)

    def rl_streaming_node(state: dict) -> dict:
        """Process trade execution and compute streaming reward."""
        execution = state.get("execution", {})
        if not execution or execution.get("status") != "FILLED":
            state["rl_streaming_skipped"] = True
            return state

        autogen_signal = state.get("autogen_signal", {})
        candle = state.get("candle_data", {})
        risk_dec = state.get("risk_decision", {})
        account = state.get("account", {})
        indicators = state.get("indicators", {})

        # Create reward event
        event = RewardEvent(
            event_type=RewardEventType.TRADE_CLOSE,
            trade_id=execution.get("order_id", f"trade_{state.get('trace_id', '')}"),
            symbol=candle.get("symbol", "BTCUSDT"),
            action=autogen_signal.get("action", "HOLD"),
            entry_price=execution.get("avg_price", candle.get("close", 0)),
            exit_price=execution.get("avg_price", candle.get("close", 0)),  # For now, same as entry
            quantity=execution.get("filled_qty", 0),
            filled_qty=execution.get("filled_qty", 0),
            avg_price=execution.get("avg_price", candle.get("close", 0)),
            unrealized_pnl_pct=0.0,
            realized_pnl_pct=0.0,  # Will be updated on actual close
            unrealized_pnl_abs=0.0,
            realized_pnl_abs=0.0,
            max_drawdown_pct=0.0,
            current_drawdown_pct=0.0,
            position_size_pct=risk_dec.get("size_pct", 0) / 100 if risk_dec.get("size_pct") else 0,
            leverage=risk_dec.get("leverage", 1),
            confidence=autogen_signal.get("confidence", 0.5),
            regime=candle.get("regime", "unknown"),
            indicators=indicators,
            agent_reasoning=autogen_signal.get("entry_reason", []),
            risk_decision=risk_dec,
            execution_latency_ms=state.get("exec_latency_ms", 0),
            status="FILLED",
        )

        # Process through reward engine
        if reward_engine:
            reward, breakdown = reward_engine.process_event(event)
            state["rl_reward"] = reward
            state["rl_breakdown"] = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown.__dict__
            state["rl_streaming_processed"] = True
        elif memory:
            reward, breakdown = memory.process_reward_event(event)
            state["rl_reward"] = reward
            state["rl_breakdown"] = breakdown.to_dict() if hasattr(breakdown, "to_dict") else breakdown.__dict__
            state["rl_streaming_processed"] = True

        return state

    return rl_streaming_node


# ═══════════════════════════════════════════════════════════════════
# META-LEARNING NODE
# ═══════════════════════════════════════════════════════════════════

def create_meta_learning_node(
    meta_learner: MetaLearner | None = None,
    config: MetaLearningConfig | None = None,
    run_interval_candles: int = 100,  # Run every N candles
) -> Callable:
    """
    Factory for meta-learning node.

    Runs periodic meta-analysis to generate improved strategy rules.
    """
    learner = meta_learner
    candle_counter = 0

    def meta_learning_node(state: dict) -> dict:
        nonlocal candle_counter, learner

        # Initialize learner if needed
        if learner is None and config:
            from orchestrator.rl_memory.memory_store import create_memory_manager
            memory = create_memory_manager()
            learner = create_meta_learner(config, memory)

        if learner is None:
            state["meta_learning_skipped"] = "no_learner"
            return state

        candle_counter += 1

        # Run analysis periodically
        if candle_counter >= run_interval_candles or learner.should_run_analysis():
            candle_counter = 0

            logger.info("[MetaLearningNode] Running meta-analysis...")
            # Run synchronously (could be async in production)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(learner.run_analysis())

            if result:
                state["meta_analysis_result"] = {
                    "timestamp": result.timestamp.isoformat(),
                    "trades_analyzed": result.trades_analyzed,
                    "confidence": result.confidence,
                    "deployed": result.deployed,
                    "winning_patterns": result.winning_patterns[:5],
                    "losing_patterns": result.losing_patterns[:5],
                    "regime_insights": result.regime_insights,
                    "suggested_rules": result.suggested_rules,
                    "backtest_validation": result.backtest_validation,
                }
                state["meta_learning_updated"] = True
                logger.info(f"[MetaLearningNode] Analysis complete: confidence={result.confidence:.2f}, deployed={result.deployed}")
            else:
                state["meta_learning_updated"] = False
                state["meta_analysis_result"] = None

        return state

    return meta_learning_node


# ═══════════════════════════════════════════════════════════════════
# SEG EVOLUTION NODE
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SEGEvolutionState:
    """Extended state for SEG evolution."""
    evolution_engine: EvolutionEngine | None = None
    strategy_generator: StrategyGenerator | None = None
    backtest_engine: BacktestEngine | None = None
    historical_data: Any = None
    last_evolution_candle: int = 0
    evolution_interval_candles: int = 500
    best_strategy: Any = None
    evolution_stats: dict = None


def create_seg_evolution_node(
    evolution_engine: EvolutionEngine | None = None,
    strategy_generator: StrategyGenerator | None = None,
    backtest_engine: BacktestEngine | None = None,
    historical_data: Any = None,
    config: EvolutionConfig | None = None,
    generator_config: GeneratorConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    evolution_interval_candles: int = 500,
    auto_deploy: bool = False,
) -> Callable:
    """
    Factory for SEG evolution node.

    Runs periodic strategy evolution:
    1. Generate new strategies (LLM)
    2. Backtest all strategies
    3. Evolve population (genetic algorithm)
    4. Deploy best strategy if improved
    """
    engine = evolution_engine
    generator = strategy_generator
    backtest = backtest_engine
    data = historical_data
    candle_counter = 0
    deployed_strategy = None

    def seg_evolution_node(state: dict) -> dict:
        nonlocal engine, generator, backtest, data, candle_counter, deployed_strategy

        # Initialize components if needed
        if engine is None:
            if config is None:
                config = EvolutionConfig()
            if generator is None and generator_config:
                generator = create_generator(generator_config)
            if backtest is None and backtest_config:
                backtest = create_backtest_engine(backtest_config)
            if data is None:
                # Try to load default data
                data = backtest.load_data() if backtest else None

            if generator and backtest and data is not None:
                engine = create_evolution_engine(config, backtest, [])
                engine.add_strategies([])  # Will be populated by generator

        if engine is None or generator is None or backtest is None or data is None:
            state["seg_evolution_skipped"] = "components_not_ready"
            return state

        candle_counter += 1

        # Run evolution periodically
        if candle_counter >= evolution_interval_candles:
            candle_counter = 0

            logger.info("[SEGEvolutionNode] Running strategy evolution...")

            try:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # 1. Generate new strategies
                market_context = {
                    "symbol": state.get("candle_data", {}).get("symbol", "BTCUSDT"),
                    "price": state.get("candle_data", {}).get("close", 0),
                    "regime": state.get("features", {}).get("regime", "unknown"),
                    "volatility": state.get("features", {}).get("atr_14", 0),
                }
                regime = state.get("features", {}).get("regime", "unknown")

                new_strategies = loop.run_until_complete(
                    generator.generate_batch(10, market_context, regime)
                )
                logger.info(f"[SEGEvolutionNode] Generated {len(new_strategies)} new strategies")

                # 2. Add to evolution engine
                engine.add_strategies(new_strategies)

                # 3. Run one generation of evolution
                stats = engine.evolve_generation(
                    data=data,
                    symbol=market_context["symbol"],
                    timeframe="1m",
                )

                # 4. Get best strategy
                best = engine.get_best_strategies(1)
                if best:
                    best_strategy = best[0]
                    state["seg_best_strategy"] = {
                        "name": best_strategy.name,
                        "fitness": best_strategy.fitness,
                        "generation": best_strategy.generation,
                        "win_rate": best_strategy.win_rate,
                        "sharpe": best_strategy.sharpe,
                        "max_drawdown": best_strategy.max_drawdown,
                        "total_pnl_pct": best_strategy.total_pnl_pct,
                    }

                    # 5. Auto-deploy if enabled and improved
                    if auto_deploy and deployed_strategy is None:
                        # First generation - deploy
                        deployed_strategy = best_strategy
                        state["seg_deployed"] = True
                        state["seg_deployed_strategy"] = best_strategy.name
                    elif auto_deploy and best_strategy.fitness > deployed_strategy.fitness * 1.02:
                        # 2% improvement - deploy new
                        deployed_strategy = best_strategy
                        state["seg_deployed"] = True
                        state["seg_deployed_strategy"] = best_strategy.name

                state["seg_evolution_completed"] = True
                state["seg_generation"] = engine.generation
                state["seg_population_size"] = len(engine.population)
                state["seg_stats"] = {
                    "best_fitness": stats.best_fitness,
                    "avg_fitness": stats.avg_fitness,
                    "unique_signatures": stats.unique_signatures,
                    "elapsed_ms": stats.elapsed_ms,
                }

                logger.info(
                    f"[SEGEvolutionNode] Evolution gen {engine.generation} complete | "
                    f"Best fitness: {stats.best_fitness:.4f}"
                )

            except Exception as e:
                logger.error(f"[SEGEvolutionNode] Evolution failed: {e}")
                state["seg_evolution_error"] = str(e)

        return state

    return seg_evolution_node


# ═══════════════════════════════════════════════════════════════════
# GRAPH INTEGRATION
# ═══════════════════════════════════════════════════════════════════

def add_streaming_rl_to_graph(
    graph: StateGraph,
    streaming_memory: StreamingMemoryManager | None = None,
    meta_learner: MetaLearner | None = None,
    evolution_engine: EvolutionEngine | None = None,
    strategy_generator: StrategyGenerator | None = None,
    backtest_engine: BacktestEngine | None = None,
    historical_data: Any = None,
    config: dict | None = None,
) -> StateGraph:
    """
    Add RL streaming, meta-learning, and SEG evolution nodes to existing LangGraph.

    Modifies graph in-place:
    - Adds rl_streaming after execution
    - Adds meta_learning as periodic side branch
    - Adds seg_evolution as periodic side branch
    """
    config = config or {}

    # 1. RL Streaming Node (after execution, before log_and_publish)
    if streaming_memory or config.get("rl_streaming"):
        rl_config = config.get("rl_streaming", {})
        graph.add_node(
            "rl_streaming",
            create_rl_streaming_node(streaming_memory, rl_config.get("config"))
        )
        graph.add_edge("execution", "rl_streaming")
        graph.add_edge("rl_streaming", "rl_session_summary")

    # 2. Meta-Learning Node (parallel branch from rl_streaming or execution)
    if meta_learner or config.get("meta_learning"):
        ml_config = config.get("meta_learning", {})
        graph.add_node(
            "meta_learning",
            create_meta_learning_node(meta_learner, ml_config.get("config"), ml_config.get("run_interval_candles", 100))
        )
        # Parallel from execution (runs periodically)
        graph.add_edge("execution", "meta_learning")
        graph.add_edge("meta_learning", END)  # Side branch, doesn't block main flow

    # 3. SEG Evolution Node (parallel branch)
    if evolution_engine or config.get("seg_evolution"):
        seg_config = config.get("seg_evolution", {})
        graph.add_node(
            "seg_evolution",
            create_seg_evolution_node(
                evolution_engine,
                strategy_generator,
                backtest_engine,
                historical_data,
                seg_config.get("evolution_config"),
                seg_config.get("generator_config"),
                seg_config.get("backtest_config"),
                seg_config.get("evolution_interval_candles", 500),
                seg_config.get("auto_deploy", False),
            )
        )
        graph.add_edge("execution", "seg_evolution")
        graph.add_edge("seg_evolution", END)  # Side branch

    return graph


def create_rl_enhanced_pipeline(
    streaming_memory: StreamingMemoryManager | None = None,
    meta_learner: MetaLearner | None = None,
    evolution_engine: EvolutionEngine | None = None,
    strategy_generator: StrategyGenerator | None = None,
    backtest_engine: BacktestEngine | None = None,
    historical_data: Any = None,
    config: dict | None = None,
) -> StateGraph:
    """
    Create a complete pipeline graph with all RL streaming + meta-learning + SEG nodes.

    This creates a new graph from scratch with all the enhanced nodes integrated.
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

    config = config or {}
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

    # Add RL streaming nodes
    rl_config = config.get("rl_streaming", {})
    graph.add_node("rl_streaming", create_rl_streaming_node(streaming_memory, rl_config.get("config")))
    graph.add_node("rl_session_summary", create_rl_session_summary_node())
    graph.add_node("rl_retrieve_similar", create_rl_retrieve_similar_node(streaming_memory))

    # Add meta-learning node
    ml_config = config.get("meta_learning", {})
    graph.add_node("meta_learning", create_meta_learning_node(
        meta_learner, ml_config.get("config"), ml_config.get("run_interval_candles", 100)
    ))

    # Add SEG evolution node
    seg_config = config.get("seg_evolution", {})
    graph.add_node("seg_evolution", create_seg_evolution_node(
        evolution_engine,
        strategy_generator,
        backtest_engine,
        historical_data,
        seg_config.get("evolution_config"),
        seg_config.get("generator_config"),
        seg_config.get("backtest_config"),
        seg_config.get("evolution_interval_candles", 500),
        seg_config.get("auto_deploy", False),
    ))

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

    # Risk -> Execution -> RL Streaming -> Session Summary -> Log
    graph.add_edge("risk_gate", "execution")
    graph.add_edge("execution", "rl_streaming")
    graph.add_edge("rl_streaming", "rl_session_summary")
    graph.add_edge("rl_session_summary", "log_and_publish")
    graph.add_edge("log_and_publish", END)

    # Parallel branches from execution
    graph.add_edge("execution", "rl_retrieve_similar")
    graph.add_edge("execution", "meta_learning")
    graph.add_edge("execution", "seg_evolution")

    # Parallel retrieval for next decision
    graph.add_edge("rl_retrieve_similar", END)
    graph.add_edge("meta_learning", END)
    graph.add_edge("seg_evolution", END)

    return graph


# Helper nodes (copied from langgraph_rl_nodes.py for completeness)
def create_rl_session_summary_node() -> Callable:
    """Factory for rl_session_summary node."""

    def rl_session_summary_node(state: dict) -> dict:
        # Could aggregate from memory_manager.get_performance_stats()
        if "streaming_memory" in state and state["streaming_memory"]:
            stats = state["streaming_memory"].get_performance_stats(1)
        else:
            stats = {"total_trades": 0}
        state["rl_session_stats"] = {
            "trades_this_session": stats.get("total_trades", 0),
            "wins": stats.get("wins", 0),
            "losses": stats.get("losses", 0),
            "total_pnl": stats.get("total_pnl_abs", 0.0),
        }
        return state

    return rl_session_summary_node


def create_rl_retrieve_similar_node(streaming_memory: StreamingMemoryManager | None = None) -> Callable:
    """Factory for rl_retrieve_similar node."""

    def rl_retrieve_similar_node(state: dict) -> dict:
        candle = state.get("candle_data", {})
        indicators = state.get("indicators", {})
        account = state.get("account", {})
        autogen_signal = state.get("autogen_signal", {})

        from orchestrator.rl_memory.memory_store import MarketState

        ms = MarketState(
            timestamp=datetime.now(),
            symbol=candle.get("symbol", "BTCUSDT"),
            price=candle.get("close", 0),
            regime=candle.get("regime", "unknown"),
            indicators=indicators,
            recent_pnl=0.0,
            open_positions=account.get("open_positions", 0) if isinstance(account, dict) else getattr(account, "open_positions", 0),
            confidence=autogen_signal.get("confidence", 0.5),
        )

        similar = []
        if streaming_memory:
            similar = streaming_memory.get_similar_states(ms, limit=5)

        state["rl_similar_states"] = similar
        return state

    return rl_retrieve_similar_node


__all__ = [
    "create_rl_streaming_node",
    "create_meta_learning_node",
    "create_seg_evolution_node",
    "add_streaming_rl_to_graph",
    "create_rl_enhanced_pipeline",
    "create_rl_session_summary_node",
    "create_rl_retrieve_similar_node",
]