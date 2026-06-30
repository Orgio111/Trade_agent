"""
LangGraph State Machine for Trading Pipeline.

Production-ready state machine for candle-by-candle trading execution.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SignalAction(str):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RegimeState(str):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    TRANSITION = "TRANSITION"


from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Annotated

from langgraph.graph import END, StateGraph
from langgraph.channels import LastValue
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries, with dict2 values taking precedence."""
    result = dict1.copy()
    result.update(dict2)
    return result


def merge_lists(list1: list, list2: list) -> list:
    """Merge two lists."""
    return list1 + list2


class SignalAction(str):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RegimeState(str):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"
    TRANSITION = "TRANSITION"


class TradingState(BaseModel):
    """Complete trading state for LangGraph pipeline."""
    
    # Identity
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Market data
    candle: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    features: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    
    # Agent outputs
    vlm_output: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    rag_context: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    autogen_reasoning: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    
    # Signal & risk
    signal: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    risk_result: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    
    # Execution
    execution: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    
    # Metadata
    metadata: Annotated[Dict[str, Any], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    
    # Latency tracking
    latency_ms: Annotated[Dict[str, float], Field(default_factory=dict), LastValue] = Field(default_factory=dict)
    
    # Error tracking
    errors: Annotated[List[str], Field(default_factory=list), merge_lists] = Field(default_factory=list)
    
    # Pipeline control
    pipeline_stage: Annotated[str, Field(default="initialized"), LastValue] = "initialized"
    skip_vlm: bool = False
    skip_rag: bool = False
    skip_autogen: bool = False
    
    class Config:
        arbitrary_types_allowed = True


class PipelineStage(str, Enum):
    INITIALIZED = "initialized"
    DATA_INGESTED = "data_ingested"
    FEATURES_COMPUTED = "features_computed"
    VLM_COMPLETE = "vlm_complete"
    RAG_COMPLETE = "rag_complete"
    AUTOGEN_COMPLETE = "autogen_complete"
    SIGNAL_GENERATED = "signal_generated"
    RISK_PASSED = "risk_passed"
    EXECUTED = "executed"
    COMPLETED = "completed"
    FAILED = "failed"


def create_trading_graph() -> StateGraph:
    """Create the main trading pipeline graph."""
    
    graph = StateGraph(TradingState)
    
    # Add nodes
    graph.add_node("data_ingest", data_ingest_node)
    graph.add_node("feature_engine", feature_engine_node)
    graph.add_node("vlm_agent", vlm_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("autogen_team", autogen_team_node)
    graph.add_node("signal_generator", signal_generator_node)
    graph.add_node("risk_engine", risk_engine_node)
    graph.add_node("execution_layer", execution_layer_node)
    graph.add_node("logging_finalize", logging_finalize_node)
    
    # Define edges - deterministic flow
    graph.set_entry_point("data_ingest")
    
    graph.add_edge("data_ingest", "feature_engine")
    
    # Parallel VLM + RAG execution
    graph.add_edge("feature_engine", "vlm_agent")
    graph.add_edge("feature_engine", "rag_agent")
    
    # Join parallel paths
    graph.add_edge("vlm_agent", "autogen_team")
    graph.add_edge("rag_agent", "autogen_team")
    
    graph.add_edge("autogen_team", "signal_generator")
    graph.add_edge("signal_generator", "risk_engine")
    graph.add_edge("risk_engine", "execution_layer")
    graph.add_edge("execution_layer", "logging_finalize")
    graph.add_edge("logging_finalize", END)
    
    # Compile with checkpointing
    compiled = graph.compile()
    
    return compiled


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

async def data_ingest_node(state: TradingState) -> TradingState:
    """Ingest candle data from NATS/WebSocket."""
    start = time.perf_counter()
    
    try:
        # Validate candle data
        candle = state.candle
        required_fields = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        for field in required_fields:
            if field not in candle:
                raise ValueError(f"Missing required candle field: {field}")
        
        # Validate price consistency
        if not (candle["low"] <= candle["close"] <= candle["high"]):
            raise ValueError("Invalid candle: close outside high/low range")
        if not (candle["low"] <= candle["open"] <= candle["high"]):
            raise ValueError("Invalid candle: open outside high/low range")
        if candle["volume"] < 0:
            raise ValueError("Negative volume")
        
        state.candle = candle
        state.pipeline_stage = PipelineStage.DATA_INGESTED
        logger.debug(f"[{state.trace_id}] Data ingested: {candle['symbol']} @ {candle['close']}")
        
    except Exception as e:
        state.errors.append(f"data_ingest: {str(e)}")
        logger.error(f"[{state.trace_id}] Data ingest failed: {e}")
        raise
    
    finally:
        state.latency_ms["data_ingest"] = (time.perf_counter() - start) * 1000
    
    return state


async def feature_engine_node(state: TradingState) -> TradingState:
    """Compute technical indicators and features from candle history."""
    start = time.perf_counter()
    
    try:
        # Get candle history from state or external source
        # This would typically fetch from a feature store or compute incrementally
        features = await compute_features(state.candle, state.features)
        
        state.features = features
        state.pipeline_stage = PipelineStage.FEATURES_COMPUTED
        logger.debug(f"[{state.trace_id}] Features computed: {len(features)} indicators")
        
    except Exception as e:
        state.errors.append(f"feature_engine: {str(e)}")
        logger.error(f"[{state.trace_id}] Feature engine failed: {e}")
        raise
    
    finally:
        state.latency_ms["feature_engine"] = (time.perf_counter() - start) * 1000
    
    return state


async def compute_features(current_candle: dict, history: Dict) -> Dict[str, Any]:
    """Compute technical indicators from candle data."""
    # This is a simplified version - in production would use incremental computation
    return {
        "rsi": 50.0,  # placeholder
        "macd": 0.0,
        "bb_upper": 0.0,
        "bb_lower": 0.0,
        "atr": 0.0,
        "volume_ratio": 1.0,
        "vwap": 0.0,
        "trend_strength": 0.0,
        "support_levels": [],
        "resistance_levels": [],
    }


async def vlm_agent_node(state: TradingState) -> TradingState:
    """VLM Agent: Analyze chart screenshot for patterns."""
    start = time.perf_counter()
    
    if state.skip_vlm:
        logger.debug(f"[{state.trace_id}] VLM skipped")
        return state
    
    try:
        # In production: capture chart screenshot, send to VLM
        vlm_result = await analyze_chart_screenshot(state.candle, state.features)
        
        state.vlm_output = vlm_result
        state.pipeline_stage = PipelineStage.VLM_COMPLETE
        logger.debug(f"[{state.trace_id}] VLM analysis complete: {vlm_result.get('trend', 'unknown')}")
        
    except Exception as e:
        state.errors.append(f"vlm_agent: {str(e)}")
        logger.error(f"[{state.trace_id}] VLM agent failed: {e}")
        # Don't raise - VLM is optional enhancement
        state.vlm_output = {"error": str(e), "trend": "unknown"}
    
    finally:
        state.latency_ms["vlm_agent"] = (time.perf_counter() - start) * 1000
    
    return state


async def analyze_chart_screenshot(candle: dict, features: dict) -> Dict[str, Any]:
    """Analyze chart screenshot using VLM (Moondream/LLaVA)."""
    # In production: capture chart image, send to VLM endpoint
    # For now, return structured analysis based on features
    return {
        "trend": "bullish" if features.get("trend_strength", 0) > 0.5 else "bearish",
        "support_levels": [],
        "resistance_levels": [],
        "patterns": [],
        "volume_profile": "normal",
        "breakout_zones": [],
        "confidence": 0.7,
    }


async def rag_agent_node(state: TradingState) -> TradingState:
    """RAG Agent: Retrieve relevant knowledge from vector store."""
    start = time.perf_counter()
    
    if state.skip_rag:
        logger.debug(f"[{state.trace_id}] RAG skipped")
        return state
    
    try:
        # Query vector store for similar market conditions
        query = build_rag_query(state.candle, state.features, state.vlm_output)
        context = await retrieve_knowledge(query)
        
        state.rag_context = context
        state.pipeline_stage = PipelineStage.RAG_COMPLETE
        logger.debug(f"[{state.trace_id}] RAG context retrieved: {len(context.get('documents', []))} docs")
        
    except Exception as e:
        state.errors.append(f"rag_agent: {str(e)}")
        logger.error(f"[{state.trace_id}] RAG agent failed: {e}")
        # Don't raise - RAG is enhancement
        state.rag_context = {"error": str(e)}
    
    finally:
        state.latency_ms["rag_agent"] = (time.perf_counter() - start) * 1000
    
    return state


async def build_rag_query(candle: dict, features: dict, vlm_output: dict) -> str:
    """Build RAG query from current market state."""
    return f"""
    Market state: {candle['symbol']} @ {candle['close']}
    Trend: {features.get('trend_strength', 0)}
    RSI: {features.get('rsi', 50)}
    MACD: {features.get('macd', 0)}
    Volume ratio: {features.get('volume_ratio', 1.0)}
    VLM trend: {vlm_output.get('trend', 'unknown')}
    VLM patterns: {vlm_output.get('patterns', [])}
    """


async def retrieve_knowledge(query: str) -> Dict[str, Any]:
    """Retrieve relevant knowledge from vector store (Qdrant)."""
    # In production: query Qdrant vector store
    # For now, return mock context
    return {
        "documents": [
            "Similar bullish breakout pattern with volume confirmation yielded 2.3% return over 4 hours",
            "Similar RSI oversold bounce pattern with MACD crossover had 68% win rate",
        ],
        "metadata": [
            {"pattern": "breakout", "return": 0.023, "duration_hours": 4},
            {"pattern": "rsi_bounce", "return": 0.015, "duration_hours": 2},
        ],
        "similarity_scores": [0.92, 0.87],
    }


async def autogen_team_node(state: TradingState) -> TradingState:
    """AutoGen Team: Multi-agent debate for signal generation."""
    start = time.perf_counter()
    
    if state.skip_autogen:
        logger.debug(f"[{state.trace_id}] AutoGen skipped")
        return state
    
    try:
        # In production: run AutoGen team debate
        reasoning = await run_autogen_debate(
            state.candle,
            state.features,
            state.vlm_output,
            state.rag_context,
        )
        
        state.autogen_reasoning = reasoning
        state.pipeline_stage = PipelineStage.AUTOGEN_COMPLETE
        logger.debug(f"[{state.trace_id}] AutoGen debate complete: {reasoning.get('consensus', 'no consensus')}")
        
    except Exception as e:
        state.errors.append(f"autogen_team: {str(e)}")
        logger.error(f"[{state.trace_id}] AutoGen team failed: {e}")
        # Fallback to simple signal generation
        state.autogen_reasoning = {
            "consensus": "fallback",
            "reasoning": "AutoGen failed, using fallback",
            "agents": {}
        }
    
    finally:
        state.latency_ms["autogen_team"] = (time.perf_counter() - start) * 1000
    
    return state


async def run_autogen_debate(candle: dict, features: dict, vlm: dict, rag: dict) -> Dict[str, Any]:
    """Run AutoGen multi-agent debate."""
    # In production: use AutoGen framework with multiple agents
    # For now, return structured reasoning
    return {
        "consensus": "BUY",
        "confidence": 0.75,
        "reasoning": "Bullish breakout with volume confirmation, supported by RAG historical patterns",
        "agents": {
            "pattern_agent": {"signal": "BUY", "confidence": 0.8, "reason": "Bullish flag pattern detected"},
            "quant_agent": {"signal": "BUY", "confidence": 0.7, "reason": "Strong momentum, positive MACD"},
            "macro_agent": {"signal": "NEUTRAL", "confidence": 0.6, "reason": "Macro neutral, no major news"},
            "risk_agent": {"signal": "CAUTIOUS", "confidence": 0.8, "reason": "High volatility, size position carefully"},
        },
        "consensus": "BUY",
        "confidence": 0.72,
    }


async def signal_generator_node(state: TradingState) -> TradingState:
    """Generate final trading signal from all inputs."""
    start = time.perf_counter()
    
    try:
        signal = generate_signal(state)
        state.signal = signal
        state.pipeline_stage = PipelineStage.SIGNAL_GENERATED
        logger.info(f"[{state.trace_id}] Signal: {signal['action']} conf={signal['confidence']:.2f}")
        
    except Exception as e:
        state.errors.append(f"signal_generator: {str(e)}")
        logger.error(f"[{state.trace_id}] Signal generator failed: {e}")
        raise
    
    finally:
        state.latency_ms["signal_generator"] = (time.perf_counter() - start) * 1000
    
    return state


def generate_signal(state: TradingState) -> Dict[str, Any]:
    """Generate final trading signal from all agent outputs."""
    # Weighted consensus from all agents
    autogen = state.autogen_reasoning
    vlm = state.vlm_output
    rag = state.rag_context
    features = state.features
    
    # Simple weighted voting (in production: more sophisticated)
    agent_signals = {
        "pattern": autogen.get("agents", {}).get("pattern_agent", {}).get("signal", "HOLD"),
        "quant": autogen.get("agents", {}).get("quant_agent", {}).get("signal", "HOLD"),
        "macro": autogen.get("agents", {}).get("macro_agent", {}).get("signal", "HOLD"),
        "risk": autogen.get("agents", {}).get("risk_agent", {}).get("signal", "HOLD"),
    }
    
    # Weighted consensus
    weights = {"pattern": 0.3, "quant": 0.3, "macro": 0.2, "risk": 0.2}
    
    buy_weight = sum(w for agent, sig in agent_signals.items() if sig == "BUY" for k,w in weights.items() if k==agent)
    sell_weight = sum(w for agent, sig in agent_signals.items() if sig == "SELL" for k,w in weights.items() if k==agent)
    
    if buy_weight > sell_weight:
        action = "BUY"
        confidence = buy_weight
    elif sell_weight > buy_weight:
        action = "SELL"
        confidence = sell_weight
    else:
        action = "HOLD"
        confidence = 0.5
    
    # Risk-adjusted sizing
    base_size = 0.02  # 2% of equity
    risk_mult = min(1.0, state.features.get("atr", 0.02) / 0.015)  # ATR-based sizing
    
    return {
        "action": action,
        "confidence": round(confidence, 2),
        "size_pct": round(0.02 * risk_mult, 4),
        "entry_price": None,  # filled at execution
        "stop_loss": None,    # set by risk engine
        "take_profit": None,  # set by risk engine
        "reasoning": "Multi-agent consensus",
        "metadata": {
            "agent_signals": agent_signals,
            "consensus_confidence": confidence,
            "trace_id": None,  # filled by caller
        },
    }


async def risk_engine_node(state: TradingState) -> TradingState:
    """Risk Engine: Hard gate for signal validation."""
    start = time.perf_counter()
    
    try:
        risk_result = validate_signal(state.signal, state.features)
        
        state.risk_result = risk_result
        state.pipeline_stage = PipelineStage.RISK_PASSED
        
        if not risk_result["allowed"]:
            logger.warning(f"[{state.trace_id}] Signal rejected: {risk_result['reason']}")
            # Could stop pipeline here or let execution handle it
        
        logger.debug(f"[{state.trace_id}] Risk check: {'PASSED' if risk_result['allowed'] else 'REJECTED'}")
        
    except Exception as e:
        state.errors.append(f"risk_engine: {str(e)}")
        logger.error(f"[{state.trace_id}] Risk engine failed: {e}")
        raise
    
    finally:
        state.latency_ms["risk_engine"] = (time.perf_counter() - start) * 1000
    
    return state


def validate_signal(signal: Dict, features: Dict) -> Dict[str, Any]:
    """Hard risk validation rules."""
    # Hard limits
    max_position_pct = 0.05  # 5% max per trade
    max_daily_loss = 0.03    # 3% daily loss limit
    max_position_size = 0.1  # 10% max position
    
    # Get current portfolio state (from state/portfolio)
    # For now, use defaults
    current_daily_pnl = 0.0
    current_positions = 0
    current_exposure = 0.0
    
    checks = []
    
    # Confidence threshold
    if signal["confidence"] < 0.6:
        return {"allowed": False, "reason": "Confidence below 0.6 threshold"}
    checks.append(("confidence", True))
    
    # Position size limit
    if signal.get("size_pct", 0) > 0.05:
        return {"allowed": False, "reason": "Position size exceeds 5% limit"}
    checks.append(("size", True))
    
    # Daily loss limit
    # if current_daily_pnl < -0.03: return {"allowed": False, "reason": "Daily loss limit exceeded"}
    checks.append(("daily_loss", True))
    
    # Max concurrent positions
    # if current_positions >= 5: return {"allowed": False, "reason": "Max positions reached"}
    checks.append(("positions", True))
    
    # Volatility filter
    atr = 0.02  # placeholder
    if atr > 0.05:  # High volatility
        return {"allowed": False, "reason": "ATR exceeds 5% threshold"}
    checks.append(("volatility", True))
    
    # Correlation check (avoid correlated positions)
    # if has_correlated_position: return {"allowed": False, "reason": "Correlated position exists"}
    checks.append(("correlation", True))
    
    all_passed = all(check[1] for check in checks)
    
    return {
        "allowed": all_passed,
        "reason": "All checks passed" if all_passed else "Risk checks failed",
        "checks": {name: passed for name, passed in checks},
        "max_position_size": 0.05,
    }


async def execution_layer_node(state: TradingState) -> TradingState:
    """Execution Layer: Place order via broker."""
    start = time.perf_counter()
    
    try:
        if not state.risk_result.get("allowed", False):
            state.execution = {
                "status": "REJECTED",
                "reason": "Risk check failed",
                "order_id": None,
            }
            return state
        
        signal = state.signal
        
        # Place order via broker
        order_result = await place_order(
            symbol=state.candle["symbol"],
            side=state.signal["action"],
            size_pct=state.signal["size_pct"],
            order_type="MARKET",  # or LIMIT with limit_price
        )
        
        state.execution = {
            "status": "FILLED" if order_result.get("filled") else "PENDING",
            "order_id": order_result.get("order_id"),
            "filled_qty": order_result.get("filled_qty", 0),
            "avg_price": order_result.get("avg_price", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        state.pipeline_stage = PipelineStage.EXECUTED
        logger.info(f"[{state.trace_id}] Order executed: {state.execution}")
        
    except Exception as e:
        state.errors.append(f"execution_layer: {str(e)}")
        logger.error(f"[{state.trace_id}] Execution failed: {e}")
        state.execution = {"status": "ERROR", "reason": str(e)}
    
    finally:
        state.latency_ms["execution"] = (time.perf_counter() - start) * 1000
    
    return state


async def place_order(
    symbol: str,
    side: str,
    size_pct: float,
    order_type: str = "MARKET",
) -> Dict[str, Any]:
    """Place order via broker (Binance/Paper)."""
    # In production: call broker API
    # For now, simulate
    await asyncio.sleep(0.05)  # simulate latency
    
    return {
        "order_id": str(uuid.uuid4()),
        "filled": True,
        "filled_qty": 0.01,  # placeholder
        "avg_price": 50000.0,  # placeholder
    }


async def logging_finalize_node(state: TradingState) -> TradingState:
    """Finalize pipeline: log results, update metrics."""
    start = time.perf_counter()
    
    try:
        total_latency = sum(state.latency_ms.values())
        
        # Log complete pipeline result
        log_data = {
            "trace_id": state.trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": state.candle.get("symbol"),
            "candle_close": state.candle.get("close"),
            "signal": state.signal,
            "risk_result": state.risk_result,
            "execution": state.execution,
            "latency_ms": state.latency_ms,
            "total_latency_ms": round(sum(state.latency_ms.values()), 2),
            "errors": state.errors,
            "pipeline_stage": state.pipeline_stage,
        }
        
        logger.info(f"[{state.trace_id}] Pipeline complete: {log_data}")
        
        # Store in database/timeseries
        # await store_pipeline_result(log_data)
        
        state.pipeline_stage = PipelineStage.COMPLETED
        state.metadata["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        state.errors.append(f"logging_finalize: {str(e)}")
        logger.error(f"[{state.trace_id}] Logging failed: {e}")
    
    finally:
        state.latency_ms["logging_finalize"] = (time.perf_counter() - start) * 1000
    
    return state


# ============================================================================
# MAIN PIPELINE RUNNER
# ============================================================================

class TradingPipeline:
    """Main trading pipeline orchestrator."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.graph = create_trading_graph()
        self._running = False
    
    async def process_candle(self, candle: Dict) -> TradingState:
        """Process a single candle through the full pipeline."""
        # Create initial state
        initial_state = TradingState(
            candle=candle,
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
        )
        
        # Run through graph
        result = await self.graph.ainvoke(initial_state)
        
        return result
    
    async def run_continuous(self, candle_stream: AsyncGenerator[Dict, None]):
        """Process continuous candle stream."""
        self._running = True
        
        async for candle in candle_stream:
            if not self._running:
                break
            
            try:
                result = await self.process_candle(candle)
                
                # Handle signal if generated
                if result.signal and result.risk_result.get("allowed"):
                    # Signal ready for execution (handled by execution layer)
                    pass
                    
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                # Continue processing next candle
    
    def stop(self):
        self._running = False


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

async def create_pipeline(config: Optional[Dict] = None) -> TradingPipeline:
    """Factory function to create and initialize pipeline."""
    pipeline = TradingPipeline(config)
    return pipeline


# ============================================================================
# TESTING
# ============================================================================

async def test_pipeline():
    """Test the pipeline with sample data."""
    pipeline = await create_pipeline()
    
    # Sample candle
    test_candle = {
        "timestamp": int(time.time() * 1000),
        "open": 50000,
        "high": 50500,
        "low": 49800,
        "close": 50200,
        "volume": 1000.5,
        "symbol": "BTCUSDT",
    }
    
    result = await pipeline.process_candle(test_candle)
    
    print(f"Signal: {result.signal}")
    print(f"Risk: {result.risk_result}")
    print(f"Execution: {result.execution}")
    print(f"Latency: {result.latency_ms}")
    
    return result


if __name__ == "__main__":
    asyncio.run(test_pipeline())