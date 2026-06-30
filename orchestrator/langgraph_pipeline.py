"""QUANTEX Multi-Agent Trading Pipeline v3 — Dual-path LangGraph state machine.

Two execution paths based on scalping confidence:
  FAST PATH (<5ms):  feature_engine → scalping_node → risk_gate → execution
  HEAVY PATH (~500ms): feature_engine → [VLM ‖ RAG] → swarm → risk_gate → execution

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                    LangGraph State Machine v3                │
  │                                                              │
  │  candle_close ─→ data_ingest ─→ feature_engine               │
  │                                    │                         │
  │                                    ▼                         │
  │                              scalping_node                   │
  │                              (CPU <5ms)                      │
  │                                    │                         │
  │                    ┌───────────────┴───────────────┐        │
  │                    │ conf > 80%?                    │        │
  │                    ▼ YES                            ▼ NO     │
  │              ┌──────────┐    ┌────→ vlm_agent ──┐   │      │
  │              │ risk_gate│    └────→ rag_agent ──┤   │      │
  │              │ (FAST)   │              ↓        │   │      │
  │              └────┬─────┘       swarm_strategy  │   │      │
  │                   │              ↓              │   │      │
  │                   │         risk_gate (HEAVY)   │   │      │
  │                   │              ↓              │   │      │
  │                   └──→ execution → log_and_publish   │      │
  └─────────────────────────────────────────────────────────────┘

Usage:
    pipeline = MultiAgentPipeline()
    result = await pipeline.process_candle(candle)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger("quantex.pipeline")

# ── Prometheus Metrics (optional, graceful fallback) ────────
try:
    from prometheus_client import Counter, Histogram

    PIPELINE_PATH_TOTAL = Counter(
        "pipeline_path_total",
        "Pipeline path selection count",
        ["path"],  # "fast" or "heavy"
    )
    PIPELINE_LATENCY_MS = Histogram(
        "pipeline_latency_ms",
        "Total pipeline latency in milliseconds",
        ["path"],  # "fast" or "heavy"
        buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
    )
    SCALPING_DECISIONS = Counter(
        "scalping_decisions_total",
        "Scalping engine decisions by action",
        ["action"],  # "BUY", "SELL", "HOLD"
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ── Configuration ───────────────────────────────────────────
# Scalping fast-path confidence threshold (0-100).
# If ScalpingEngine confidence > this value, skip VLM/RAG/swarm.
# Override via TradingState.scalping_confidence_threshold or env var.
SCALPING_CONFIDENCE_THRESHOLD: int = int(os.environ.get("SCALPING_CONFIDENCE_THRESHOLD", "80"))


# ── Enums ───────────────────────────────────────────────────

class PipelineStage(str, Enum):
    INITIALIZED = "initialized"
    DATA_INGESTED = "data_ingested"
    FEATURES_COMPUTED = "features_computed"
    VLM_COMPLETE = "vlm_complete"
    RAG_COMPLETE = "rag_complete"
    STRATEGY_COMPLETE = "strategy_complete"
    RISK_PASSED = "risk_passed"
    RISK_REJECTED = "risk_rejected"
    EXECUTED = "executed"
    LOGGED = "logged"
    FAILED = "failed"


class TradingState(BaseModel):
    """Complete trading state for the multi-agent pipeline v3."""

    # Identity & tracing
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = "BTCUSDT"

    # Market data (from NATS candle_close)
    candle: dict[str, Any] = Field(default_factory=dict)
    candle_history: List[dict[str, Any]] = Field(default_factory=list)

    # Computed features
    features: dict[str, Any] = Field(default_factory=dict)

    # Scalping fast-path result
    scalping_confidence: int = 0  # 0-100, from ScalpingEngine
    scalping_action: str = "HOLD"  # BUY, SELL, HOLD
    scalping_decision: dict[str, Any] = Field(default_factory=dict)

    # VLM output
    vlm_output: dict[str, Any] = Field(default_factory=dict)

    # RAG context
    rag_context: dict[str, Any] = Field(default_factory=dict)

    # Strategy / swarm decision
    strategy_signal: dict[str, Any] = Field(default_factory=dict)

    # Risk gate result
    risk_result: dict[str, Any] = Field(default_factory=dict)

    # Execution result
    execution: dict[str, Any] = Field(default_factory=dict)

    # Pipeline metadata
    stage: str = PipelineStage.INITIALIZED.value
    errors: List[str] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)

    # Pipeline control
    skip_vlm: bool = False
    skip_rag: bool = False
    skip_execution: bool = False
    use_scalping_fast_path: bool = True  # Enable/disable fast path
    scalping_confidence_threshold: int = SCALPING_CONFIDENCE_THRESHOLD  # 0-100, override per-pipeline

    # Shared context for events (ACP v2)
    pipeline_context: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True




# ── Node Implementations ────────────────────────────────────

async def data_ingest_node(state: TradingState) -> TradingState:
    """Validate and ingest candle data from NATS."""
    t0 = time.perf_counter()

    try:
        candle = state.candle
        if not candle:
            raise ValueError("Empty candle data")

        required = ["open", "high", "low", "close", "volume"]
        for f in required:
            if f not in candle:
                raise ValueError(f"Missing field: {f}")

        if candle["low"] <= candle["close"] <= candle["high"]:
            pass  # valid
        else:
            raise ValueError("Invalid candle: close outside range")

        state.symbol = candle.get("symbol", state.symbol)
        state.stage = PipelineStage.DATA_INGESTED.value
        logger.debug(f"[{state.trace_id}] Data ingested: {state.symbol} @ {candle['close']}")

    except Exception as e:
        state.errors.append(f"data_ingest: {e}")
        raise
    finally:
        state.latency_ms["data_ingest"] = (time.perf_counter() - t0) * 1000

    return state


async def feature_engine_node(state: TradingState) -> TradingState:
    """Compute technical indicators from candle history."""
    t0 = time.perf_counter()

    try:
        # Import here to avoid circular imports
        from .feature_engine import FeatureEngine

        fe = FeatureEngine()
        candle = state.candle

        # Build a minimal DataFrame from recent candles
        history = state.candle_history[-200:] if state.candle_history else [candle]
        import pandas as pd
        df = pd.DataFrame(history)

        if len(df) >= 20:
            df = fe.compute_all(df)
            latest = df.iloc[-1].to_dict()

            state.features = {
                "rsi": float(latest.get("rsi_14", 50)),
                "macd": float(latest.get("macd", 0)),
                "macd_hist": float(latest.get("macd_hist", 0)),
                "atr_14": float(latest.get("atr_14", 0)),
                "bb_position": float(latest.get("bb_position", 0.5)),
                "volume_ratio": float(latest.get("vol_ratio", 1.0)),
                "ema_5_cross_20": int(latest.get("ema_5_cross_20", 0)),
                "ema_20_cross_50": int(latest.get("ema_20_cross_50", 0)),
                "price_vs_ema200": float(latest.get("price_vs_ema200", 0)),
                "trend_strength": float(abs(latest.get("macd_hist", 0)) * 100),
            }

            # Regime detection
            state.features["regime"] = fe.detect_regime(df)
        else:
            # Minimal features from single candle
            state.features = {
                "rsi": 50.0, "macd": 0.0, "macd_hist": 0.0,
                "atr_14": candle.get("high", 0) - candle.get("low", 0),
                "bb_position": 0.5, "volume_ratio": 1.0,
                "ema_5_cross_20": 0, "ema_20_cross_50": 0,
                "price_vs_ema200": 0, "trend_strength": 0,
                "regime": "unknown",
            }

        state.stage = PipelineStage.FEATURES_COMPUTED.value

    except Exception as e:
        state.errors.append(f"feature_engine: {e}")
        logger.error(f"[{state.trace_id}] Feature engine failed: {e}")
        # Don't raise — features are enhancement, not blocker
        state.features = {"rsi": 50.0, "regime": "unknown"}
    finally:
        state.latency_ms["feature_engine"] = (time.perf_counter() - t0) * 1000

    return state


async def scalping_node(state: TradingState) -> TradingState:
    """Scalping fast-path: CPU-only decision in <5ms. Skips VLM/RAG/swarm if confident."""
    t0 = time.perf_counter()

    if not state.use_scalping_fast_path:
        state.scalping_confidence = 0
        state.scalping_action = "HOLD"
        return state

    try:
        from .scalping_engine import ScalpingEngine

        engine = ScalpingEngine()

        # Use last 5 candles for scalping window
        last_5 = state.candle_history[-5:] if state.candle_history else [state.candle]

        # Derive volatility and trend from features if available
        atr = state.features.get("atr_14", 0)
        price = state.candle.get("close", 0)
        if price > 0 and atr > 0:
            vol_pct = atr / price
            if vol_pct > 0.02:
                volatility = "high"
            elif vol_pct < 0.005:
                volatility = "low"
            else:
                volatility = "medium"
        else:
            volatility = "medium"

        # Derive trend from features
        ema_cross = state.features.get("ema_5_cross_20", 0)
        if ema_cross > 0:
            trend = "bullish"
        elif ema_cross < 0:
            trend = "bearish"
        else:
            trend = "neutral"

        decision = engine.decide(
            current_price=state.candle.get("close", 0),
            last_5_candles=last_5,
            volatility=volatility,
            trend=trend,
        )

        state.scalping_decision = decision.to_dict()
        state.scalping_confidence = decision.confidence
        state.scalping_action = decision.action
        state.latency_ms["scalping"] = (time.perf_counter() - t0) * 1000

        # Set strategy_signal here (not in routing function) for fast path
        if decision.confidence > state.scalping_confidence_threshold and decision.action in ("BUY", "SELL"):
            state.strategy_signal = {
                "direction": "long" if decision.action == "BUY" else "short",
                "confidence": decision.confidence / 100.0,
                "leverage": 3,
                "reasoning": decision.reason,
                "source": "scalping_fast_path",
                "size_pct": decision.size_pct,
            }

        logger.info(
            f"[{state.trace_id}] Scalping: {decision.action} conf={decision.confidence} "
            f"size={decision.size_pct}% reason={decision.reason} "
            f"({state.latency_ms['scalping']:.1f}ms)"
        )

    except Exception as e:
        state.errors.append(f"scalping: {e}")
        logger.warning(f"[{state.trace_id}] Scalping engine failed: {e}")
        state.scalping_confidence = 0
        state.scalping_action = "HOLD"
        state.latency_ms["scalping"] = (time.perf_counter() - t0) * 1000

    return state


def _route_after_scalping(state: TradingState) -> str:
    """Route based on scalping confidence.

    If confidence > threshold (default 80%) and action is BUY/SELL → fast path to risk_gate.
    Otherwise → heavy path through VLM/RAG/swarm.
    """
    conf = state.scalping_confidence
    action = state.scalping_action

    if conf > state.scalping_confidence_threshold and action in ("BUY", "SELL"):
        logger.info(f"[{state.trace_id}] FAST PATH: scalping conf={conf}% → risk_gate")
        return "risk_gate"

    logger.info(f"[{state.trace_id}] HEAVY PATH: scalping conf={conf}% → VLM+RAG+swarm")
    return "vlm_agent"


async def vlm_agent_node(state: TradingState) -> TradingState:
    """VLM Agent: analyze chart for patterns (GPU inference)."""
    t0 = time.perf_counter()

    if state.skip_vlm:
        state.vlm_output = {"trend": "skipped", "confidence": 0}
        return state

    try:
        from .vlm_agent import VLMAgent

        vlm = VLMAgent()
        candles = state.candle_history[-60:] if state.candle_history else [state.candle]
        result = await vlm.analyze(
            symbol=state.symbol,
            candles=candles,
            indicators=state.features,
            timeframe=state.candle.get("interval", "1m"),
        )
        state.vlm_output = result.to_dict()
        state.stage = PipelineStage.VLM_COMPLETE.value
        logger.debug(f"[{state.trace_id}] VLM: {result.trend} (conf={result.confidence:.2f})")

    except Exception as e:
        state.errors.append(f"vlm_agent: {e}")
        logger.warning(f"[{state.trace_id}] VLM failed (non-blocking): {e}")
        state.vlm_output = {"trend": "unknown", "confidence": 0, "error": str(e)}
    finally:
        state.latency_ms["vlm_agent"] = (time.perf_counter() - t0) * 1000

    return state


async def rag_agent_node(state: TradingState) -> TradingState:
    """RAG Agent: retrieve similar patterns from vector memory."""
    t0 = time.perf_counter()

    if state.skip_rag:
        state.rag_context = {"documents": [], "source": "skipped"}
        return state

    try:
        from .rag_agent import RAGAgent

        rag = RAGAgent()
        result = await rag.retrieve(
            symbol=state.symbol,
            regime=state.features.get("regime", "unknown"),
            rsi=state.features.get("rsi", 50),
            macd=state.features.get("macd", 0),
            volume_ratio=state.features.get("volume_ratio", 1.0),
            price=state.candle.get("close", 0),
            vlm_trend=state.vlm_output.get("trend", ""),
        )
        state.rag_context = result.to_dict()
        state.stage = PipelineStage.RAG_COMPLETE.value
        logger.debug(
            f"[{state.trace_id}] RAG: {len(result.documents)} docs "
            f"({result.source}, {result.query_time_ms:.0f}ms)"
        )

    except Exception as e:
        state.errors.append(f"rag_agent: {e}")
        logger.warning(f"[{state.trace_id}] RAG failed (non-blocking): {e}")
        state.rag_context = {"documents": [], "source": "error", "error": str(e)}
    finally:
        state.latency_ms["rag_agent"] = (time.perf_counter() - t0) * 1000

    return state


async def swarm_strategy_node(state: TradingState) -> TradingState:
    """Strategy Agent: AutoGen GroupChat debate → fallback to swarm → fallback to features.

    Priority:
      1. AutoGen multi-agent debate (agents.md spec) — 5 agents + GroupChat
      2. Swarm debate system (original) — if AutoGen fails
      3. Feature-based signal — last resort
    """
    t0 = time.perf_counter()

    # ── PATH 1: AutoGen Multi-Agent Debate ──────────────────────
    try:
        from .autogen_team import TradingTeam, autogen_to_langgraph
        from .model_loader import ModelLoader, load_agent_models

        # Build candle data for AutoGen team
        candle_data = {
            "symbol": state.symbol,
            "price": state.candle.get("close", 0),
            "ohlcv": {
                "open": state.candle.get("open", 0),
                "high": state.candle.get("high", 0),
                "low": state.candle.get("low", 0),
                "close": state.candle.get("close", 0),
                "volume": state.candle.get("volume", 0),
            },
            "indicators": {
                "rsi_14": state.features.get("rsi", 50),
                "macd": state.features.get("macd", {}),
                "atr_14": state.features.get("atr_14", 0),
                "ema_9": state.features.get("ema_9", 0),
                "ema_21": state.features.get("ema_21", 0),
                "bb_width": state.features.get("bb_width", 0),
            },
            "regime": state.features.get("regime", "unknown"),
            "volume_ratio": state.features.get("volume_ratio", 1.0),
        }

        vlm_output = state.vlm_output if state.vlm_output else None
        market_context = {
            "rag_documents": state.rag_context.get("documents", []),
            "rag_pattern_stats": state.rag_context.get("pattern_stats", {}),
        } if state.rag_context else None

        # Run AutoGen trading cycle (synchronous — runs in thread pool)
        import asyncio
        loop = asyncio.get_event_loop()
        team = TradingTeam()
        autogen_result = await loop.run_in_executor(
            None,
            lambda: team.run_trading_cycle(
                candle_data=candle_data,
                vlm_output=vlm_output,
                market_context=market_context,
            ),
        )

        # Convert to LangGraph-compatible signal
        langgraph_signal = autogen_to_langgraph(autogen_result)

        # Map to existing strategy_signal format
        action = langgraph_signal.get("action", "HOLD")
        state.strategy_signal = {
            "direction": action.lower(),
            "confidence": langgraph_signal.get("confidence", 0.0),
            "leverage": 1.0,
            "reasoning": "; ".join(langgraph_signal.get("entry_reason", [])),
            "risk_notes": langgraph_signal.get("risk_notes", ""),
            "source": "autogen_debate",
        }

        state.stage = PipelineStage.STRATEGY_COMPLETE.value
        logger.info(
            f"[{state.trace_id}] AutoGen Strategy: {action} "
            f"conf={langgraph_signal.get('confidence', 0):.2f}"
        )

    except Exception as autogen_err:
        logger.warning(f"AutoGen debate unavailable ({autogen_err}), trying swarm fallback")

        # ── PATH 2: Swarm Debate System ─────────────────────────
        try:
            from .inference_integration import InferenceIntegration
            from .swarm.debate_system import AgentSwarm

            context = {
                "symbol": state.symbol,
                "price": state.candle.get("close", 0),
                "candle": state.candle,
                "df": None,
                "regime": state.features.get("regime", "unknown"),
                "rsi": state.features.get("rsi", 50),
                "macd": state.features.get("macd", 0),
                "volume_ratio": state.features.get("volume_ratio", 1.0),
                "atr": state.features.get("atr_14", 0),
                "vlm_trend": state.vlm_output.get("trend", "unknown"),
                "rag_documents": state.rag_context.get("documents", []),
                "rag_pattern_stats": state.rag_context.get("pattern_stats", {}),
            }

            nim = InferenceIntegration()
            swarm = AgentSwarm(nim)
            decision = await asyncio.wait_for(
                swarm.run_swarm_debate(context),
                timeout=60.0,
            )

            state.strategy_signal = {
                "direction": decision.direction,
                "confidence": decision.confidence,
                "leverage": decision.leverage,
                "reasoning": decision.reasoning[:500],
                "vote_breakdown": decision.vote_breakdown,
                "dissents": [
                    {"agent": d["agent"], "signal": d["signal"]}
                    for d in decision.dissents
                ],
                "source": "swarm_debate",
            }
            state.stage = PipelineStage.STRATEGY_COMPLETE.value
            logger.info(
                f"[{state.trace_id}] Swarm Strategy: {state.strategy_signal['direction']} "
                f"conf={state.strategy_signal['confidence']:.2f}"
            )

        except Exception as swarm_err:
            # ── PATH 3: Feature-based fallback ────────────────────
            logger.warning(f"Swarm debate unavailable ({swarm_err}), using feature-based signal")
            state.strategy_signal = _fallback_signal(state.features, state.vlm_output, state.rag_context)
            state.stage = PipelineStage.STRATEGY_COMPLETE.value

    except Exception as e:
        state.errors.append(f"strategy: {e}")
        logger.error(f"[{state.trace_id}] Strategy failed: {e}")
        state.strategy_signal = {"direction": "hold", "confidence": 0.0, "error": str(e)}
    finally:
        state.latency_ms["strategy"] = (time.perf_counter() - t0) * 1000

    return state


def _fallback_signal(features: dict, vlm: dict, rag: dict) -> dict:
    """Simple feature-based signal when swarm is unavailable."""
    rsi = features.get("rsi", 50)
    macd_hist = features.get("macd_hist", 0)
    vol_ratio = features.get("volume_ratio", 1.0)
    vlm_trend = vlm.get("trend", "neutral")

    score = 0.0
    reasons = []

    # RSI
    if rsi < 30:
        score += 0.3
        reasons.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > 70:
        score -= 0.3
        reasons.append(f"RSI overbought ({rsi:.0f})")

    # MACD
    if macd_hist > 0:
        score += 0.2
        reasons.append("MACD bullish")
    elif macd_hist < 0:
        score -= 0.2
        reasons.append("MACD bearish")

    # Volume
    if vol_ratio > 1.5:
        score += 0.1
        reasons.append(f"Volume spike ({vol_ratio:.1f}x)")

    # VLM
    if vlm_trend == "bullish":
        score += 0.2
        reasons.append("VLM bullish")
    elif vlm_trend == "bearish":
        score -= 0.2
        reasons.append("VLM bearish")

    # RAG pattern stats
    rag_stats = rag.get("pattern_stats", {})
    if rag_stats.get("sample_size", 0) > 3:
        adj = rag_stats.get("confidence_adjustment", 0)
        score += adj
        reasons.append(f"RAG adj={adj:+.2f}")

    if score > 0.3:
        direction = "long"
    elif score < -0.3:
        direction = "short"
    else:
        direction = "hold"

    return {
        "direction": direction,
        "confidence": min(abs(score), 1.0),
        "leverage": 3 if direction != "hold" else 1,
        "reasoning": "; ".join(reasons) if reasons else "No clear signal",
        "source": "fallback_features",
    }


async def risk_gate_node(state: TradingState) -> TradingState:
    """Risk Engine: hard gate for signal validation."""
    t0 = time.perf_counter()

    try:
        from .risk.risk_engine import RiskEngine

        signal = state.strategy_signal
        if signal.get("direction", "hold") == "hold":
            state.risk_result = {
                "allowed": False,
                "reason": "HOLD signal — no trade",
                "severity": "soft",
            }
            state.stage = PipelineStage.RISK_REJECTED.value
            return state

        risk = RiskEngine()

        # Build portfolio state (would come from PaperAccount in production)
        portfolio = {
            "balance": 1000.0,
            "drawdown": 0.0,
            "daily_pnl": 0.0,
            "consecutive_losses": 0,
            "open_positions": 0,
        }

        proposed_trade = {
            "direction": signal.get("direction", "hold"),
            "confidence": signal.get("confidence", 0),
            "leverage": signal.get("leverage", 1),
            "notional": portfolio["balance"] * 0.02,
            "volatility_percentile": 50,
        }

        result = risk.check_all_gates(proposed_trade, portfolio)

        state.risk_result = {
            "allowed": result.approved,
            "reason": result.reason,
            "severity": result.severity,
            "size_multiplier": result.size_multiplier,
            "max_leverage": result.max_leverage,
        }

        if result.approved:
            state.stage = PipelineStage.RISK_PASSED.value
        else:
            state.stage = PipelineStage.RISK_REJECTED.value

        logger.info(
            f"[{state.trace_id}] Risk: {'PASSED' if result.approved else 'REJECTED'} — {result.reason}"
        )

    except Exception as e:
        state.errors.append(f"risk_gate: {e}")
        logger.error(f"[{state.trace_id}] Risk gate failed: {e}")
        state.risk_result = {"allowed": False, "reason": f"Risk error: {e}", "severity": "hard"}
        state.stage = PipelineStage.RISK_REJECTED.value
    finally:
        state.latency_ms["risk_gate"] = (time.perf_counter() - t0) * 1000

    return state


async def execution_node(state: TradingState) -> TradingState:
    """Execution Layer: place order via broker."""
    t0 = time.perf_counter()

    if state.skip_execution or not state.risk_result.get("allowed", False):
        state.execution = {"status": "skipped", "reason": "Risk rejected or execution skipped"}
        return state

    try:
        signal = state.strategy_signal
        risk = state.risk_result

        # Determine order details
        direction = signal.get("direction", "hold")
        side = "buy" if direction == "long" else "sell"
        size_mult = risk.get("size_multiplier", 1.0)
        base_size = 0.02  # 2% of balance
        final_size = base_size * size_mult

        state.execution = {
            "status": "simulated",
            "symbol": state.symbol,
            "side": side,
            "size_pct": round(final_size, 4),
            "leverage": risk.get("max_leverage", 3),
            "price": state.candle.get("close", 0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        state.stage = PipelineStage.EXECUTED.value
        logger.info(
            f"[{state.trace_id}] Execution: {side} {state.symbol} "
            f"size={final_size:.4f} lev={risk.get('max_leverage', 3)}x"
        )

    except Exception as e:
        state.errors.append(f"execution: {e}")
        logger.error(f"[{state.trace_id}] Execution failed: {e}")
        state.execution = {"status": "error", "reason": str(e)}
    finally:
        state.latency_ms["execution"] = (time.perf_counter() - t0) * 1000

    return state


async def log_and_publish_node(state: TradingState) -> TradingState:
    """Log results and publish events to NATS."""
    t0 = time.perf_counter()

    try:
        total_latency = sum(state.latency_ms.values())

        log_data = {
            "trace_id": state.trace_id,
            "symbol": state.symbol,
            "stage": state.stage,
            "candle_close": state.candle.get("close"),
            "strategy": state.strategy_signal,
            "risk": state.risk_result,
            "execution": state.execution,
            "latency_ms": state.latency_ms,
            "total_latency_ms": round(total_latency, 2),
            "errors": state.errors,
        }

        logger.info(f"[{state.trace_id}] Pipeline complete: total={total_latency:.1f}ms stage={state.stage}")

        # Build pipeline context for downstream events
        state.pipeline_context = {
            "trace_id": state.trace_id,
            "total_latency_ms": round(total_latency, 2),
            "pipeline_stage": state.stage,
            "vlm_trend": state.vlm_output.get("trend"),
            "rag_source": state.rag_context.get("source"),
            "rag_docs": len(state.rag_context.get("documents", [])),
            "strategy_source": state.strategy_signal.get("source"),
            "risk_allowed": state.risk_result.get("allowed", False),
            "scalping_threshold": state.scalping_confidence_threshold,
            "scalping_confidence": state.scalping_confidence,
            "scalping_action": state.scalping_action,
            "fast_path": state.strategy_signal.get("source") == "scalping_fast_path",
        }

        state.stage = PipelineStage.LOGGED.value

        # ── Record Prometheus metrics ──
        if PROMETHEUS_AVAILABLE:
            is_fast = state.strategy_signal.get("source") == "scalping_fast_path"
            path = "fast" if is_fast else "heavy"
            PIPELINE_PATH_TOTAL.labels(path=path).inc()
            PIPELINE_LATENCY_MS.labels(path=path).observe(total_latency)
            action = state.scalping_decision.get("action", "HOLD")
            SCALPING_DECISIONS.labels(action=action).inc()

    except Exception as e:
        state.errors.append(f"logging: {e}")
        logger.error(f"[{state.trace_id}] Logging failed: {e}")
    finally:
        state.latency_ms["logging"] = (time.perf_counter() - t0) * 1000

    return state


# ── Graph Construction ──────────────────────────────────────

def create_pipeline_graph() -> Any:
    """
    Create the LangGraph trading pipeline with dual-path routing.

    Graph topology:
        data_ingest → feature_engine → scalping_node
                                            │
                                    ┌───────┴───────┐
                                    │ conf > 80%?   │
                                    ▼ YES           ▼ NO
                              risk_gate       [VLM ‖ RAG] → swarm → risk_gate
                                    │               │
                                    └───────┬───────┘
                                            ▼
                                    execution → log_and_publish
    """
    graph = StateGraph(TradingState)

    # Add nodes
    graph.add_node("data_ingest", data_ingest_node)
    graph.add_node("feature_engine", feature_engine_node)
    graph.add_node("scalping_node", scalping_node)
    graph.add_node("vlm_agent", vlm_agent_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("swarm_strategy", swarm_strategy_node)
    graph.add_node("risk_gate", risk_gate_node)
    graph.add_node("execution", execution_node)
    graph.add_node("log_and_publish", log_and_publish_node)

    # Entry point
    graph.set_entry_point("data_ingest")

    # Sequential: data → features → scalping
    graph.add_edge("data_ingest", "feature_engine")
    graph.add_edge("feature_engine", "scalping_node")

    # Conditional routing: scalping → fast path or heavy path
    graph.add_conditional_edges(
        "scalping_node",
        _route_after_scalping,
        {
            "risk_gate": "risk_gate",          # Fast path: skip VLM/RAG/swarm
            "vlm_agent": "vlm_agent",           # Heavy path: full analysis
        },
    )

    # Heavy path: VLM → RAG → swarm (sequential for simplicity)
    graph.add_edge("vlm_agent", "rag_agent")
    graph.add_edge("rag_agent", "swarm_strategy")

    # Heavy path: swarm → risk_gate
    graph.add_edge("swarm_strategy", "risk_gate")

    # Common tail: risk → execution → logging
    graph.add_edge("risk_gate", "execution")
    graph.add_edge("execution", "log_and_publish")
    graph.add_edge("log_and_publish", END)

    return graph


# ── Pipeline Runner ─────────────────────────────────────────

class MultiAgentPipeline:
    """
    Main trading pipeline orchestrator.

    Can be driven by:
      1. NATS events (market.candle.>) — production mode
      2. Direct invocation (process_candle) — testing mode
    """

    def __init__(self):
        self.graph = create_pipeline_graph()
        self._compiled = self.graph.compile()
        self._running = False

    async def process_candle(self, candle: dict) -> dict:
        """
        Process a single candle through the full pipeline.

        Returns the final pipeline state as a dict.
        """
        # Build initial state
        initial = TradingState(
            candle=candle,
            symbol=candle.get("symbol", "BTCUSDT"),
            trace_id=str(uuid.uuid4()),
        )

        # Run through graph
        result = await self._compiled.ainvoke(initial)

        # Extract final state
        return {
            "trace_id": result.trace_id,
            "symbol": result.symbol,
            "stage": result.stage,
            "strategy": result.strategy_signal,
            "risk": result.risk_result,
            "execution": result.execution,
            "latency_ms": result.latency_ms,
            "total_latency_ms": round(sum(result.latency_ms.values()), 2),
            "errors": result.errors,
            "context": result.pipeline_context,
        }

    async def run_continuous(self, candle_stream: AsyncGenerator[dict, None]):
        """Process continuous candle stream."""
        self._running = True

        async for candle in candle_stream:
            if not self._running:
                break

            try:
                result = await self.process_candle(candle)
                if result["stage"] == PipelineStage.EXECUTED.value:
                    logger.info(
                        f"Trade executed: {result['strategy']['direction']} "
                        f"@ {result['execution'].get('price', 0)}"
                    )
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                # Continue processing next candle

    def stop(self):
        self._running = False
