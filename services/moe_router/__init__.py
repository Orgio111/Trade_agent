"""
MoE Router Service + Agent Brain System.

Provides:
- Mixture of Experts router for task routing
- Agent brain system (Trend, Scalping, Risk, Macro, etc.)
- Dynamic weight adaptation based on performance
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from services.feature_engine import FeatureVector, create_feature_engine_service
from services.orderbook import OrderBookMetrics, create_order_book_service


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class TaskType(Enum):
    """Types of tasks for routing."""
    MARKET_REGIME = "market_regime"
    TREND_ANALYSIS = "trend_analysis"
    SCALPING_SIGNAL = "scalping_signal"
    RISK_ASSESSMENT = "risk_assessment"
    MACRO_ANALYSIS = "macro_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    ORDER_FLOW = "order_flow"
    PATTERN_RECOGNITION = "pattern_recognition"
    EXECUTION_TIMING = "execution_timing"


class AgentRole(Enum):
    """Agent roles in the swarm."""
    TREND = "trend"
    SCALPING = "scalping"
    RISK = "risk"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    ORDER_FLOW = "order_flow"
    PATTERN = "pattern"
    EXECUTION = "execution"


@dataclass
class AgentOpinion:
    """Single agent's opinion."""
    agent_id: str
    role: AgentRole
    task_type: TaskType
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 - 1.0
    reasoning: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "task_type": self.task_type.value,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class MoEDecision:
    """Final decision from MoE router."""
    symbol: str
    final_signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    agent_opinions: list[AgentOpinion]
    weights_used: dict[str, float]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "final_signal": self.final_signal,
            "confidence": self.confidence,
            "agent_opinions": [o.to_dict() for o in self.agent_opinions],
            "weights_used": self.weights_used,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
        }


# ═══════════════════════════════════════════════════════════════════
# BASE AGENT CLASS
# ═══════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Base class for all trading agents."""

    def __init__(self, agent_id: str, role: AgentRole, config: dict | None = None):
        self.agent_id = agent_id
        self.role = role
        self.config = config or {}
        self.performance_history: deque = deque(maxlen=100)
        self._enabled = True

    @abstractmethod
    async def analyze(self, context: dict) -> AgentOpinion:
        """Analyze context and return opinion."""
        pass

    def record_outcome(self, was_correct: bool, pnl: float = 0.0):
        """Record outcome for weight adaptation."""
        self.performance_history.append({
            "timestamp": datetime.now(),
            "correct": was_correct,
            "pnl": pnl,
        })

    def get_accuracy(self, window: int = 50) -> float:
        """Get recent accuracy."""
        recent = list(self.performance_history)[-window:]
        if not recent:
            return 0.5
        return sum(1 for r in recent if r["correct"]) / len(recent)

    def get_avg_pnl(self, window: int = 50) -> float:
        """Get average PnL."""
        recent = list(self.performance_history)[-window:]
        if not recent:
            return 0.0
        return np.mean([r["pnl"] for r in recent])

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value


# ═══════════════════════════════════════════════════════════════════
# CONCRETE AGENT IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

class TrendAgent(BaseAgent):
    """Trend following agent - analyzes trend strength and direction."""

    def __init__(self, agent_id: str = "trend_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.TREND, config)
        self.min_adx = config.get("min_adx", 25.0) if config else 25.0
        self.ema_fast_period = config.get("ema_fast", 9) if config else 9
        self.ema_slow_period = config.get("ema_slow", 21) if config else 21

    async def analyze(self, context: dict) -> AgentOpinion:
        fv: FeatureVector = context.get("features")
        ob_metrics: OrderBookMetrics = context.get("orderbook")

        if not fv:
            return AgentOpinion(
                agent_id=self.agent_id,
                role=self.role,
                task_type=TaskType.TREND_ANALYSIS,
                signal="HOLD",
                confidence=0.1,
                reasoning=["No features available"],
            )

        # Trend analysis logic
        reasons = []
        signal = "HOLD"
        confidence = 0.5

        # EMA cross
        ema_cross = 0
        if fv.ema_9 > fv.ema_21:
            ema_cross = 1
        elif fv.ema_9 < fv.ema_21:
            ema_cross = -1

        # ADX trend strength
        adx = fv.adx

        # MACD
        macd_bullish = fv.macd > fv.macd_signal and fv.macd_hist > 0
        macd_bearish = fv.macd < fv.macd_signal and fv.macd_hist < 0

        # Price vs key EMAs
        above_ema50 = fv.close > fv.ema_50
        above_ema200 = fv.close > fv.ema_200

        # Combine signals
        bullish_signals = sum([
            ema_cross == 1,
            adx > self.min_adx,
            macd_bullish,
            above_ema50,
            above_ema200,
        ])

        bearish_signals = sum([
            ema_cross == -1,
            adx > self.min_adx,
            macd_bearish,
            not above_ema50,
            not above_ema200,
        ])

        if bullish_signals >= 3:
            signal = "BUY"
            confidence = min(0.5 + bullish_signals * 0.1, 0.95)
            reasons.append(f"Trend bullish: {bullish_signals}/5 signals")
        elif bearish_signals >= 3:
            signal = "SELL"
            confidence = min(0.5 + bearish_signals * 0.1, 0.95)
            reasons.append(f"Trend bearish: {bearish_signals}/5 signals")
        else:
            signal = "HOLD"
            confidence = 0.4
            reasons.append(f"Trend unclear: bullish={bullish_signals}, bearish={bearish_signals}")

        # Add orderbook confirmation
        if ob_metrics:
            if signal == "BUY" and ob_metrics.order_flow_imbalance > 0.1:
                confidence = min(confidence + 0.1, 0.95)
                reasons.append("OFI confirms buy pressure")
            elif signal == "SELL" and ob_metrics.order_flow_imbalance < -0.1:
                confidence = min(confidence + 0.1, 0.95)
                reasons.append("OFI confirms sell pressure")

        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.TREND_ANALYSIS,
            signal=signal,
            confidence=confidence,
            reasoning=reasons,
            metadata={
                "ema_cross": ema_cross,
                "adx": adx,
                "bullish_signals": bullish_signals,
                "bearish_signals": bearish_signals,
            },
        )


class ScalpingAgent(BaseAgent):
    """Ultra-fast scalping agent - 5 candle window."""

    def __init__(self, agent_id: str = "scalping_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.SCALPING, config)
        self.momentum_threshold = config.get("momentum_threshold", 0.001) if config else 0.001
        self.rejection_wick_ratio = config.get("wick_ratio", 0.7) if config else 0.7

    async def analyze(self, context: dict) -> AgentOpinion:
        fv: FeatureVector = context.get("features")
        ob_metrics: OrderBookMetrics = context.get("orderbook")

        if not fv:
            return AgentOpinion(
                agent_id=self.agent_id,
                role=self.role,
                task_type=TaskType.SCALPING_SIGNAL,
                signal="HOLD",
                confidence=0.1,
                reasoning=["No features available"],
            )

        reasons = []
        signal = "HOLD"
        confidence = 0.5

        # Momentum from recent price action
        momentum = fv.price_change_pct
        high_low_range = fv.high_low_range

        # Volume spike
        vol_spike = fv.vol_ratio > 1.5

        # Bollinger band position
        bb_pos = fv.bb_position

        # RSI extremes
        rsi_oversold = fv.rsi_14 < 30
        rsi_overbought = fv.rsi_14 > 70

        # Quick scoring
        buy_score = 0
        sell_score = 0

        if momentum > self.momentum_threshold:
            buy_score += 1
        elif momentum < -self.momentum_threshold:
            sell_score += 1

        if vol_spike:
            if momentum > 0:
                buy_score += 1
            else:
                sell_score += 1

        if rsi_oversold:
            buy_score += 1
        if rsi_overbought:
            sell_score += 1

        if bb_pos < 0.2:
            buy_score += 1
        elif bb_pos > 0.8:
            sell_score += 1

        # Order flow confirmation
        if ob_metrics:
            if ob_metrics.order_flow_imbalance > 0.2:
                buy_score += 1
            elif ob_metrics.order_flow_imbalance < -0.2:
                sell_score += 1

        if buy_score >= 3:
            signal = "BUY"
            confidence = min(0.6 + buy_score * 0.05, 0.9)
            reasons.append(f"Scalp buy: score={buy_score}")
        elif sell_score >= 3:
            signal = "SELL"
            confidence = min(0.6 + sell_score * 0.05, 0.9)
            reasons.append(f"Scalp sell: score={sell_score}")
        else:
            signal = "HOLD"
            confidence = 0.3
            reasons.append(f"Scalp hold: buy={buy_score}, sell={sell_score}")

        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.SCALPING_SIGNAL,
            signal=signal,
            confidence=confidence,
            reasoning=reasons,
            metadata={
                "buy_score": buy_score,
                "sell_score": sell_score,
                "momentum": momentum,
                "vol_ratio": fv.vol_ratio,
            },
        )


class RiskAgent(BaseAgent):
    """Risk assessment agent - hard filter on position sizing and exposure."""

    def __init__(self, agent_id: str = "risk_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.RISK, config)
        self.max_position_pct = config.get("max_position_pct", 0.05) if config else 0.05
        self.max_drawdown = config.get("max_drawdown", 0.03) if config else 0.03
        self.max_leverage = config.get("max_leverage", 3) if config else 3

    async def analyze(self, context: dict) -> AgentOpinion:
        # Risk agent doesn't generate directional signal
        # It provides risk assessment metadata

        portfolio = context.get("portfolio", {})
        proposed_signal = context.get("proposed_signal", "HOLD")
        proposed_size = context.get("proposed_size", 0.0)

        reasons = []
        approved = True
        risk_level = "LOW"

        # Check position size
        if proposed_size > self.max_position_pct:
            reasons.append(f"Position size {proposed_size:.2%} exceeds max {self.max_position_pct:.2%}")
            approved = False
            risk_level = "HIGH"

        # Check portfolio drawdown
        current_dd = portfolio.get("drawdown", 0)
        if current_dd > self.max_drawdown:
            reasons.append(f"Portfolio drawdown {current_dd:.2%} exceeds limit {self.max_drawdown:.2%}")
            approved = False
            risk_level = "CRITICAL"

        # Check leverage
        current_lev = portfolio.get("leverage", 1)
        if current_lev > self.max_leverage:
            reasons.append(f"Leverage {current_lev}x exceeds max {self.max_leverage}x")
            risk_level = "HIGH"

        # Correlation check
        open_positions = portfolio.get("open_positions", [])
        if len(open_positions) >= 3:
            reasons.append("Max concurrent positions reached")
            approved = False

        signal = "HOLD" if not approved else proposed_signal
        confidence = 0.9 if approved else 0.1

        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.RISK_ASSESSMENT,
            signal=signal,
            confidence=confidence,
            reasoning=reasons,
            metadata={
                "approved": approved,
                "risk_level": risk_level,
                "max_position_pct": self.max_position_pct,
                "current_drawdown": current_dd,
            },
        )


class MacroAgent(BaseAgent):
    """Macro analysis agent - higher timeframe, fundamental factors."""

    def __init__(self, agent_id: str = "macro_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.MACRO, config)

    async def analyze(self, context: dict) -> AgentOpinion:
        # Macro agent would analyze:
        # - Funding rates
        # - Basis (futures vs spot)
        # - Macro news
        # - Correlation with traditional markets

        # Simplified for now
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.MACRO_ANALYSIS,
            signal="HOLD",
            confidence=0.5,
            reasoning=["Macro analysis pending implementation"],
            metadata={},
        )


class SentimentAgent(BaseAgent):
    """Sentiment analysis agent - news, social media, funding rates."""

    def __init__(self, agent_id: str = "sentiment_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.SENTIMENT, config)

    async def analyze(self, context: dict) -> AgentOpinion:
        # Would integrate with news API, Twitter, etc.
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.SENTIMENT_ANALYSIS,
            signal="HOLD",
            confidence=0.5,
            reasoning=["Sentiment analysis pending implementation"],
            metadata={},
        )


class OrderFlowAgent(BaseAgent):
    """Order flow analysis agent - OFI, CVD, delta."""

    def __init__(self, agent_id: str = "orderflow_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.ORDER_FLOW, config)

    async def analyze(self, context: dict) -> AgentOpinion:
        ob_metrics: OrderBookMetrics = context.get("orderbook")

        if not ob_metrics:
            return AgentOpinion(
                agent_id=self.agent_id,
                role=self.role,
                task_type=TaskType.ORDER_FLOW,
                signal="HOLD",
                confidence=0.3,
                reasoning=["No orderbook data"],
                metadata={},
            )

        reasons = []
        signal = "HOLD"
        confidence = 0.5

        ofi = ob_metrics.order_flow_imbalance
        cvd = ob_metrics.cvd

        # OFI-based signal
        if ofi > 0.3:
            signal = "BUY"
            confidence = min(0.6 + ofi * 0.5, 0.9)
            reasons.append(f"Strong positive OFI: {ofi:.3f}")
        elif ofi < -0.3:
            signal = "SELL"
            confidence = min(0.6 + abs(ofi) * 0.5, 0.9)
            reasons.append(f"Strong negative OFI: {ofi:.3f}")

        # CVD confirmation
        if signal == "BUY" and cvd > 0:
            reasons.append(f"CVD confirms buying: {cvd:.0f}")
        elif signal == "SELL" and cvd < 0:
            reasons.append(f"CVD confirms selling: {cvd:.0f}")

        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.ORDER_FLOW,
            signal=signal,
            confidence=confidence,
            reasoning=reasons,
            metadata={"ofi": ofi, "cvd": cvd},
        )


class PatternAgent(BaseAgent):
    """Chart pattern recognition agent."""

    def __init__(self, agent_id: str = "pattern_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.PATTERN, config)

    async def analyze(self, context: dict) -> AgentOpinion:
        # Would use VLM or classical pattern recognition
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.PATTERN_RECOGNITION,
            signal="HOLD",
            confidence=0.5,
            reasoning=["Pattern recognition pending VLM integration"],
            metadata={},
        )


class ExecutionAgent(BaseAgent):
    """Execution timing agent - optimal entry/exit timing."""

    def __init__(self, agent_id: str = "execution_001", config: dict | None = None):
        super().__init__(agent_id, AgentRole.EXECUTION, config)

    async def analyze(self, context: dict) -> AgentOpinion:
        # Analyze spread, liquidity, timing
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            task_type=TaskType.EXECUTION_TIMING,
            signal="HOLD",
            confidence=0.5,
            reasoning=["Execution timing pending"],
            metadata={},
        )


# ═══════════════════════════════════════════════════════════════════════
# MoE ROUTER
# ═══════════════════════════════════════════════════════════════════

class MoERouter:
    """
    Mixture of Experts Router.
    Routes tasks to appropriate agents and aggregates opinions.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.agents: dict[AgentRole, BaseAgent] = {}
        self.base_weights: dict[AgentRole, float] = {
            AgentRole.TREND: 1.5,
            AgentRole.SCALPING: 1.2,
            AgentRole.RISK: 2.0,  # Risk gets highest base weight
            AgentRole.MACRO: 0.8,
            AgentRole.SENTIMENT: 0.5,
            AgentRole.ORDER_FLOW: 1.3,
            AgentRole.PATTERN: 0.7,
            AgentRole.EXECUTION: 0.5,
        }
        self._init_agents()

        # Performance tracking for weight adaptation
        self.agent_performance: dict[AgentRole, deque] = {
            role: deque(maxlen=100) for role in AgentRole
        }

    def _init_agents(self):
        """Initialize all agents."""
        self.agents[AgentRole.TREND] = TrendAgent(config=self.config.get("trend"))
        self.agents[AgentRole.SCALPING] = ScalpingAgent(config=self.config.get("scalping"))
        self.agents[AgentRole.RISK] = RiskAgent(config=self.config.get("risk"))
        self.agents[AgentRole.MACRO] = MacroAgent(config=self.config.get("macro"))
        self.agents[AgentRole.SENTIMENT] = SentimentAgent(config=self.config.get("sentiment"))
        self.agents[AgentRole.ORDER_FLOW] = OrderFlowAgent(config=self.config.get("orderflow"))
        self.agents[AgentRole.PATTERN] = PatternAgent(config=self.config.get("pattern"))
        self.agents[AgentRole.EXECUTION] = ExecutionAgent(config=self.config.get("execution"))

    def get_adaptive_weights(self) -> dict[AgentRole, float]:
        """Compute adaptive weights based on recent performance."""
        weights = {}
        for role, agent in self.agents.items():
            if not agent.enabled:
                weights[role] = 0.0
                continue

            base = self.base_weights.get(role, 1.0)
            accuracy = agent.get_accuracy()
            avg_pnl = agent.get_avg_pnl()

            # Weight = base * (accuracy boost) * (pnl boost)
            acc_boost = max(0.5, accuracy / 0.5)  # 1.0 at 50% accuracy, 2.0 at 100%
            pnl_boost = max(0.5, 1.0 + avg_pnl * 10)  # Scale PnL

            weights[role] = base * acc_boost * pnl_boost

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    async def route(
        self,
        symbol: str,
        context: dict,
        task_types: list[TaskType] | None = None,
    ) -> MoEDecision:
        """
        Route analysis to agents and aggregate opinions.
        """
        task_types = task_types or [
            TaskType.TREND_ANALYSIS,
            TaskType.SCALPING_SIGNAL,
            TaskType.RISK_ASSESSMENT,
            TaskType.ORDER_FLOW,
        ]

        # Get adaptive weights
        weights = self.get_adaptive_weights()

        # Collect opinions from relevant agents
        opinions = []
        relevant_roles = {
            TaskType.TREND_ANALYSIS: [AgentRole.TREND, AgentRole.PATTERN],
            TaskType.SCALPING_SIGNAL: [AgentRole.SCALPING, AgentRole.ORDER_FLOW],
            TaskType.RISK_ASSESSMENT: [AgentRole.RISK],
            TaskType.ORDER_FLOW: [AgentRole.ORDER_FLOW],
            TaskType.MARKET_REGIME: [AgentRole.TREND, AgentRole.MACRO],
            TaskType.MACRO_ANALYSIS: [AgentRole.MACRO],
            TaskType.SENTIMENT_ANALYSIS: [AgentRole.SENTIMENT],
            TaskType.PATTERN_RECOGNITION: [AgentRole.PATTERN],
            TaskType.EXECUTION_TIMING: [AgentRole.EXECUTION],
        }

        for task_type in task_types:
            roles = relevant_roles.get(task_type, [])
            for role in roles:
                if role not in self.agents or not self.agents[role].enabled:
                    continue

                agent = self.agents[role]
                try:
                    opinion = await agent.analyze(context)
                    opinions.append(opinion)
                except Exception as e:
                    logger.error(f"Agent {agent.agent_id} error: {e}")

        # Aggregate opinions with weights
        final_signal, final_confidence, reasoning = self._aggregate(opinions, weights)

        # Create trace ID
        trace_id = f"moe_{symbol}_{int(time.time() * 1000)}"

        decision = MoEDecision(
            symbol=symbol,
            final_signal=final_signal,
            confidence=final_confidence,
            agent_opinions=opinions,
            weights_used={r.value: w for r, w in weights.items()},
            reasoning=reasoning,
            trace_id=trace_id,
        )

        return decision

    def _aggregate(
        self,
        opinions: list[AgentOpinion],
        weights: dict[AgentRole, float],
    ) -> tuple[str, float, str]:
        """Aggregate weighted opinions into final decision."""
        if not opinions:
            return "HOLD", 0.0, "No agent opinions available"

        # Weighted voting
        vote_scores = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total_weight = 0.0

        reasons = []

        for opinion in opinions:
            w = weights.get(opinion.role, 0.0)
            if w <= 0:
                continue

            vote_scores[opinion.signal] += w * opinion.confidence
            total_weight += w
            reasons.append(f"{opinion.role.value}: {opinion.signal} ({opinion.confidence:.2f}) - {'; '.join(opinion.reasoning[:2])}")

        if total_weight == 0:
            return "HOLD", 0.0, "No valid weights"

        # Normalize
        for k in vote_scores:
            vote_scores[k] /= total_weight

        # Determine final signal
        final_signal = max(vote_scores, key=vote_scores.get)
        final_confidence = vote_scores[final_signal]

        # Risk override: if RISK agent says HOLD/REJECT, force HOLD
        risk_opinions = [o for o in opinions if o.role == AgentRole.RISK]
        for ro in risk_opinions:
            if not ro.metadata.get("approved", True):
                return "HOLD", 0.1, f"RISK REJECT: {ro.reasoning[0] if ro.reasoning else 'Risk limits exceeded'}"

        reasoning = f"Final: {final_signal} ({final_confidence:.2f}) | Votes: " + \
                    ", ".join([f"{k}={v:.2f}" for k, v in vote_scores.items()])

        return final_signal, final_confidence, reasoning

    def record_outcome(self, decision: MoEDecision, was_correct: bool, pnl: float = 0.0):
        """Record outcome for all agents involved."""
        for opinion in decision.agent_opinions:
            if opinion.agent_id in [a.agent_id for a in self.agents.values()]:
                # Find agent and record
                for agent in self.agents.values():
                    if agent.agent_id == opinion.agent_id:
                        agent.record_outcome(was_correct, pnl)
                        self.agent_performance[agent.role].append({
                            "correct": was_correct,
                            "pnl": pnl,
                            "timestamp": datetime.now(),
                        })
                        break


# ═══════════════════════════════════════════════════════════════════
# MOE SERVICE (Integration with feature engine + orderbook)
# ═══════════════════════════════════════════════════════════════════

class MoEService:
    """
    Complete MoE service with feature engine + orderbook + router.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.router = MoERouter(self.config.get("moe", {}))

        # Sub-services
        self.feature_service = create_feature_engine_service(self.config.get("features", {}))
        self.orderbook_service = create_order_book_service(self.config.get("orderbook", {}))

        # Callbacks
        self._decision_callbacks: list[Callable] = []

        # Wire up
        self.feature_service.on_features(self._on_features)
        self.orderbook_service.on_metrics(self._on_orderbook)

        # Latest data cache
        self._latest_features: dict[str, FeatureVector] = {}
        self._latest_orderbook: dict[str, OrderBookMetrics] = {}

    def on_decision(self, callback: Callable[[MoEDecision], None]):
        self._decision_callbacks.append(callback)

    async def _on_features(self, fv: FeatureVector):
        self._latest_features[fv.symbol] = fv
        await self._maybe_analyze(fv.symbol)

    async def _on_orderbook(self, metrics: OrderBookMetrics):
        self._latest_orderbook[metrics.symbol] = metrics
        await self._maybe_analyze(metrics.symbol)

    async def _maybe_analyze(self, symbol: str):
        """Trigger analysis when both features and orderbook available."""
        if symbol not in self._latest_features or symbol not in self._latest_orderbook:
            return

        context = {
            "features": self._latest_features[symbol],
            "orderbook": self._latest_orderbook[symbol],
            "portfolio": self.config.get("portfolio", {}),
        }

        decision = await self.router.route(symbol, context)

        for cb in self._decision_callbacks:
            try:
                await cb(decision)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")

    async def start(self, symbols: list[str]):
        await self.feature_service.start(symbols)
        await self.orderbook_service.start(symbols)

    async def stop(self):
        await self.feature_service.stop()
        await self.orderbook_service.stop()


def create_moe_service(config: dict | None = None) -> MoEService:
    """Factory for MoEService."""
    return MoEService(config)


if __name__ == "__main__":
    import asyncio

    async def test():
        config = {
            "features": {"market_data": {"exchanges": ["binance"], "binance": {"testnet": True}}},
            "orderbook": {"market_data": {"exchanges": ["binance"], "binance": {"testnet": True}}},
            "moe": {},
        }

        service = create_moe_service(config)

        async def on_decision(d: MoEDecision):
            print(f"DECISION: {d.symbol} {d.final_signal} conf={d.confidence:.2f} trace={d.trace_id}")
            for o in d.agent_opinions:
                print(f"  {o.role.value}: {o.signal} ({o.confidence:.2f}) - {o.reasoning[0] if o.reasoning else ''}")

        service.on_decision(on_decision)
        await service.start(["BTCUSDT", "ETHUSDT"])
        await asyncio.sleep(30)
        await service.stop()

    asyncio.run(test())