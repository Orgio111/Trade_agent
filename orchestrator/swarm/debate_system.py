"""
QUANTEX Agent Swarm Debate System — Multi-agent consensus with
parallel analysis, cross-examination, weighted voting, and risk veto.

Architecture:
  ROUND 1: Parallel independent analysis (7 agents)
  ROUND 2: Cross-examination debate (agents challenge each other)
  ROUND 3: Weighted vote aggregation with credibility scores
  ROUND 4: Risk veto check (blocking — can veto any trade)
  FINAL: Decision with full reasoning trace
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from ..agents import (
    AgentOpinion, MarketAnalystAgent, RiskGuardianAgent,
    SentimentAgent, DeepSeekAnalysisAgent,
)
from ..nim_client import NIMOrchestrator
from ..strategy import Signal
from ..feature_engine import FeatureEngine


@dataclass
class SwarmDecision:
    direction: str  # "long", "short", "hold"
    confidence: float
    leverage: int
    reasoning: str
    vote_breakdown: dict
    agent_opinions: list[dict]
    dissents: list[dict]
    risk_verdict: dict
    debate_log: list[str]
    timestamp: int = 0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = int(time.time() * 1000)


class ScalpingAgent:
    """Fast scalping analysis — very short timeframe (1-5m)."""
    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "scalping_agent"

    async def analyze(self, context: dict) -> AgentOpinion:
        prompt = f"""You are a scalping trader. Analyze for a 1-5 minute scalp trade:

Symbol: {context.get('symbol', 'BTC/USDT')}
Price: ${context.get('price', 0):.2f}
1m Change: {context.get('change_1m', 0):.2f}%
Volume Spike: {context.get('vol_spike', False)}
Order Book Imbalance: {context.get('book_imbalance', 0):.3f}

Output JSON: {{"signal":"long|short|hold","confidence":0.0-1.0,"reasoning":"","target_px":0.0,"sl_px":0.0}}
Focus ONLY on immediate micro-structure. Be quick, be decisive."""
        try:
            resp = await self.nim.route_inference("fast", [{"role": "user", "content": prompt}], temperature=0.1)
            result = json.loads(resp)
            return AgentOpinion(self.agent_id, result["signal"], result["confidence"], result["reasoning"])
        except Exception as e:
            return AgentOpinion(self.agent_id, "hold", 0.0, f"Error: {e}")


class SwingAgent:
    """Medium-term swing analysis (4h-1d timeframe)."""
    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "swing_agent"

    async def analyze(self, context: dict) -> AgentOpinion:
        prompt = f"""You are a swing trader. Analyze for a 4h-1d swing trade:

Symbol: {context.get('symbol', 'BTC/USDT')}
Price: ${context.get('price', 0):.2f}
Regime: {context.get('regime', 'unknown')}
RSI(14): {context.get('rsi', 50):.1f}
EMA Trend: {context.get('ema_trend', 'mixed')}
Support: ${context.get('support', 0):.2f}
Resistance: ${context.get('resistance', 0):.2f}

Output JSON: {{"signal":"long|short|hold","confidence":0.0-1.0,"reasoning":"","target_px":0.0}}
Be patient. Look for high RR setups (>2:1)."""
        try:
            resp = await self.nim.route_inference("reasoning", [{"role": "user", "content": prompt}], temperature=0.1)
            result = json.loads(resp)
            return AgentOpinion(self.agent_id, result["signal"], result["confidence"], result["reasoning"])
        except Exception as e:
            return AgentOpinion(self.agent_id, "hold", 0.0, f"Error: {e}")


class RegimeAgent:
    """Market regime classifier using technical analysis."""
    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "regime_agent"

    async def analyze(self, context: dict) -> AgentOpinion:
        fe = FeatureEngine()
        regime = fe.detect_regime(context.get("df"))
        signal_map = {
            "strong_uptrend": "long", "weak_trend": "hold",
            "ranging": "hold", "high_volatility": "hold",
            "strong_downtrend": "short",
        }
        conf_map = {
            "strong_uptrend": 0.8, "weak_trend": 0.3,
            "ranging": 0.2, "high_volatility": 0.1,
            "strong_downtrend": 0.8,
        }
        return AgentOpinion(
            self.agent_id,
            signal_map.get(regime, "hold"),
            conf_map.get(regime, 0.3),
            f"Regime: {regime}",
            metadata={"regime": regime},
        )


class AnomalyAgent:
    """Anomaly and manipulation detection."""
    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "anomaly_agent"

    async def analyze(self, context: dict) -> AgentOpinion:
        vol = context.get("volume", [])
        price = context.get("price", 0)
        anomalies = []

        # Volume spike > 3x average
        if len(vol) > 20:
            avg_vol = sum(vol[-20:]) / 20
            if vol[-1] > avg_vol * 3:
                anomalies.append(f"Volume spike: {vol[-1]/avg_vol:.1f}x")

        # Price dump/pump
        change_5m = context.get("change_5m", 0)
        if abs(change_5m) > 3:
            anomalies.append(f"Sharp move: {change_5m:.1f}% in 5m")

        if anomalies:
            return AgentOpinion(
                self.agent_id, "hold", 0.2,
                f"Anomalies detected: {'; '.join(anomalies)}",
            )
        return AgentOpinion(self.agent_id, "long", 0.9, "No anomalies detected")


class ExecutionAgent:
    """Execution planning and order routing."""
    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "execution_agent"

    async def analyze(self, context: dict) -> dict:
        prompt = f"""Plan execution for:
Signal: {context.get('direction', 'hold')}
Confidence: {context.get('confidence', 0):.2f}
Balance: ${context.get('balance', 10):.2f}
ATR: {context.get('atr', 0):.4f}
Regime: {context.get('regime', 'unknown')}

Output JSON: {{
    "order_type": "market|limit",
    "leverage": int,
    "dca_levels": int,
    "entry_ladder": [{"level":1,"pct":0.20}, ...],
    "reasoning": ""
}}"""
        try:
            resp = await self.nim.route_inference("fast", [{"role": "user", "content": prompt}], temperature=0.05)
            return json.loads(resp)
        except Exception:
            return {"order_type": "market", "leverage": 3, "dca_levels": 1, "entry_ladder": [{"level": 1, "pct": 1.0}]}


class AgentSwarm:
    """
    Full multi-agent swarm with debate cycle.

    Usage:
        swarm = AgentSwarm(nim_client)
        decision = await swarm.run_swarm_debate(market_context)
        # decision.direction, decision.confidence, decision.leverage, ...
    """

    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agents = {
            "scalping": ScalpingAgent(nim),
            "swing": SwingAgent(nim),
            "sentiment": SentimentAgent(),
            "regime": RegimeAgent(nim),
            "anomaly": AnomalyAgent(nim),
            "deepseek": DeepSeekAnalysisAgent(nim),
            "market": MarketAnalystAgent(nim),
        }
        self.vote_aggregator = VoteAggregator()
        self.risk_guardian = RiskGuardianAgent(nim)

    async def run_swarm_debate(self, context: dict) -> SwarmDecision:
        """
        Full 4-round swarm debate cycle.

        context requires at minimum:
            symbol, price, df (DataFrame), regime
        """
        debate_log = []
        debate_log.append("═" * 50)
        debate_log.append("ROUND 1: Parallel Independent Analysis")

        # ── ROUND 1: Parallel analysis ─────────────
        t0 = time.time()
        analysis_tasks = []
        for agent_id, agent in self.agents.items():
            analysis_tasks.append(self._run_agent(agent_id, agent, context))
        opinions = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        valid_opinions = []
        for i, op in enumerate(opinions):
            agent_id = list(self.agents.keys())[i]
            if isinstance(op, Exception):
                debate_log.append(f"  ⚠️ {agent_id} error: {op}")
                continue
            valid_opinions.append(op)
            debate_log.append(f"  {agent_id}: {op.signal.upper()} conf={op.confidence:.2f}")

        agent_time = time.time() - t0
        debate_log.append(f"  (All agents completed in {agent_time:.2f}s)")

        # ── ROUND 2: Cross-examination debate ──────
        debate_log.append("")
        debate_log.append("ROUND 2: Cross-Examination")
        challenges = await self._run_debate_round(valid_opinions)
        debate_log.extend(challenges)

        # ── ROUND 3: Weighted Vote Aggregation ─────
        debate_log.append("")
        debate_log.append("ROUND 3: Vote Aggregation")
        vote_result = self.vote_aggregator.aggregate(valid_opinions)
        debate_log.append(
            f"  Result: {vote_result['direction'].upper()} "
            f"conf={vote_result['confidence']:.3f}"
        )

        # ── ROUND 4: Risk Veto ─────────────────────
        debate_log.append("")
        debate_log.append("ROUND 4: Risk Veto Check")
        risk_decision = await self.risk_guardian.evaluate(
            Signal(
                direction=vote_result["direction"],
                confidence=vote_result["confidence"],
                reason=vote_result.get("reason", ""),
            ),
            context.get("portfolio", {}),
        )
        vetoed = risk_decision.signal == "hold"
        debate_log.append(f"  Risk verdict: {'APPROVED' if not vetoed else 'VETOED'}")
        debate_log.append(f"  Reason: {risk_decision.reasoning}")

        # ── FINAL ──────────────────────────────────
        if vetoed:
            final_direction = "hold"
            final_confidence = 0.0
        else:
            final_direction = vote_result["direction"]
            final_confidence = vote_result["confidence"] * risk_decision.confidence

        dissents = [
            {"agent": o.agent_id, "signal": o.signal, "confidence": o.confidence}
            for o in valid_opinions if o.signal != final_direction
        ]

        decision = SwarmDecision(
            direction=final_direction,
            confidence=round(final_confidence, 4),
            leverage=3 if final_direction != "hold" else 1,
            reasoning=self._compile_reasoning(valid_opinions, risk_decision),
            vote_breakdown=vote_result,
            agent_opinions=[{"id": o.agent_id, "signal": o.signal,
                             "confidence": o.confidence, "reasoning": o.reasoning}
                            for o in valid_opinions],
            dissents=dissents,
            risk_verdict={
                "approved": not vetoed,
                "reason": risk_decision.reasoning,
                "size_modifier": risk_decision.confidence,
            },
            debate_log=debate_log,
        )
        return decision

    async def _run_agent(self, agent_id: str, agent, context: dict) -> AgentOpinion:
        """Run a single agent with timeout."""
        try:
            return await asyncio.wait_for(
                agent.analyze(context), timeout=30.0
            )
        except asyncio.TimeoutError:
            return AgentOpinion(agent_id, "hold", 0.0, "Timeout (30s)")

    async def _run_debate_round(self, opinions: list[AgentOpinion]) -> list[str]:
        """Generate debate challenges between agents."""
        challenges = []
        for i, op1 in enumerate(opinions):
            for j, op2 in enumerate(opinions):
                if i >= j or op1.signal == op2.signal:
                    continue
                challenge = (
                    f"  Debate: {op1.agent_id}({op1.signal.upper()}) vs "
                    f"{op2.agent_id}({op2.signal.upper()})"
                )
                challenges.append(challenge)
        return challenges

    @staticmethod
    def _compile_reasoning(opinions: list[AgentOpinion], risk: AgentOpinion) -> str:
        """Compile final reasoning from all agents."""
        parts = ["Swarm Decision:"]
        for o in opinions:
            parts.append(f"  {o.agent_id}: {o.signal} ({o.confidence:.2f}) — {o.reasoning[:100]}")
        parts.append(f"  Risk: {risk.reasoning}")
        return "\n".join(parts)


class VoteAggregator:
    """
    Weighted voting system with per-agent credibility tracking.
    """

    def __init__(self):
        self.base_weights = {
            "scalping_agent": 0.20,
            "swing_agent": 0.25,
            "regime_agent": 0.20,
            "sentiment_agent": 0.10,
            "anomaly_agent": 0.15,
            "deepseek_analyst": 0.15,
            "market_analyst": 0.15,
        }

    def aggregate(self, opinions: list[AgentOpinion]) -> dict:
        """Weighted vote aggregation with credibility."""
        direction_scores = {"long": 0.0, "short": 0.0, "hold": 0.0}
        total_weight = 0.0

        for opinion in opinions:
            weight = self.base_weights.get(opinion.agent_id, 0.1)
            credibility = opinion.confidence  # Use confidence as credibility
            direction_scores[opinion.signal] += weight * opinion.confidence * credibility
            total_weight += weight

        if total_weight == 0:
            return {"direction": "hold", "confidence": 0.0, "details": {}}

        best_dir = max(direction_scores, key=direction_scores.get)
        best_score = direction_scores[best_dir] / total_weight

        return {
            "direction": best_dir,
            "confidence": round(best_score, 4),
            "details": {k: round(v / total_weight, 4) if total_weight > 0 else 0
                       for k, v in direction_scores.items()},
        }
