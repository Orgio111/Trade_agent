"""
Sentiment-To-Logic (STL) Protocol.
Converts qualitative agent debate into a calibrated quantitative confidence score.
Confidence < 75% → trade automatically blocked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.nim_client import nim_json

log = logging.getLogger(__name__)

_CALIBRATION_SYSTEM = """You are a debate judge at a hedge fund. Given the arguments from Bull, Bear,
Fundamental, Sentiment, and Quant agents, calibrate a final confidence score.

Apply the STL Protocol:
1. Weight each agent by historical accuracy (default equal weights if no history).
2. Penalize contradictions between agents (-5% per contradiction).
3. Penalize "shallow reasoning" (arguments without specific data points, -10%).
4. Penalize "Regime mismatch" (if Quant says not tradeable, cap at 60%).

Return JSON:
{
  "bull_score": <0-1>,
  "bear_score": <0-1>,
  "confidence_pct": <0-100>,
  "final_side": "BUY" | "SELL" | "HOLD",
  "blocked": true | false,
  "block_reason": "<if blocked, explain why>",
  "agent_weights": {"bull": float, "bear": float, "fundamental": float, "sentiment": float, "quant": float},
  "contradictions": ["<contradiction>", ...],
  "rationale": "<2-3 sentence synthesis>"
}
confidence_pct MUST be < 75 if any of: quant says not tradeable, VaR too high, contradictions > 2."""


@dataclass
class STLResult:
    bull_score:     float
    bear_score:     float
    consensus_score: float
    confidence_pct: float
    final_side:     str
    blocked:        bool
    block_reason:   str
    agent_weights:  dict[str, float]
    contradictions: list[str]
    rationale:      str


async def run_stl_protocol(
    symbol: str,
    bull_arg: dict[str, Any],
    bear_arg: dict[str, Any],
    fundamental: dict[str, Any],
    sentiment: dict[str, Any],
    quant: dict[str, Any],
    r_mem_context: str = "",
) -> STLResult:
    """
    Run the full STL Protocol across all 5 agent arguments.
    Returns a calibrated STLResult with confidence_pct.
    """
    debate_input = (
        f"Symbol: {symbol}\n\n"
        f"BULL Agent (score={bull_arg.get('quantitative_score', 0):.2f}):\n"
        f"  Argument: {bull_arg.get('argument', '')}\n"
        f"  Supporting: {bull_arg.get('supporting_factors', [])}\n"
        f"  Risks: {bull_arg.get('risk_factors', [])}\n\n"
        f"BEAR Agent (score={bear_arg.get('quantitative_score', 0):.2f}):\n"
        f"  Argument: {bear_arg.get('argument', '')}\n"
        f"  Supporting: {bear_arg.get('supporting_factors', [])}\n"
        f"  Risks: {bear_arg.get('risk_factors', [])}\n\n"
        f"FUNDAMENTAL Agent: {fundamental.get('rationale', '')} "
        f"(trend={fundamental.get('trend', 'HOLD')}, conf={fundamental.get('confidence', 0):.2f})\n\n"
        f"SENTIMENT Agent: news={sentiment.get('news_score', 0):.2f}, "
        f"social={sentiment.get('social_score', 0):.2f}, "
        f"fear_greed={sentiment.get('fear_greed_index', 50):.0f}\n\n"
        f"QUANT Agent: regime={quant.get('regime', 'MEDIUM')}, "
        f"hurst={quant.get('hurst', 0.5):.3f}, "
        f"tradeable={quant.get('tradeable', True)}, "
        f"confidence_adjustment={quant.get('confidence_adjustment', 0):.2f}\n\n"
        f"R-Mem Historical Context:\n{r_mem_context[:500] if r_mem_context else 'No similar setups.'}"
    )

    result = await nim_json(
        [
            {"role": "system", "content": _CALIBRATION_SYSTEM},
            {"role": "user", "content": debate_input + "\n\nApply STL Protocol and return JSON."},
        ],
        temperature=0.05,
        max_tokens=1024,
    )

    bull_score    = float(result.get("bull_score", 0.5))
    bear_score    = float(result.get("bear_score", 0.5))
    confidence    = float(result.get("confidence_pct", 50.0))
    final_side    = result.get("final_side", "HOLD")
    blocked       = result.get("blocked", confidence < 75.0)
    block_reason  = result.get("block_reason", "")

    if confidence < 75.0 and not blocked:
        blocked = True
        block_reason = f"Confidence {confidence:.1f}% < 75% threshold"

    consensus = abs(bull_score - bear_score)

    log.info(
        "STL Protocol: %s → %s | conf=%.1f%% | blocked=%s",
        symbol, final_side, confidence, blocked,
    )
    if blocked:
        log.warning("Trade BLOCKED: %s", block_reason)

    return STLResult(
        bull_score=bull_score,
        bear_score=bear_score,
        consensus_score=consensus,
        confidence_pct=confidence,
        final_side=final_side,
        blocked=blocked,
        block_reason=block_reason,
        agent_weights=result.get("agent_weights", {}),
        contradictions=result.get("contradictions", []),
        rationale=result.get("rationale", ""),
    )
