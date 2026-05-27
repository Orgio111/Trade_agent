"""
Council: Adversarial Bull/Bear debate with Sentiment-To-Logic (STL) Protocol.

Two adversarial agents argue the trade. STL converts qualitative debate into
quantitative confidence scores. The supervisor synthesizes the final decision.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from core.config import get_settings
from core.messaging import MsgType, get_bus
from core.models import (
    CouncilDecision,
    DebateArgument,
    FeatureSignal,
    FundamentalSignal,
    SentimentSignal,
    Side,
    TechnicalSignal,
)
from agents.stl_protocol import STLResult, run_stl_protocol
from agents.quant_sentinel import QuantSentinelAgent
from core.nim_client import nim_json
from core.observability import AGENT_LATENCY, COUNCIL_CONSENSUS

logger = logging.getLogger(__name__)

# ── Memory-augmented system prompt variants ─────────────────────────────────
# These accept an optional {memory_context} placeholder injected by
# the SemanticMemoryPipeline before agent deliberation.

_BULL_SYSTEM = """You are the BULL analyst in a hedge fund debate. Your job is to build
the strongest possible BUY case. Be rigorous and data-driven; cherry-pick
only legitimate bullish signals. Output JSON:
{
  "position": "BUY",
  "argument": "<2-3 sentences>",
  "quantitative_score": <0.0-1.0 confidence you are right>,
  "supporting_factors": ["<factor>", ...],
  "risk_factors": ["<risk that could invalidate your thesis>", ...]
}"""

_BEAR_SYSTEM = """You are the BEAR analyst in a hedge fund debate. Your job is to build
the strongest possible SELL/SHORT case. Be rigorous. Output JSON:
{
  "position": "SELL",
  "argument": "<2-3 sentences>",
  "quantitative_score": <0.0-1.0 confidence you are right>,
  "supporting_factors": ["<factor>", ...],
  "risk_factors": ["<risk that could invalidate your thesis>", ...]
}"""

_SUPERVISOR_SYSTEM = """You are the Portfolio Manager. You have seen the Bull and Bear debate.
Synthesize using the STL Protocol:
  - Assign a bull_score and bear_score (must sum to 1.0).
  - consensus_score = |bull_score - bear_score| (directional conviction).
  - final_side = "BUY" if bull_score > 0.55, "SELL" if bear_score > 0.55, else "HOLD".
Return JSON:
{
  "bull_score": <0-1>,
  "bear_score": <0-1>,
  "consensus_score": <0-1>,
  "final_side": "BUY" | "SELL" | "HOLD",
  "rationale": "<2-3 sentences of PM reasoning>"
}"""


def _build_market_context(
    symbol: str,
    technical: TechnicalSignal | None,
    fundamental: FundamentalSignal | None,
    sentiment: SentimentSignal | None,
    features: FeatureSignal | None = None,
) -> str:
    parts = [f"Symbol: {symbol}"]
    if technical:
        parts.append(
            f"Technical: RSI={technical.rsi_14:.1f}, EMA_fast={technical.ema_fast:.2f}, "
            f"EMA_slow={technical.ema_slow:.2f}, ATR={technical.atr_14:.4f}, "
            f"MACD={technical.macd:.4f}/{technical.macd_signal:.4f}, "
            f"trend={technical.trend.value}, conf={technical.confidence:.2f}"
        )
    if features:
        def _fmt(val, fmt: str) -> str:
            """Format a value or return 'N/A' if None."""
            return f"{val:{fmt}}" if val is not None else "N/A"

        parts.append(
            f"Order Flow: OFI={_fmt(features.ofi, '+.3f')}, CVD={_fmt(features.cvd, '.0f')}, "
            f"CVD_delta={_fmt(features.cvd_delta, '+.2f')}, trade_strength={_fmt(features.trade_strength, '.2f')}"
        )
        if features.funding_rate is not None:
            parts.append(
                f"Funding: rate={features.funding_rate:+.5f}% (ann.), "
                f"delta={_fmt(features.funding_rate_delta, '+.5f')}"
            )
        if features.open_interest is not None:
            parts.append(
                f"Open Interest: OI={features.open_interest:.0f}, "
                f"delta={_fmt(features.open_interest_delta, '+.0f')}, "
            )
            if features.oi_price_delta_corr is not None:
                parts[-1] += f"OI-price_corr={features.oi_price_delta_corr:+.2f}"
        parts.append(f"Feature trend: {features.trend.value} conf={features.confidence:.2f}")
    if fundamental:
        parts.append(
            f"Fundamental: on_chain_score={fundamental.on_chain_score:.2f}, "
            f"NVT={fundamental.nvt_ratio}, MVRV={fundamental.mvrv_zscore}, "
            f"trend={fundamental.trend.value}, conf={fundamental.confidence:.2f}"
        )
    if sentiment:
        parts.append(
            f"Sentiment: news={sentiment.news_score:.2f}, social={sentiment.social_score:.2f}, "
            f"fear_greed={sentiment.fear_greed_index:.0f}, "
            f"trend={sentiment.trend.value}, conf={sentiment.confidence:.2f}"
        )
    return "\n".join(parts)


async def _argue(system_prompt: str, context: str, role: str) -> DebateArgument:
    result = await nim_json(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Market context:\n{context}\n\nMake your case."},
        ],
        temperature=0.4,
    )
    return DebateArgument(
        agent_name=role,
        position=Side(result.get("position", "HOLD")),
        argument=result.get("argument", ""),
        quantitative_score=float(result.get("quantitative_score", 0.5)),
        supporting_factors=result.get("supporting_factors", []),
        risk_factors=result.get("risk_factors", []),
    )


class CouncilAgent:
    """Runs the adversarial Bull/Bear debate and applies STL Protocol."""

    async def deliberate(
        self,
        symbol: str,
        closes: np.ndarray | None = None,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        technical: TechnicalSignal | None = None,
        fundamental: FundamentalSignal | None = None,
        sentiment: SentimentSignal | None = None,
        features: FeatureSignal | None = None,
        memory_context: str = "",
    ) -> CouncilDecision:
        session_id = str(uuid.uuid4())
        context = _build_market_context(symbol, technical, fundamental, sentiment, features)

        # Inject historical memory context if available
        enriched_context = context
        if memory_context:
            enriched_context = f"{context}\n\n{memory_context}"

        with AGENT_LATENCY.labels(agent="council").time():
            bull_arg, bear_arg = await asyncio.gather(
                _argue(_BULL_SYSTEM, enriched_context, "BullAgent"),
                _argue(_BEAR_SYSTEM, enriched_context, "BearAgent"),
            )

            # ── QuantSentinel analysis for STL calibration ───────────────
            quant_result: dict = {}
            if closes is not None and highs is not None and lows is not None:
                try:
                    quant_agent = QuantSentinelAgent()
                    quant_result = await quant_agent.analyze(symbol, closes, highs, lows)
                except Exception as exc:
                    logger.debug("QuantSentinel analysis failed: %s", exc)

            # ── Apply STL Protocol calibration ───────────────────────────
            stl: STLResult | None = None
            try:
                fundamental_dict = {
                    "rationale": fundamental.rationale if fundamental else "",
                    "trend": fundamental.trend.value if fundamental else "HOLD",
                    "confidence": fundamental.confidence if fundamental else 0.5,
                } if fundamental else {}
                sentiment_dict = {
                    "news_score": sentiment.news_score if sentiment else 0.0,
                    "social_score": sentiment.social_score if sentiment else 0.0,
                    "fear_greed_index": sentiment.fear_greed_index if sentiment else 50,
                } if sentiment else {}

                stl = await run_stl_protocol(
                    symbol=symbol,
                    bull_arg={
                        "quantitative_score": bull_arg.quantitative_score,
                        "argument": bull_arg.argument,
                        "supporting_factors": bull_arg.supporting_factors,
                        "risk_factors": bull_arg.risk_factors,
                    },
                    bear_arg={
                        "quantitative_score": bear_arg.quantitative_score,
                        "argument": bear_arg.argument,
                        "supporting_factors": bear_arg.supporting_factors,
                        "risk_factors": bear_arg.risk_factors,
                    },
                    fundamental=fundamental_dict,
                    sentiment=sentiment_dict,
                    quant=quant_result,
                    r_mem_context=memory_context[:500] if memory_context else "",
                )
            except Exception as exc:
                logger.debug("STL Protocol calibration failed: %s", exc)

            # ── Use STL scores if available, otherwise fallback to NIM synthesis ──
            if stl is not None and not stl.blocked:
                bull_score = stl.bull_score
                bear_score = stl.bear_score
                consensus_score = stl.consensus_score
                final_side = Side(stl.final_side)
                rationale = stl.rationale
                contradictions = stl.contradictions
                logger.info(
                    "Council[%s] STL-calibrated → %s  conf=%.1f%%  consensus=%.2f  contradictions=%d",
                    symbol, final_side.value, stl.confidence_pct, consensus_score, len(contradictions),
                )
            else:
                # Fallback: original NIM synthesis
                debate_summary = (
                    f"Bull says (score={bull_arg.quantitative_score:.2f}):\n{bull_arg.argument}\n\n"
                    f"Bear says (score={bear_arg.quantitative_score:.2f}):\n{bear_arg.argument}\n\n"
                    f"Historical memory was injected as context for both analysts."
                )

                synthesis = await nim_json(
                    [
                        {"role": "system", "content": _SUPERVISOR_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{enriched_context}\n\nDebate:\n{debate_summary}\n\n"
                                "Apply the STL Protocol and produce the final decision JSON."
                            ),
                        },
                    ],
                    temperature=0.1,
                )

                bull_score = float(synthesis.get("bull_score", 0.5))
                bear_score = float(synthesis.get("bear_score", 0.5))
                consensus_score = float(synthesis.get("consensus_score", abs(bull_score - bear_score)))
                final_side = Side(synthesis.get("final_side", "HOLD"))
                rationale = synthesis.get("rationale", "")
                contradictions = []

            if stl is not None and stl.confidence_pct < 75.0 and final_side != Side.HOLD:
                logger.warning(
                    "STL override: confidence %.1f%% < 75%%, forcing HOLD", stl.confidence_pct,
                )
                final_side = Side.HOLD
                consensus_score = 0.0

        decision = CouncilDecision(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            bull_score=bull_score,
            bear_score=bear_score,
            consensus_score=consensus_score,
            final_side=final_side,
            debate_log=[bull_arg, bear_arg],
            rationale=rationale,
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            features=features,
        )

        cfg = get_settings()
        bus = await get_bus()
        await bus.publish(
            cfg.stream_signals, MsgType.COUNCIL_DECISION, decision.model_dump(mode="json")
        )
        COUNCIL_CONSENSUS.labels(symbol=symbol, side=final_side.value).observe(consensus_score)
        logger.info(
            "Council[%s] → %s  bull=%.2f bear=%.2f consensus=%.2f" + (
                "  STL-blocked=%.1f%%" if stl and stl.blocked else ""
            ),
            symbol,
            final_side.value,
            bull_score,
            bear_score,
            consensus_score,
            *(stl.confidence_pct,) if stl and stl.blocked else (),
        )
        return decision
