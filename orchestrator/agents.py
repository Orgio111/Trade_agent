"""
QUANTEX AI Agent System — Market analyst, risk guardian, and execution agents
with NIM-powered reasoning.
"""
import json
from dataclasses import dataclass, field
from typing import Optional, Union, AsyncGenerator
from .nim_client import NIMOrchestrator
from .strategy import Signal
import aiohttp


@dataclass
class AgentOpinion:
    agent_id: str
    signal: str  # "long", "short", "hold"
    confidence: float
    reasoning: str
    model_used: str = ""
    metadata: dict = field(default_factory=dict)


class MarketAnalystAgent:
    """Analyzes market data using LLM reasoning."""

    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "market_analyst"

    async def analyze(self, market_context: dict) -> AgentOpinion:
        prompt = f"""
        You are a professional crypto market analyst. Analyze this market context:

        Symbol: {market_context.get('symbol', 'BTC/USDT')}
        Current Price: ${market_context.get('price', 0):.2f}
        24h Change: {market_context.get('change_24h', 0):.2f}%
        RSI(14): {market_context.get('rsi', 50):.1f}
        Volume Ratio (vs 20 SMA): {market_context.get('vol_ratio', 1.0):.2f}x
        Market Regime: {market_context.get('regime', 'unknown')}

        Key Levels:
        - Support: ${market_context.get('support', 0):.2f}
        - Resistance: ${market_context.get('resistance', 0):.2f}

        Provide a brief trading signal analysis. Output JSON:
        {{"signal": "long|short|hold", "confidence": 0.0-1.0, "reasoning": "brief rationale"}}
        """
        try:
            response = await self.nim.route_inference(
                "reasoning",
                [{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            result = json.loads(response)
            return AgentOpinion(
                agent_id=self.agent_id,
                signal=result["signal"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                model_used="deepseek-r1",
            )
        except Exception as e:
            return AgentOpinion(
                agent_id=self.agent_id,
                signal="hold",
                confidence=0.0,
                reasoning=f"LLM error: {e}",
            )


class RiskGuardianAgent:
    """Evaluates risk of proposed trades with veto power."""

    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "risk_guardian"

    async def evaluate(
        self,
        signal: Signal,
        portfolio: dict,
    ) -> AgentOpinion:
        prompt = f"""
        You are a Risk Guardian. Evaluate this proposed trade:

        Signal: {signal.direction} | Confidence: {signal.confidence:.2f}
        Entry: ${signal.entry_price:.2f} | SL: ${signal.stop_loss:.2f}
        RR: {self._calc_rr(signal):.2f}

        Portfolio Status:
        - Balance: ${portfolio.get('balance', 0):.2f}
        - Open Positions: {portfolio.get('open_positions', 0)}
        - Drawdown: {portfolio.get('drawdown', 0):.2%}
        - Consecutive Losses: {portfolio.get('consecutive_losses', 0)}

        Rules:
        - VETO if drawdown > 15%
        - VETO if 3+ consecutive losses
        - VETO if R/R < 1.5

        Output JSON:
        {{"approved": bool, "reason": "", "max_leverage": 1-10, "size_modifier": 0.1-1.0}}
        """
        try:
            response = await self.nim.route_inference(
                "reasoning",
                [{"role": "user", "content": prompt}],
                temperature=0.05,
            )
            result = json.loads(response)
            return AgentOpinion(
                agent_id=self.agent_id,
                signal=signal.direction if result["approved"] else "hold",
                confidence=result.get("size_modifier", 1.0),
                reasoning=result["reason"],
            )
        except Exception as e:
            return AgentOpinion(
                agent_id=self.agent_id,
                signal="hold",
                confidence=0.5,
                reasoning=f"Risk check error: {e}",
            )

    @staticmethod
    def _calc_rr(signal: Signal) -> float:
        if not signal.entry_price or not signal.stop_loss:
            return 0.0
        if signal.direction == "long":
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profits[0]["price"] - signal.entry_price if signal.take_profits else risk * 2
        else:
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profits[0]["price"] if signal.take_profits else risk * 2
        return reward / risk if risk > 0 else 0.0


class DeepSeekAnalysisAgent:
    """Enhanced DeepSeek R1 analysis agent with streaming reasoning and multi-source context."""

    def __init__(self, nim: NIMOrchestrator):
        self.nim = nim
        self.agent_id = "deepseek_analyst"

    async def analyze(
        self,
        symbol: str,
        price: float,
        indicators: dict,
        regime: str,
        sentiment: Optional[dict] = None,
        memory_context: Optional[str] = None,
        stream: bool = False,
    ) -> Union[AgentOpinion, AsyncGenerator]:
        """
        Deep market analysis using DeepSeek R1 reasoning.
        When stream=True, returns an async generator of reasoning tokens.
        """
        parts = [
            "You are QUANTEX DeepSeek R1 Trading Analyst — a professional quantitative analyst.",
            "",
            "## Market Context",
            f"Symbol: {symbol}",
            f"Current Price: ${price:.2f}",
            f"Market Regime: {regime}",
            "",
            "## Technical Indicators",
        ]
        for key, val in indicators.items():
            parts.append(f"  {key}: {val}")

        parts.append("")
        parts.append("## Sentiment Data")
        if sentiment:
            for key, val in sentiment.items():
                parts.append(f"  {key}: {val}")
        else:
            parts.append("  No sentiment data available")

        if memory_context:
            parts.append("")
            parts.append("## Similar Historical Patterns")
            parts.append(memory_context)

        parts.extend([
            "",
            "## Analysis Requirements",
            "1. Evaluate the current technical setup (trend, momentum, volatility)",
            "2. Assess if the market regime supports the trade direction",
            "3. Check sentiment alignment (fear = potential reversal, greed = caution)",
            "4. Identify key support and resistance levels",
            "5. Provide a clear signal with confidence score",
            "",
            "## Output Format",
            'Respond with ONLY valid JSON:',
            '{',
            '    "signal": "long|short|hold",',
            '    "confidence": 0.0-1.0,',
            '    "reasoning": "concise rationale",',
            '    "key_levels": {"support": 0.0, "resistance": 0.0},',
            '    "regime_alignment": "aligned|neutral|conflicting",',
            '    "risk_factors": ["factor1", "factor2"]',
            '}',
        ])
        prompt = "\n".join(parts)

        try:
            response = await self.nim.route_inference(
                "reasoning",
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=stream,
            )

            if stream:
                return response  # Returns async generator

            result = json.loads(response)
            return AgentOpinion(
                agent_id=self.agent_id,
                signal=result.get("signal", "hold"),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
                model_used="deepseek-r1",
                metadata={
                    "key_levels": result.get("key_levels", {}),
                    "regime_alignment": result.get("regime_alignment", "neutral"),
                    "risk_factors": result.get("risk_factors", []),
                },
            )
        except Exception as e:
            return AgentOpinion(
                agent_id=self.agent_id,
                signal="hold",
                confidence=0.0,
                reasoning=f"Analysis error: {e}",
            )


class SentimentAgent:
    """Analyzes market sentiment from Fear & Greed and social sources."""

    async def analyze(self) -> AgentOpinion:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.alternative.me/fng/?limit=1") as resp:
                    data = await resp.json()
                    fng_value = int(data["data"][0]["value"])

            if fng_value < 25:
                signal = "long"
                confidence = 0.7
                reasoning = f"Extreme Fear ({fng_value}) — potential buy opportunity"
            elif fng_value > 75:
                signal = "short"
                confidence = 0.6
                reasoning = f"Extreme Greed ({fng_value}) — market may be overbought"
            else:
                signal = "hold"
                confidence = 0.4
                reasoning = f"Neutral sentiment ({fng_value})"

            return AgentOpinion(
                agent_id="sentiment_agent",
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                metadata={"fear_greed": fng_value},
            )
        except Exception as e:
            return AgentOpinion(
                agent_id="sentiment_agent",
                signal="hold",
                confidence=0.3,
                reasoning=f"Sentiment fetch error: {e}",
            )



