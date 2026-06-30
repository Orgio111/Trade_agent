"""AutoGen Multi-Agent Trading Team — Debate + Consensus Engine.

Produces structured JSON trading signals via 5-agent GroupChat debate,
then hooks into LangGraph pipeline via autogen_to_langgraph().

Architecture (agents.md spec):
    MarketFeed (proxy)
        → QuantAgent → PatternAgent → MacroAgent
        → RiskAgent (hard filter)
        → Coordinator (final JSON decision)

Output format (strict):
    {
        "action": "BUY" | "SELL" | "HOLD",
        "confidence": float 0-1,
        "entry_reason": [str, ...],
        "risk": {"approved": bool, "notes": str}
    }

Models (RTX 4050 6GB — Ollama sequential):
    Coordinator : phi3:mini       (2.2GB)
    QuantAgent  : qwen2.5:3b     (1.9GB)
    PatternAgent: moondream      (1.7GB)
    MacroAgent  : qwen2.5:3b    (shared with Quant)
    RiskAgent    : qwen2:1.5b    (0.9GB)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from pydantic import BaseModel, Field

logger = logging.getLogger("quantex.autogen_team")

# ── Ollama config ──────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # Ollama doesn't need a real key

# Model assignments — fits RTX 4050 6GB with sequential loading
MODEL_COORDINATOR = "phi3:mini"
MODEL_QUANT = "qwen2.5:3b"
MODEL_PATTERN = "moondream"
MODEL_MACRO = "qwen2.5:3b"
MODEL_RISK = "qwen2:1.5b"


def _ollama_config(model: str, timeout: int = 60) -> dict:
    """Build AutoGen LLM config for an Ollama-hosted model.

    AG2 (v0.14+) requires api_type='ollama' or 'openai' for Ollama endpoints.
    Using 'ollama' tag with base_url pointing to Ollama's OpenAI-compatible API.
    """
    return {
        "config_list": [
            {
                "model": model,
                "base_url": OLLAMA_BASE_URL,
                "api_key": OLLAMA_API_KEY,
                "api_type": "ollama",
                "timeout": timeout,
            }
        ]
    }


# ── Pydantic output models ────────────────────────────────────

class RiskDecision(BaseModel):
    approved: bool = False
    notes: str = ""


class TradingSignal(BaseModel):
    """Structured output from Coordinator — the consensus signal."""
    action: str = Field(pattern="^(BUY|SELL|HOLD)$")
    confidence: float = Field(ge=0.0, le=1.0)
    entry_reason: list[str] = Field(default_factory=list)
    risk: RiskDecision = Field(default_factory=RiskDecision)


# ── System prompts (agents.md sections 4.1–4.5) ──────────────

PROMPT_COORDINATOR = """You are the final decision maker in a hedge-fund trading AI team.

Rules:
- You DO NOT hallucinate data.
- You only decide based on other agents' analysis.
- Output MUST be structured JSON — NOTHING else, no markdown, no explanation:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0-1>,
  "entry_reason": [<string>, ...],
  "risk": {
    "approved": <true|false>,
    "notes": "<string>"
  }
}

If RiskAgent says NOT approved → set action to "HOLD" and risk.approved to false.
Return ONLY the JSON object, no other text."""

PROMPT_QUANT = """You are a quantitative trading analyst.

Focus:
- Statistical patterns (mean reversion, momentum)
- Volatility analysis (ATR, Bollinger squeeze)
- Probability of breakout
- Volume profile anomalies

Return structured insights only. Be concise and data-driven.
End your analysis with a one-line SUMMARY: <your bias and confidence>."""

PROMPT_PATTERN = """You analyze chart structure from candlestick data and VLM output.

Detect:
- Trend direction (up/down/sideways)
- Support / resistance levels
- Breakout patterns (ascending triangle, bull flag, etc.)
- Candlestick patterns (engulfing, doji, hammer)

Be strict and visual-based reasoning only.
End with SUMMARY: <pattern name and directional bias>."""

PROMPT_MACRO = """You analyze macroeconomic + sentiment context.

Consider:
- News sentiment (risk-on / risk-off)
- Liquidity conditions
- Market regime (trending / ranging / volatile)
- Correlations (DXY, yields, equities)

If no real news data is provided, state "NO_NEWS_DATA" and focus on market microstructure.
End with SUMMARY: <macro bias — bullish/bearish/neutral and key factor>."""

PROMPT_RISK = """You are a strict risk manager — a HARD FILTER.

Rules:
- NEVER allow high-risk trades blindly
- Check: drawdown, volatility, leverage, position concentration
- You can override BUY/SELL → HOLD
- If confidence < 0.5 → REJECT
- If volatility extreme → REJECT
- If conflicting signals → REJECT

Output ONLY this JSON:
{
  "approved": true|false,
  "notes": "<brief reason>"
}

No other text."""


# ── Agent factory ──────────────────────────────────────────────

def create_trading_team(
    ollama_base_url: str = OLLAMA_BASE_URL,
    max_round: int = 6,
) -> "TradingTeam":
    """Create the full AutoGen trading team with all 5 agents + MarketFeed proxy."""
    return TradingTeam(ollama_base_url=ollama_base_url, max_round=max_round)


class TradingTeam:
    """AutoGen GroupChat-based multi-agent trading team.

    Usage:
        team = TradingTeam()
        result = team.run_trading_cycle(candle_data={...}, vlm_output={...})
        signal = autogen_to_langgraph(result)
    """

    def __init__(
        self,
        ollama_base_url: str = OLLAMA_BASE_URL,
        max_round: int = 6,
    ):
        self.ollama_base_url = ollama_base_url
        self.max_round = max_round
        self._build_agents()
        self._build_groupchat()

    # ── Agent construction ─────────────────────────────────────

    def _build_agents(self):
        """Build all 5 AssistantAgents + 1 UserProxyAgent per agents.md."""

        # 4.1 Coordinator — final decision maker
        self.coordinator = AssistantAgent(
            name="Coordinator",
            system_message=PROMPT_COORDINATOR,
            llm_config=_ollama_config(MODEL_COORDINATOR),
        )

        # 4.2 Quant Agent
        self.quant_agent = AssistantAgent(
            name="QuantAgent",
            system_message=PROMPT_QUANT,
            llm_config=_ollama_config(MODEL_QUANT),
        )

        # 4.3 Pattern / VLM Agent
        self.pattern_agent = AssistantAgent(
            name="PatternAgent",
            system_message=PROMPT_PATTERN,
            llm_config=_ollama_config(MODEL_PATTERN),
        )

        # 4.4 Macro / News Agent
        self.macro_agent = AssistantAgent(
            name="MacroAgent",
            system_message=PROMPT_MACRO,
            llm_config=_ollama_config(MODEL_MACRO),
        )

        # 4.5 Risk Agent (HARD FILTER)
        self.risk_agent = AssistantAgent(
            name="RiskAgent",
            system_message=PROMPT_RISK,
            llm_config=_ollama_config(MODEL_RISK),
        )

        # 5. MarketFeed proxy (agents.md section 5)
        self.market_feed = UserProxyAgent(
            name="MarketFeed",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False,
            llm_config=False,
        )

    def _build_groupchat(self):
        """Build GroupChat + GroupChatManager (agents.md section 6).

        Speaker flow enforced: MarketFeed → Quant → Pattern → Macro → Risk → Coordinator.
        This ensures every agent speaks exactly once per cycle.
        """
        # Ordered agent list (execution sequence per agents.md)
        agent_list = [
            self.market_feed,    # 0: Initiator — sends market data
            self.quant_agent,     # 1: Stats + volatility
            self.pattern_agent,   # 2: Chart + VLM
            self.macro_agent,     # 3: News + sentiment
            self.risk_agent,      # 4: Hard filter
            self.coordinator,     # 5: Final JSON consensus
        ]
        agent_names = [a.name for a in agent_list]

        # Build explicit speaker transitions: each agent → next in chain
        allowed_transitions = {}
        for i, agent in enumerate(agent_list):
            # Each agent can only pass to the next one (or self for retry)
            next_agents = [agent_list[i + 1]] if i < len(agent_list) - 1 else []
            allowed_transitions[agent] = next_agents

        self.groupchat = GroupChat(
            agents=agent_list,
            messages=[],
            max_round=self.max_round,
            speaker_selection_method="auto",
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
            allow_repeat_speaker=None,
            send_introductions=True,
            select_speaker_auto_verbose=False,
        )
        self.manager = GroupChatManager(
            groupchat=self.groupchat,
            llm_config=_ollama_config(MODEL_COORDINATOR),
        )

    # ── Main execution (agents.md section 7) ───────────────────

    async def async_run_trading_cycle(
        self,
        candle_data: dict[str, Any],
        vlm_output: dict[str, Any] | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Async wrapper — runs sync trading cycle in thread pool."""
        import asyncio
        return await asyncio.to_thread(
            self.run_trading_cycle, candle_data, vlm_output, market_context,
        )

    def run_trading_cycle(
        self,
        candle_data: dict[str, Any],
        vlm_output: dict[str, Any] | None = None,
        market_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one trading cycle — full agent debate → consensus.

        Args:
            candle_data: Latest candle + history summary.
                Expected keys: symbol, price, ohlcv, indicators (rsi, macd, atr, etc.)
            vlm_output: VLM chart analysis result (trend, pattern, support/resistance).
            market_context: Optional macro context (news, sentiment, regime).

        Returns:
            dict with the parsed TradingSignal JSON from Coordinator.
        """
        t0 = time.perf_counter()

        # Build the input message for MarketFeed (section 7 spec)
        message = self._build_market_message(candle_data, vlm_output, market_context)

        logger.info("Starting trading cycle — %d agents, max_round=%d",
                    len(self.groupchat.agents), self.max_round)

        try:
            # MarketFeed initiates chat → triggers GroupChat debate
            self.market_feed.initiate_chat(
                self.manager,
                message=message,
            )

            # Extract Coordinator's final message from GroupChat history
            result = self._extract_final_signal()
        except Exception as e:
            logger.error("Trading cycle failed: %s", e)
            result = TradingSignal(
                action="HOLD",
                confidence=0.0,
                entry_reason=[f"error: {e}"],
                risk=RiskDecision(approved=False, notes=f"cycle error: {e}"),
            ).model_dump()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Trading cycle completed in %.0fms — action=%s",
                    elapsed_ms, result.get("action", "?"))

        return result

    def _build_market_message(
        self,
        candle_data: dict,
        vlm_output: dict | None,
        market_context: dict | None,
    ) -> str:
        """Build the input message that MarketFeed sends to the group."""
        parts = ["NEW MARKET DATA:\n"]

        # Candle data
        parts.append(f"Candle:\n{json.dumps(candle_data, indent=2, default=str)}\n")

        # VLM analysis (if available — from langgraph_pipeline vlm_agent node)
        if vlm_output:
            parts.append(f"VLM Analysis:\n{json.dumps(vlm_output, indent=2, default=str)}\n")
        else:
            parts.append("VLM Analysis: NOT_AVAILABLE\n")

        # Macro context
        if market_context:
            parts.append(f"Macro Context:\n{json.dumps(market_context, indent=2, default=str)}\n")

        parts.append(
            "\nTASK: All agents analyze and produce a final trading decision.\n"
            "QuantAgent → PatternAgent → MacroAgent → RiskAgent (filter) → Coordinator (final JSON)."
        )
        return "".join(parts)

    def _extract_final_signal(self) -> dict[str, Any]:
        """Parse Coordinator's final JSON from the GroupChat message history."""
        # Walk backwards through groupchat messages to find Coordinator's JSON
        messages = self.groupchat.messages

        for msg in reversed(messages):
            name = msg.get("name", "")
            content = msg.get("content", "")

            # Look for Coordinator's response
            if name != "Coordinator" and msg.get("role") != "assistant":
                continue

            parsed = _try_parse_json(content)
            if parsed:
                # Validate it's a trading signal
                try:
                    signal = TradingSignal(**parsed)
                    return signal.model_dump()
                except Exception:
                    continue

        # Fallback: scan ALL messages for any valid JSON signal
        for msg in reversed(messages):
            content = msg.get("content", "")
            parsed = _try_parse_json(content)
            if parsed and "action" in parsed:
                try:
                    signal = TradingSignal(**parsed)
                    return signal.model_dump()
                except Exception:
                    continue

        # Absolute fallback
        logger.warning("No valid signal found in GroupChat — returning HOLD")
        return TradingSignal(
            action="HOLD",
            confidence=0.0,
            entry_reason=["no consensus reached"],
            risk=RiskDecision(approved=False, notes="no valid signal extracted"),
        ).model_dump()


# ── JSON parser (robust — handles markdown, extra text) ────────

def _try_parse_json(text: str) -> dict | None:
    """Try to extract a JSON object from text.

    Handles: raw JSON, ```json blocks, mixed text+JSON.
    """
    if not text:
        return None

    # Try direct parse
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    markdown_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if markdown_match:
        try:
            result = json.loads(markdown_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ── LangGraph integration hook (agents.md section 10) ─────────

def autogen_to_langgraph(result: dict[str, Any]) -> dict[str, Any]:
    """Convert AutoGen team output to LangGraph-compatible signal.

    If risk.approved is False → force HOLD.
    This is the hard filter that prevents risky trades from reaching execution.
    """
    risk = result.get("risk", {})
    approved = risk.get("approved", False) if isinstance(risk, dict) else False

    if approved:
        return {
            "action": result.get("action", "HOLD"),
            "confidence": result.get("confidence", 0.0),
            "entry_reason": result.get("entry_reason", []),
            "risk_notes": risk.get("notes", ""),
        }
    else:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "entry_reason": result.get("entry_reason", []),
            "risk_notes": risk.get("notes", "risk rejected — forced HOLD"),
        }


# ── CLI entry point ────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    # Demo candle data
    demo_candle = {
        "symbol": "BTCUSDT",
        "price": 61250.0,
        "ohlcv": {
            "open": 61000.0, "high": 61380.0,
            "low": 60900.0, "close": 61250.0,
            "volume": 1523.5,
        },
        "indicators": {
            "rsi_14": 62.3,
            "macd": {"macd": 45.2, "signal": 38.1, "histogram": 7.1},
            "atr_14": 580.0,
            "ema_9": 61150.0,
            "ema_21": 60800.0,
            "bb_width": 0.032,
        },
        "regime": "trending",
    }

    demo_vlm = {
        "trend": "bullish",
        "pattern": "ascending_triangle",
        "support": 60500.0,
        "resistance": 62000.0,
        "confidence": 0.78,
    }

    print("=" * 60)
    print("QUANTEX AutoGen Trading Team — Demo Run")
    print("=" * 60)

    team = create_trading_team()
    result = team.run_trading_cycle(
        candle_data=demo_candle,
        vlm_output=demo_vlm,
    )

    print("\n" + "=" * 60)
    print("RAW RESULT:")
    print(json.dumps(result, indent=2))

    langgraph_signal = autogen_to_langgraph(result)
    print("\nLANGGRAPH SIGNAL:")
    print(json.dumps(langgraph_signal, indent=2))
