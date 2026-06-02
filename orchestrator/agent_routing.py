"""
QUANTEX Agent-Model Architecture — Intelligent model routing and agent mapping.

Maps each agent type to its optimal model based on:
  - Task requirements (reasoning depth, latency sensitivity, context needs)
  - Cost optimization (free tier vs paid tier routing)
  - Fallback chains for reliability

Agent -> Model Mapping:
  - Supervisor/Planner: DeepSeek R1/V4 Flash (best reasoning)
  - Market Analyst: DeepSeek Chat V3 (fast analysis)
  - Risk Guardian: Qwen3 (structured logic, reliability)
  - Execution: Mistral Small (low latency)
  - Sentiment: Llama 4 Scout (good text understanding)
  - Memory: Gemma (lightweight, efficient)
  - Edge inference: Phi (tiny, runs anywhere)
  - Scalping: Nemotron Nano (ultra-fast)
  - Swing: Gemini Flash (balanced)
"""

import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str  # "nvidia_nim", "openrouter", "local"
    api_model_id: str
    context_window: int
    reasoning_score: float  # 0-1, how good at reasoning
    speed_score: float  # 0-1, how fast (higher = faster)
    cost_per_1k_tokens: float = 0.0  # USD
    free_tier: bool = True
    supports_streaming: bool = True
    supports_tools: bool = False
    reliability: float = 0.95  # 0-1 success rate


@dataclass
class AgentModelMap:
    """Maps an agent role to its model configuration."""
    agent_id: str
    agent_name: str
    primary_model: ModelConfig
    fallback_model: Optional[ModelConfig] = None
    tertiary_model: Optional[ModelConfig] = None
    latency_sensitive: bool = False
    max_retries: int = 2


# ── Model Definitions ──────────────────────────────────────

# NVIDIA NIM models (cloud, free credits available)
NIM_REASONING = ModelConfig(
    name="DeepSeek V4 Flash",
    provider="nvidia_nim",
    api_model_id="deepseek/deepseek-v4-flash",
    context_window=1_000_000,
    reasoning_score=0.95,
    speed_score=0.7,
    free_tier=True,
    supports_streaming=True,
)

NIM_CODING = ModelConfig(
    name="Qwen3 Coder 480B",
    provider="nvidia_nim",
    api_model_id="qwen/qwen3-coder-480b-a35b-instruct",
    context_window=128_000,
    reasoning_score=0.85,
    speed_score=0.5,
    free_tier=True,
)

NIM_EMBEDDING = ModelConfig(
    name="NV-EmbedQA-E5",
    provider="nvidia_nim",
    api_model_id="nvidia/nv-embedqa-e5-v5",
    context_window=512,
    reasoning_score=0.2,
    speed_score=0.9,
    free_tier=True,
)

NIM_MULTIMODAL = ModelConfig(
    name="Nemotron Nano Omni",
    provider="nvidia_nim",
    api_model_id="nvidia/nemotron-nano-omni",
    context_window=128_000,
    reasoning_score=0.5,
    speed_score=0.8,
    free_tier=True,
)

# OpenRouter free models
OR_DEEPSEEK_R1 = ModelConfig(
    name="DeepSeek R1 (Free)",
    provider="openrouter",
    api_model_id="deepseek/deepseek-r1:free",
    context_window=128_000,
    reasoning_score=0.98,
    speed_score=0.3,
    free_tier=True,
)

OR_QWQ_32B = ModelConfig(
    name="QwQ-32B (Free)",
    provider="openrouter",
    api_model_id="qwen/qwq-32b:free",
    context_window=128_000,
    reasoning_score=0.90,
    speed_score=0.4,
    free_tier=True,
)

OR_LLAMA_4_SCOUT = ModelConfig(
    name="Llama 4 Scout (Free)",
    provider="openrouter",
    api_model_id="meta-llama/llama-4-scout:free",
    context_window=256_000,
    reasoning_score=0.60,
    speed_score=0.75,
    free_tier=True,
)

OR_MISTRAL_SMALL = ModelConfig(
    name="Mistral Small 3.1 (Free)",
    provider="openrouter",
    api_model_id="mistralai/mistral-small-3.1-24b-instruct:free",
    context_window=32_000,
    reasoning_score=0.55,
    speed_score=0.85,
    free_tier=True,
)

OR_GEMMA_3 = ModelConfig(
    name="Gemma 3 (Free)",
    provider="openrouter",
    api_model_id="google/gemma-3-27b-it:free",
    context_window=32_000,
    reasoning_score=0.50,
    speed_score=0.80,
    free_tier=True,
)

OR_PHI_4 = ModelConfig(
    name="Phi-4 (Free)",
    provider="openrouter",
    api_model_id="microsoft/phi-4:free",
    context_window=16_000,
    reasoning_score=0.45,
    speed_score=0.90,
    free_tier=True,
)

# Local models (self-hosted)
LOCAL_MISTRAL = ModelConfig(
    name="Mistral 7B (Local)",
    provider="local",
    api_model_id="mistral-7b",
    context_window=8_000,
    reasoning_score=0.30,
    speed_score=0.95,
    free_tier=True,
)


# ── Agent -> Model Mapping ──────────────────────────────────

AGENT_MODEL_MAP: dict[str, AgentModelMap] = {
    # Supervisor/Planner — needs the best reasoning
    "supervisor": AgentModelMap(
        agent_id="supervisor",
        agent_name="Supervisor Agent",
        primary_model=NIM_REASONING,
        fallback_model=OR_DEEPSEEK_R1,
        tertiary_model=OR_QWQ_32B,
        max_retries=3,
    ),
    # Market Analyst — good reasoning, moderate speed
    "market_analyst": AgentModelMap(
        agent_id="market_analyst",
        agent_name="Market Analyst",
        primary_model=OR_DEEPSEEK_R1,
        fallback_model=NIM_REASONING,
        max_retries=2,
    ),
    # Risk Guardian — structured, reliable, fast
    "risk_guardian": AgentModelMap(
        agent_id="risk_guardian",
        agent_name="Risk Guardian",
        primary_model=OR_QWQ_32B,
        fallback_model=OR_LLAMA_4_SCOUT,
        latency_sensitive=True,
        max_retries=2,
    ),
    # Execution — lowest latency priority
    "execution": AgentModelMap(
        agent_id="execution",
        agent_name="Execution Agent",
        primary_model=OR_MISTRAL_SMALL,
        fallback_model=OR_PHI_4,
        tertiary_model=LOCAL_MISTRAL,
        latency_sensitive=True,
    ),
    # Sentiment — good text understanding
    "sentiment": AgentModelMap(
        agent_id="sentiment",
        agent_name="Sentiment Agent",
        primary_model=OR_LLAMA_4_SCOUT,
        fallback_model=OR_GEMMA_3,
    ),
    # Scalping — ultra-fast, low latency
    "scalping": AgentModelMap(
        agent_id="scalping",
        agent_name="Scalping Agent",
        primary_model=OR_MISTRAL_SMALL,
        fallback_model=OR_PHI_4,
        latency_sensitive=True,
    ),
    # Swing — balanced reasoning
    "swing": AgentModelMap(
        agent_id="swing",
        agent_name="Swing Agent",
        primary_model=NIM_REASONING,
        fallback_model=OR_LLAMA_4_SCOUT,
    ),
    # DeepSeek analysis — deep reasoning
    "deepseek": AgentModelMap(
        agent_id="deepseek",
        agent_name="DeepSeek Analysis Agent",
        primary_model=OR_DEEPSEEK_R1,
        fallback_model=NIM_REASONING,
        tertiary_model=OR_QWQ_32B,
        max_retries=3,
    ),
    # Memory curation — lightweight
    "memory": AgentModelMap(
        agent_id="memory",
        agent_name="Memory Curator",
        primary_model=OR_GEMMA_3,
        fallback_model=OR_PHI_4,
    ),
    # Anomaly detection — fast pattern matching
    "anomaly": AgentModelMap(
        agent_id="anomaly",
        agent_name="Anomaly Detector",
        primary_model=OR_MISTRAL_SMALL,
        fallback_model=OR_PHI_4,
        latency_sensitive=True,
    ),
    # Regime classifier — fast classification
    "regime": AgentModelMap(
        agent_id="regime",
        agent_name="Regime Classifier",
        primary_model=OR_LLAMA_4_SCOUT,
        fallback_model=OR_GEMMA_3,
    ),
}


class AgentModelRouter:
    """
    Intelligent model router for agents.

    Selects the optimal model based on:
      - Task type (reasoning-heavy vs latency-sensitive)
      - Cost constraints (try free tier first)
      - Availability (fallback on failure)
      - Performance tracking (prefer reliable routes)
    """

    def __init__(self):
        self._route_stats: dict[str, dict] = {}
        self._fallback_cache: dict[str, bool] = {}

    def get_model_chain(self, agent_id: str, task_type: str = "reasoning") -> list[ModelConfig]:
        """
        Get the ordered model chain for an agent.

        Returns [primary, fallback, tertiary] models to try in order.
        """
        agent_map = AGENT_MODEL_MAP.get(agent_id)
        if not agent_map:
            # Default chain for unknown agents
            return [NIM_REASONING, OR_MISTRAL_SMALL]

        models = [agent_map.primary_model]
        if agent_map.fallback_model:
            models.append(agent_map.fallback_model)
        if agent_map.tertiary_model:
            models.append(agent_map.tertiary_model)

        return models

    def select_model(self, agent_id: str, task_type: str = "reasoning",
                     latency_budget_ms: Optional[float] = None) -> ModelConfig:
        """
        Select the best model for a task given constraints.

        Args:
            agent_id: Which agent is making the request
            task_type: "reasoning", "fast", "analysis", "embedding"
            latency_budget_ms: Max acceptable latency in ms

        Returns:
            ModelConfig to use
        """
        agent_map = AGENT_MODEL_MAP.get(agent_id)
        if not agent_map:
            return NIM_REASONING

        # For latency-sensitive agents, prioritize speed
        if agent_map.latency_sensitive or latency_budget_ms is not None:
            primary = agent_map.primary_model
            # Check if primary is fast enough
            if primary.speed_score < 0.7 and agent_map.fallback_model:
                fallback = agent_map.fallback_model
                if fallback.speed_score > primary.speed_score:
                    return fallback
            return primary

        return agent_map.primary_model

    def record_result(self, agent_id: str, model_name: str, success: bool,
                      latency_ms: float):
        """Record a routing result for performance tracking."""
        key = f"{agent_id}:{model_name}"
        if key not in self._route_stats:
            self._route_stats[key] = {"attempts": 0, "successes": 0, "latencies": []}
        self._route_stats[key]["attempts"] += 1
        if success:
            self._route_stats[key]["successes"] += 1
        self._route_stats[key]["latencies"].append(latency_ms)

    def get_route_stats(self, agent_id: Optional[str] = None) -> dict:
        """Get routing statistics."""
        if agent_id:
            prefix = f"{agent_id}:"
            return {k: v for k, v in self._route_stats.items() if k.startswith(prefix)}
        return dict(self._route_stats)

    def get_agent_model_summary(self) -> list[dict]:
        """Get a summary of all agent-to-model mappings."""
        summary = []
        for agent_id, agent_map in AGENT_MODEL_MAP.items():
            summary.append({
                "agent_id": agent_id,
                "agent_name": agent_map.agent_name,
                "primary_model": agent_map.primary_model.name,
                "primary_provider": agent_map.primary_model.provider,
                "primary_free": agent_map.primary_model.free_tier,
                "fallback_model": agent_map.fallback_model.name if agent_map.fallback_model else None,
                "latency_sensitive": agent_map.latency_sensitive,
                "reasoning_score": agent_map.primary_model.reasoning_score,
                "speed_score": agent_map.primary_model.speed_score,
            })
        return summary

    def get_free_model_list(self) -> dict[str, list[str]]:
        """Get all available free models grouped by capability."""
        models = {
            "reasoning": [],
            "fast": [],
            "coding": [],
            "embedding": [],
            "lightweight": [],
        }
        for name, config in [
            ("DeepSeek V4 Flash", NIM_REASONING),
            ("Qwen3 Coder 480B", NIM_CODING),
            ("NV-EmbedQA-E5", NIM_EMBEDDING),
            ("Nemotron Nano Omni", NIM_MULTIMODAL),
            ("DeepSeek R1", OR_DEEPSEEK_R1),
            ("QwQ-32B", OR_QWQ_32B),
            ("Llama 4 Scout", OR_LLAMA_4_SCOUT),
            ("Mistral Small 3.1", OR_MISTRAL_SMALL),
            ("Gemma 3", OR_GEMMA_3),
            ("Phi-4", OR_PHI_4),
        ]:
            if config.reasoning_score > 0.8:
                models["reasoning"].append(f"{name} ({config.provider})")
            if config.speed_score > 0.8:
                models["fast"].append(f"{name} ({config.provider})")
            if config.reasoning_score > 0.6 and config.speed_score > 0.6:
                models["coding"].append(f"{name} ({config.provider})")
            if "embed" in name.lower():
                models["embedding"].append(f"{name} ({config.provider})")
            if config.speed_score > 0.85 and config.reasoning_score < 0.6:
                models["lightweight"].append(f"{name} ({config.provider})")

        return models


class TaskRouter:
    """
    Routes tasks to the appropriate agent and model based on
    task characteristics, availability, and cost constraints.

    Task types:
      - "analysis": market analysis, needs reasoning -> market_analyst agent
      - "risk": risk assessment -> risk_guardian -> Mistral (fast)
      - "sentiment": sentiment analysis -> sentiment -> Llama
      - "execution": order execution -> execution -> Mistral (lowest latency)
      - "planning": strategy planning -> supervisor -> DeepSeek R1
      - "scalping": fast scalping -> scalping -> Mistral Small
      - "swing": swing analysis -> swing -> DeepSeek V4
      - "memory": memory operations -> memory -> Gemma/Phi
      - "anomaly": anomaly detection -> anomaly -> Mistral Small
      - "deepseek": deep reasoning -> deepseek -> DeepSeek R1
    """

    TASK_AGENT_MAP = {
        "analysis": "market_analyst",
        "risk": "risk_guardian",
        "sentiment": "sentiment",
        "execution": "execution",
        "planning": "supervisor",
        "scalping": "scalping",
        "swing": "swing",
        "memory": "memory",
        "anomaly": "anomaly",
        "deepseek": "deepseek",
        "regime": "regime",
    }

    def __init__(self):
        self.model_router = AgentModelRouter()

    def resolve(self, task_type: str,
                latency_budget_ms: Optional[float] = None) -> dict:
        """
        Resolve a task to its optimal agent and model.

        Returns:
            dict with agent_id, model_config, fallback_chain, reasoning
        """
        agent_id = self.TASK_AGENT_MAP.get(task_type, "market_analyst")
        model = self.model_router.select_model(agent_id, task_type, latency_budget_ms)
        model_chain = self.model_router.get_model_chain(agent_id, task_type)

        return {
            "agent_id": agent_id,
            "agent_name": AGENT_MODEL_MAP.get(agent_id, AgentModelMap("unknown", "Unknown", model)).agent_name,
            "model": {
                "name": model.name,
                "provider": model.provider,
                "api_id": model.api_model_id,
                "free_tier": model.free_tier,
                "reasoning_score": model.reasoning_score,
                "speed_score": model.speed_score,
                "context_window": model.context_window,
            },
            "fallback_chain": [
                {"name": m.name, "provider": m.provider, "free": m.free_tier}
                for m in model_chain[1:]
            ],
            "reasoning": (
                f"Task '{task_type}' -> Agent '{agent_id}' -> Model '{model.name}' "
                f"(reasoning={model.reasoning_score:.1f}, speed={model.speed_score:.1f}, "
                f"free={model.free_tier})"
            ),
        }

    def resolve_multi(self, tasks: list[tuple[str, Optional[float]]]) -> list[dict]:
        """Resolve multiple tasks at once."""
        return [self.resolve(task_type, latency) for task_type, latency in tasks]

    def get_default_system_prompt(self, agent_id: str) -> str:
        """Get the default system prompt for an agent type."""
        prompts = {
            "supervisor": (
                "You are the SUPERVISOR AGENT, the final decision-maker. "
                "You receive input from all sub-agents and must synthesize "
                "their analysis into a clear trade decision. Be conservative. "
                "Risk management is your top priority."
            ),
            "market_analyst": (
                "You are the MARKET ANALYST AGENT. Analyze market structure, "
                "trends, and price action. Identify key support/resistance, "
                "liquidity zones, and potential entry points. Be objective."
            ),
            "risk_guardian": (
                "You are the RISK GUARDIAN AGENT. Your only job is to prevent "
                "bad trades. Evaluate every signal for risk. Be conservative. "
                "When in doubt, say HOLD. You are the most important agent."
            ),
            "execution": (
                "You are the EXECUTION AGENT. Plan optimal order execution "
                "with minimal slippage. Consider TWAP, VWAP, iceberg orders. "
                "Be precise. Every basis point matters."
            ),
            "sentiment": (
                "You are the SENTIMENT AGENT. Analyze market sentiment from "
                "news, social media, and on-chain data. Identify crowd extremes. "
                "Contrarian signals are valuable."
            ),
            "scalping": (
                "You are the SCALPING AGENT. Fast micro-structure analysis. "
                "Look for immediate entries with tight stops. "
                "Be decisive and quick. This is NOT for swing trades."
            ),
            "swing": (
                "You are the SWING AGENT. Look for medium-term setups with "
                "good risk-reward ratios (>2:1). Patience is key. "
                "Higher timeframe analysis."
            ),
            "memory": (
                "You are the MEMORY CURATOR. Store and retrieve trade patterns "
                "from the vector database. Identify historically similar setups. "
                "Pattern recognition is your strength."
            ),
            "anomaly": (
                "You are the ANOMALY DETECTOR. Watch for market manipulation, "
                "spoofing, unusual order flow, and abnormal volatility. "
                "Flag anything suspicious immediately."
            ),
            "deepseek": (
                "You are the DEEPSEEK ANALYSIS AGENT. You have access to deep "
                "reasoning capabilities. Analyze market context thoroughly "
                "before making any recommendation. Think step by step."
            ),
        }
        return prompts.get(agent_id, "You are an AI trading agent. Analyze and respond.")

    def get_task_recommendation(self, context: dict) -> str:
        """
        Recommend which task type to use based on market context.

        Factors: volatility, timeframe, confidence, regime
        """
        volatility = context.get("volatility", 0.02)
        timeframe = context.get("timeframe", "1h")
        confidence = context.get("confidence", 0.5)
        regime = context.get("regime", "unknown")

        # High volatility + low confidence = check risk first
        if volatility > 0.03 and confidence < 0.5:
            return "risk"
        # Low volatility + short timeframe = scalping
        if volatility < 0.01 and timeframe in ("1m", "5m"):
            return "scalping"
        # Medium timeframe = swing analysis
        if timeframe in ("1h", "4h"):
            return "swing"
        # High confidence = deeper analysis
        if confidence > 0.7:
            return "deepseek"
        # Default
        return "analysis"
