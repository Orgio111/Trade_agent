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
    api_model_id="deepseek-ai/deepseek-v4-flash",
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

# Groq LPU models (ultra-fast inference)
GR_LLAMA3_8B = ModelConfig(
    name="Llama 3 8B (Groq)",
    provider="groq",
    api_model_id="llama3-8b-8192",
    context_window=8_192,
    reasoning_score=0.55,
    speed_score=0.95,
    free_tier=True,
)

GR_QWEN3_32B = ModelConfig(
    name="Qwen3 32B (Groq)",
    provider="groq",
    api_model_id="qwen/qwen3-32b",
    context_window=32_768,
    reasoning_score=0.85,
    speed_score=0.75,
    free_tier=True,
)

GR_LLAMA70B = ModelConfig(
    name="Llama 3.3 70B (Groq)",
    provider="groq",
    api_model_id="llama-3.3-70b-versatile",
    context_window=32_768,
    reasoning_score=0.92,
    speed_score=0.70,
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

# ── Agent -> Provider Chain Mapping ──────────────────────────
#
# Maps each agent to its optimal provider chain for the InferenceRouter.
# The router tries providers in order, falling through on failure.
#
# Agent type → [primary_provider, fallback_provider, tertiary_provider]
#
# Design principles:
#   - Scalping/Execution: Groq first (ultra-fast LPU inference, 800+ tok/s)
#   - Market Analyst/DeepSeek: NVIDIA NIM first (highest quality reasoning)
#   - Risk Guardian: OpenRouter first (fast structured output models like QwQ-32B)
#   - Sentiment/Regime: OpenRouter free tier (cheap, good enough for classification)
#   - All: OpenRouter as universal fallback (200+ models, highest availability)

AGENT_PROVIDER_CHAINS: dict[str, list[str]] = {
    # ═══ Reasoning agents — quality first, speed second ═══
    # Uses NVIDIA NIM for highest quality analysis, falls back to Groq, then OpenRouter
    "supervisor":        ["nvidia_nim", "groq", "openrouter"],
    "market_analyst":    ["nvidia_nim", "groq", "openrouter"],
    "deepseek_analyst":  ["nvidia_nim", "groq", "openrouter"],
    "swing_agent":       ["nvidia_nim", "groq", "openrouter"],

    # ═══ Ultra-fast agents — speed first, any provider ═══
    # Groq LPU inference (800+ tok/s) for latency-sensitive decisions
    "scalping_agent":    ["groq", "openrouter", "nvidia_nim"],
    "execution_agent":   ["groq", "openrouter", "nvidia_nim"],
    "anomaly_agent":     ["groq", "openrouter", "nvidia_nim"],

    # ═══ Risk & safety — structured, reliable, moderate speed ═══
    # Groq first for fast risk checks, then OpenRouter for thorough analysis
    "risk_guardian":     ["groq", "openrouter", "nvidia_nim"],

    # ═══ Classification — cheap, fast, low-stakes ═══
    # OpenRouter free tier is sufficient for classification tasks
    "sentiment_agent":   ["openrouter", "groq", "nvidia_nim"],
    "regime_agent":      ["openrouter", "groq", "nvidia_nim"],
    "memory":            ["openrouter", "groq", "nvidia_nim"],

    # ═══ Default fallback for any unregistered agent ═══
    "default":           ["groq", "nvidia_nim", "openrouter"],
}

# Agent → InferenceRouter task_type override
# Allows agents to request specific model capabilities independent of their prompt's task_type
AGENT_TASK_TYPE_OVERRIDES: dict[str, str] = {
    "scalping_agent":    "urgent",           # Ultra-low latency required
    "execution_agent":   "urgent",           # Ultra-low latency required
    "anomaly_agent":     "fast",             # Fast pattern matching
    "risk_guardian":     "reasoning",        # Needs careful reasoning
    "sentiment_agent":   "classification",   # Simple text classification
    "regime_agent":      "classification",   # Simple regime classification
    "memory":            "classification",   # Simple text operations
    "market_analyst":    "analysis",         # Deep market analysis
    "supervisor":        "reasoning",        # Complex multi-factor reasoning
    "deepseek_analyst":  "reasoning",        # Deep reasoning with R1
    "swing_agent":       "analysis",         # Medium-term market analysis
}


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
      - Adaptive performance tracking (real observed P50 latency reorders chains)

    Adaptive routing tracks per-agent per-provider latencies using exponential
    moving averages (EMA). After collecting enough samples, it automatically
    reorders provider chains so faster providers are tried first.

    Configuration:
      - EMA_ALPHA:          Smoothing factor (0.1 = smooth, 0.5 = responsive)
      - MIN_SAMPLES:        Minimum requests before adaptive reordering kicks in
      - MIN_SAMPLES_FAST:   Lower threshold for latency-sensitive agents
      - FAILURE_LATENCY_MS: Assumed latency when a provider fails (timeout * 2)
    """

    # EMA smoothing — lower = smoother (less noisy), higher = more responsive
    EMA_ALPHA = 0.2
    # Minimum observations before adaptive reordering activates
    MIN_SAMPLES = 10
    MIN_SAMPLES_FAST = 5  # Lower threshold for latency-sensitive agents
    # Penalty for failed requests (in ms) — used instead of actual latency
    FAILURE_LATENCY_MS = 30_000  # 30s

    def __init__(self):
        self._route_stats: dict[str, dict] = {}
        self._fallback_cache: dict[str, bool] = {}

        # ── Adaptive routing state ────────────────────────────────────────
        # Structure: {agent_id: {provider_name: {
        #     "ema_latency_ms": float,   # Exponential moving average of latencies
        #     "samples": int,             # Total observations
        #     "successes": int,           # Successful requests
        #     "failures": int,            # Failed requests
        #     "last_updated": float,      # Timestamp of last observation
        #     "p50_latency_ms": float,    # Raw P50 (for reference)
        #     "_raw": [float],            # Recent raw latencies (for P50 calc)
        # }}}
        self._perf: dict[str, dict[str, dict]] = {}

    # ── Model selection (static, from config) ────────────────────────────

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

    # ── Adaptive provider chain reordering ─────────────────────────────

    def record_provider_latency(
        self,
        agent_id: str,
        provider: str,
        latency_ms: float,
        success: bool = True,
    ):
        """
        Record a latency observation for an agent→provider pair.

        Updates the EMA-based latency tracking. Failed requests are recorded
        with a high latency penalty to deprioritize unreliable providers.

        Args:
            agent_id:  Which agent made the request
            provider:  Which provider was used ("groq", "nvidia_nim", "openrouter")
            latency_ms: Observed latency in milliseconds
            success:   Whether the request succeeded
        """
        # Ensure agent entry exists
        if agent_id not in self._perf:
            self._perf[agent_id] = {}

        # Ensure provider entry exists
        if provider not in self._perf[agent_id]:
            self._perf[agent_id][provider] = {
                "ema_latency_ms": 0.0,
                "samples": 0,
                "successes": 0,
                "failures": 0,
                "last_updated": 0.0,
                "p50_latency_ms": 0.0,
                "_raw": [],
            }

        perf = self._perf[agent_id][provider]

        # Use penalty latency for failures instead of the actual (missing) value
        effective_latency = latency_ms if success else max(latency_ms, self.FAILURE_LATENCY_MS)

        # Update EMA
        if perf["samples"] == 0:
            perf["ema_latency_ms"] = effective_latency
        else:
            perf["ema_latency_ms"] = (
                self.EMA_ALPHA * effective_latency
                + (1 - self.EMA_ALPHA) * perf["ema_latency_ms"]
            )

        perf["samples"] += 1
        if success:
            perf["successes"] += 1
        else:
            perf["failures"] += 1
        perf["last_updated"] = time.time()

        # Keep a sliding window of raw latencies for P50 calculation
        perf["_raw"].append(effective_latency)
        if len(perf["_raw"]) > 50:
            perf["_raw"] = perf["_raw"][-25:]  # Keep last 25

        # Recalculate P50 from recent raw samples
        if len(perf["_raw"]) >= 3:
            sorted_raw = sorted(perf["_raw"])
            perf["p50_latency_ms"] = sorted_raw[len(sorted_raw) // 2]

    def get_optimal_provider_chain(
        self,
        agent_id: str,
        base_chain: list[str] | None = None,
    ) -> list[str]:
        """
        Get an agent's provider chain sorted by observed P50 latency.

        Uses EMA-smoothed latency to reorder providers so the fastest
        (for this specific agent) is tried first. Falls back to the
        static `base_chain` if there aren't enough samples yet.

        Args:
            agent_id:   Which agent's chain to optimize
            base_chain: The static default chain (from AGENT_PROVIDER_CHAINS)
                        If None, uses the "default" chain.

        Returns:
            Provider names ordered by performance (fastest first)
        """
        if base_chain is None:
            base_chain = AGENT_PROVIDER_CHAINS.get(agent_id, AGENT_PROVIDER_CHAINS["default"])

        agent_perf = self._perf.get(agent_id, {})

        # Determine minimum samples threshold
        agent_map = AGENT_MODEL_MAP.get(agent_id)
        min_samples = self.MIN_SAMPLES_FAST if (agent_map and agent_map.latency_sensitive) else self.MIN_SAMPLES

        # Check if we have enough data to adapt
        has_enough_data = all(
            provider in agent_perf and agent_perf[provider]["samples"] >= min_samples
            for provider in base_chain
        )

        if not has_enough_data:
            return list(base_chain)

        # Sort by EMA latency (ascending) — fastest first
        def _sort_key(provider: str) -> float:
            if provider in agent_perf:
                return agent_perf[provider]["ema_latency_ms"]
            return float("inf")  # Unknown providers go last

        return sorted(base_chain, key=_sort_key)

    def get_provider_latency(self, agent_id: str, provider: str) -> float | None:
        """Get the EMA latency for a specific agent→provider pair."""
        return (
            self._perf.get(agent_id, {}).get(provider, {}).get("ema_latency_ms")
        )

    # ── Result tracking (legacy) ────────────────────────────────────────

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

    # ── Stats & monitoring ────────────────────────────────────────────

    def get_route_stats(self, agent_id: Optional[str] = None) -> dict:
        """Get routing statistics."""
        if agent_id:
            prefix = f"{agent_id}:"
            return {k: v for k, v in self._route_stats.items() if k.startswith(prefix)}
        return dict(self._route_stats)

    def get_adaptive_routing_summary(self, agent_id: Optional[str] = None) -> dict:
        """
        Get adaptive routing performance data.

        Returns per-provider EMA latencies, sample counts, and whether
        each agent has enough data for adaptive reordering.

        Args:
            agent_id: If provided, returns data only for this agent

        Returns:
            dict with agent-level and provider-level performance data
        """
        result = {}

        targets = [agent_id] if agent_id else list(self._perf.keys())

        for aid in targets:
            if aid not in self._perf:
                continue

            agent_entry = {}
            for provider, perf in self._perf[aid].items():
                agent_entry[provider] = {
                    "ema_latency_ms": round(perf["ema_latency_ms"], 1),
                    "p50_latency_ms": round(perf["p50_latency_ms"], 1),
                    "samples": perf["samples"],
                    "successes": perf["successes"],
                    "failures": perf["failures"],
                    "success_rate": (
                        round(perf["successes"] / perf["samples"], 3)
                        if perf["samples"] > 0 else 0.0
                    ),
                    "last_updated": perf["last_updated"],
                }

            # Check if all providers in the default chain have enough data
            base_chain = AGENT_PROVIDER_CHAINS.get(aid)
            if base_chain:
                # Use the same threshold as get_optimal_provider_chain
                agent_map = AGENT_MODEL_MAP.get(aid)
                min_s = self.MIN_SAMPLES_FAST if (agent_map and agent_map.latency_sensitive) else self.MIN_SAMPLES
                enough = all(
                    p in self._perf[aid] and self._perf[aid][p]["samples"] >= min_s
                    for p in base_chain
                )
                agent_entry["_adaptive_ready"] = enough

                # Always show what the adapted chain would be (get_optimal_provider_chain handles its own threshold)
                adapted = self.get_optimal_provider_chain(aid, base_chain)
                agent_entry["_adapted_chain"] = adapted
                if adapted != list(base_chain):
                    agent_entry["_chain_adapted"] = True

            result[aid] = agent_entry

        return result

    def reset_adaptive_data(self, agent_id: Optional[str] = None):
        """Reset adaptive routing data for one or all agents."""
        if agent_id:
            self._perf.pop(agent_id, None)
        else:
            self._perf.clear()

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
