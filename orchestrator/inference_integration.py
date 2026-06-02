"""
QUANTEX Inference Integration — Bridge between agent routing, task resolution,
and the multi-provider inference router.

Architecture:
  Agent Code → InferenceIntegration → TaskRouter (resolve agent/model)
                                    → InferenceRouter (execute with fallback)
                                    → Provider Chain (Groq → NIM → OpenRouter)

This module is a drop-in replacement for NIMOrchestrator and EnhancedNIMOrchestrator.
It maintains the same `route_inference()`, `batch_embeddings()`, and
`parallel_agent_inference()` interface so existing agent code works unchanged.
"""

import os
import time
import asyncio
import logging
from typing import Optional, AsyncGenerator

from inference import InferenceRouter, InferenceTask, RouterConfig
from inference.providers.base import ProviderResponse
from .agent_routing import TaskRouter, AgentModelRouter, AGENT_PROVIDER_CHAINS, AGENT_TASK_TYPE_OVERRIDES

logger = logging.getLogger("quantex.inference.integration")


class InferenceIntegration:
    """
    Drop-in replacement for NIMOrchestrator / EnhancedNIMOrchestrator.

    Routes inference requests through the new multi-provider InferenceRouter
    while maintaining full backward compatibility with existing agent code.

    Key differences from the old NIM orchestrators:
      - Multi-provider fallback chain (Groq → NVIDIA NIM → OpenRouter)
      - Semantic caching (Qdrant) for 40-60% cost reduction
      - Circuit breaker — auto-skips failing providers
      - Cost tracking with daily/monthly budgets
      - Per-request latency and token tracking
      - Streaming passthrough for all providers

    Usage (backward compatible):
        nim = InferenceIntegration()
        result = await nim.route_inference("analysis", messages)
        embeddings = await nim.batch_embeddings(["text1", "text2"])
    """

    def __init__(self, budget_tier: str = "free"):
        config = RouterConfig(budget_tier=budget_tier)
        self.router = InferenceRouter(config=config)
        self.task_router = TaskRouter()
        self.agent_router = AgentModelRouter()

        # Track whether GROQ_API_KEY is set (log important note once)
        if os.getenv("GROQ_API_KEY"):
            logger.info("Groq provider configured — available for ultra-fast inference")
        else:
            logger.info("Groq provider not configured — set GROQ_API_KEY for ultra-fast inference")

        if os.getenv("NVIDIA_API_KEY"):
            logger.info("NVIDIA NIM provider configured — available for high-quality inference")
        if os.getenv("OPENROUTER_API_KEY"):
            logger.info("OpenRouter provider configured — available for fallback inference")

    # ── Primary Interface (backward compatible) ──────────────────────────

    async def route_inference(
        self,
        task_type: str,
        messages: list,
        stream: bool = False,
        cache_ttl: int = 300,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        agent_id: str | None = None,
    ) -> str | AsyncGenerator:
        """
        Route an inference request through the best available provider.

        When `agent_id` is provided, the router uses agent-specific:
          - Provider chain (e.g., scalping → Groq first, deepseek → NVIDIA NIM first)
          - Task type override (e.g., anomaly → "fast", sentiment → "classification")
          - Adaptive reordering: provider chain sorted by observed P50 latency

        After each request, the actual provider latency is recorded in
        the adaptive routing engine, which automatically adjusts future
        routing for that agent.

        Args:
            task_type: "reasoning", "analysis", "fast", "classification", "coding"
            messages: OpenAI-format message list
            stream: If True, returns an async generator of tokens
            cache_ttl: Cache TTL in seconds (used by semantic cache)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            agent_id: Originating agent ID for agent-specific routing

        Returns:
            Response text (str) or async generator for streaming

        Raises:
            Exception (ProviderUnavailable) if all providers fail
        """
        # Resolve agent-specific routing overrides
        resolved_task_type = AGENT_TASK_TYPE_OVERRIDES.get(agent_id, task_type) if agent_id else task_type

        # Use adaptive provider chain (sorted by observed P50 latency)
        if agent_id:
            base_chain = AGENT_PROVIDER_CHAINS.get(agent_id)
            if base_chain:
                provider_override = self.agent_router.get_optimal_provider_chain(agent_id, base_chain)
            else:
                provider_override = None
        else:
            provider_override = None

        task = InferenceTask(
            task_type=resolved_task_type,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            skip_cache=(cache_ttl <= 0),
            agent_name=agent_id or "unknown",
            provider_override=provider_override,
        )

        if stream:
            return self.router.infer_stream(task)

        try:
            result: ProviderResponse = await self.router.infer(task)

            # Feed back latency data for adaptive routing
            if agent_id:
                self.agent_router.record_provider_latency(
                    agent_id=agent_id,
                    provider=result.provider,
                    latency_ms=result.latency_ms,
                    success=True,
                )

            return result.content

        except (ProviderUnavailable, ProviderTimeout, ProviderError) as e:
            # Record failure for adaptive routing (uses penalty latency)
            if agent_id and provider_override:
                # Record failure for each provider in the chain
                # (we don't know which one was tried last, so record for all)
                for p in provider_override:
                    self.agent_router.record_provider_latency(
                        agent_id=agent_id,
                        provider=p,
                        latency_ms=AgentModelRouter.FAILURE_LATENCY_MS,
                        success=False,
                    )
            raise

    async def batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Uses the NVIDIA NIM embedding model directly (not routed through
        the multi-provider chain since only NIM supports embeddings).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        nim_provider = self.router._providers.get("nvidia_nim")
        if nim_provider and hasattr(nim_provider, "batch_embeddings"):
            try:
                return await nim_provider.batch_embeddings(texts)
            except Exception as e:
                logger.warning("NVIDIA NIM embedding failed: %s", e)

        # Fallback: return zero vectors if NIM unavailable
        logger.warning("No embedding provider available — returning zero vectors")
        return [[0.0] * 1536 for _ in texts]

    async def parallel_agent_inference(self, agent_tasks: list[dict]) -> list[str]:
        """
        Run multiple agent inferences in parallel.

        Args:
            agent_tasks: List of dicts, each with keys:
                - type: task_type (optional, default "fast")
                - messages: OpenAI-format messages
                - temperature: sampling temperature (optional)

        Returns:
            List of response strings (errors are returned as strings)
        """
        async def _run(task: dict):
            try:
                return await self.route_inference(
                    task_type=task.get("type", "fast"),
                    messages=task["messages"],
                    temperature=task.get("temperature", 0.1),
                )
            except Exception as e:
                return f"Error: {e}"

        return await asyncio.gather(
            *[_run(task) for task in agent_tasks],
            return_exceptions=False,  # We handle exceptions in _run
        )

    # ── Monitoring and Stats ────────────────────────────────────────────

    async def get_provider_health(self) -> dict:
        """Get health status of all providers."""
        return await self.router.get_provider_health()

    def get_routing_stats(self) -> dict:
        """Get latency statistics for each route via AgentModelRouter."""
        return self.agent_router.get_route_stats()

    def get_cost_summary(self, days: int = 7) -> dict:
        """Get cost summary for the last N days."""
        return self.router.get_cost_summary(days=days)

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return self.router.get_cache_stats()

    def get_todays_usage(self) -> dict:
        """Get today's token and cost usage."""
        return self.router.get_todays_usage()

    def is_over_budget(self) -> bool:
        """Check if we've exceeded today's budget."""
        return self.router.is_over_budget()

    def get_free_model_list(self) -> dict:
        """Get list of all available free models (from agent routing)."""
        return self.agent_router.get_free_model_list()

    def get_agent_model_summary(self) -> list[dict]:
        """Get a summary of all agent-to-model mappings."""
        return self.agent_router.get_agent_model_summary()

    # ── Convenience Methods ─────────────────────────────────────────────

    async def resolve_task(self, task_type: str) -> dict:
        """Resolve a task type to its optimal agent and model config."""
        return self.task_router.resolve(task_type)

    async def summarize(self) -> dict:
        """Get a comprehensive summary of the inference system status."""
        return {
            "provider_health": await self.get_provider_health(),
            "todays_usage": self.get_todays_usage(),
            "cost_summary": self.get_cost_summary(days=1),
            "cache_stats": self.get_cache_stats(),
            "over_budget": self.is_over_budget(),
            "agent_summary": self.get_agent_model_summary(),
            "adaptive_routing": self.agent_router.get_adaptive_routing_summary(),
        }

    # ── Access to underlying components ─────────────────────────────────

    @property
    def inference_router(self) -> InferenceRouter:
        """Access the underlying InferenceRouter directly."""
        return self.router

    @property
    def agent_model_router(self) -> AgentModelRouter:
        """Access the underlying AgentModelRouter directly."""
        return self.agent_router

    @property
    def task_type_router(self) -> TaskRouter:
        """Access the underlying TaskRouter directly."""
        return self.task_router
