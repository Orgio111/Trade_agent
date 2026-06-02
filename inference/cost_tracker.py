"""
Cost Tracker — Monitor token usage and API costs across all providers.

Tracks:
  - Tokens per provider per model
  - Daily and cumulative costs
  - Cost per task type (analysis, reasoning, fast inference)
  - Budget alerts
  - Cost savings from caching
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date


# Budget limits per tier
BUDGET_TIERS = {
    "free": {"daily_usd": 0.0, "monthly_usd": 0.0},
    "low": {"daily_usd": 0.50, "monthly_usd": 15.0},
    "medium": {"daily_usd": 2.0, "monthly_usd": 60.0},
}


@dataclass
class UsageRecord:
    """A single usage event."""
    provider: str
    model: str
    task_type: str
    tokens_prompt: int
    tokens_completion: int
    cost_usd: float
    latency_ms: float
    cached: bool
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    Track token usage and API costs across all providers.

    Provides daily/monthly summaries, budget enforcement, and cost attribution
    by task type, provider, and model.
    """

    def __init__(self, budget_tier: str = "free"):
        self.budget_tier = budget_tier
        self._records: list[UsageRecord] = []
        self._daily_usage: dict[str, dict] = {}
        self._budget = BUDGET_TIERS.get(budget_tier, BUDGET_TIERS["free"])

    def record(
        self,
        provider: str,
        model: str,
        task_type: str,
        tokens_prompt: int,
        tokens_completion: int,
        cost_usd: float,
        latency_ms: float,
        cached: bool = False,
    ):
        """Record a usage event."""
        record = UsageRecord(
            provider=provider,
            model=model,
            task_type=task_type,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            cached=cached,
        )
        self._records.append(record)

        # Update daily stats
        today = date.today().isoformat()
        if today not in self._daily_usage:
            self._daily_usage[today] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "requests": 0,
                "cached_requests": 0,
            }
        self._daily_usage[today]["total_cost"] += cost_usd
        self._daily_usage[today]["total_tokens"] += tokens_prompt + tokens_completion
        self._daily_usage[today]["requests"] += 1
        if cached:
            self._daily_usage[today]["cached_requests"] += 1

    def get_today(self) -> dict:
        """Get today's usage summary."""
        today = date.today().isoformat()
        return self._daily_usage.get(today, {
            "total_cost": 0.0,
            "total_tokens": 0,
            "requests": 0,
            "cached_requests": 0,
        })

    def get_summary(self, days: int = 7) -> dict:
        """Get usage summary for the last N days."""
        cutoff = (date.today().timestamp() - days * 86400)
        recent = [r for r in self._records if r.timestamp > cutoff]

        # Per-provider breakdown
        providers: dict[str, dict] = {}
        for r in recent:
            if r.provider not in providers:
                providers[r.provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "avg_latency_ms": 0.0,
                    "total_latency": 0.0,
                }
            providers[r.provider]["requests"] += 1
            providers[r.provider]["tokens"] += r.tokens_prompt + r.tokens_completion
            providers[r.provider]["cost"] += r.cost_usd
            providers[r.provider]["total_latency"] += r.latency_ms

        # Compute averages
        for p in providers.values():
            p["avg_latency_ms"] = round(p["total_latency"] / p["requests"], 1) if p["requests"] > 0 else 0
            del p["total_latency"]

        # Per-task-type breakdown
        task_types: dict[str, dict] = {}
        for r in recent:
            if r.task_type not in task_types:
                task_types[r.task_type] = {"requests": 0, "cost": 0.0}
            task_types[r.task_type]["requests"] += 1
            task_types[r.task_type]["cost"] += r.cost_usd

        total_cost = sum(p["cost"] for p in providers.values())
        total_tokens = sum(p["tokens"] for p in providers.values())
        cached_count = sum(1 for r in recent if r.cached)
        cache_rate = cached_count / len(recent) if recent else 0.0

        return {
            "period_days": days,
            "total_requests": len(recent),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_request": round(total_cost / len(recent), 6) if recent else 0.0,
            "cache_hit_rate": round(cache_rate, 3),
            "by_provider": providers,
            "by_task_type": task_types,
            "budget_used_pct": round(
                (self.get_today()["total_cost"] / self._budget["daily_usd"]) * 100
                if self._budget["daily_usd"] > 0 else 0.0, 1
            ),
            "budget_tier": self.budget_tier,
        }

    def is_over_budget(self) -> bool:
        """Check if we've exceeded the daily budget."""
        today = self.get_today()
        if self._budget["daily_usd"] <= 0:
            return False
        return today["total_cost"] >= self._budget["daily_usd"]

    def get_expensive_providers(self, top_n: int = 3) -> list[dict]:
        """Get the most expensive providers."""
        summary = self.get_summary()
        providers = summary.get("by_provider", {})
        sorted_providers = sorted(
            providers.items(),
            key=lambda x: x[1]["cost"],
            reverse=True,
        )
        return [
            {"provider": name, "cost_usd": round(data["cost"], 4), "requests": data["requests"]}
            for name, data in sorted_providers[:top_n]
        ]

    def set_budget_tier(self, tier: str):
        """Change the budget tier."""
        if tier in BUDGET_TIERS:
            self.budget_tier = tier
            self._budget = BUDGET_TIERS[tier]
