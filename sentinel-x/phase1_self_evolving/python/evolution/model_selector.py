"""
Autonomous Model Selector — dynamically routes tasks to the best-performing
NIM model based on observed latency and prediction quality.

Uses Upper Confidence Bound (UCB1) bandit algorithm to balance
exploration vs exploitation across available models.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from core.nim_client import nim_json, nim_chat

log = logging.getLogger(__name__)

# Available NIM models ranked by capability (descending)
CANDIDATE_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "nvidia/nemotron-4-340b-instruct",
]


@dataclass
class ModelStats:
    model:        str
    n_calls:      int   = 0
    total_reward: float = 0.0   # accuracy proxy: +1 win, -1 loss, 0 blocked
    total_latency: float = 0.0
    failures:     int   = 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.n_calls, 1)

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency / max(self.n_calls, 1)) * 1000

    def ucb1_score(self, total_calls: int, c: float = 1.41) -> float:
        """UCB1: avg_reward + c * sqrt(ln(N) / n_i)"""
        if self.n_calls == 0:
            return float("inf")  # always try untested models first
        exploration = c * math.sqrt(math.log(total_calls + 1) / self.n_calls)
        # Penalize high latency (normalize: 500ms = -0.2 penalty)
        latency_penalty = min(self.avg_latency_ms / 2500.0, 0.4)
        return self.avg_reward + exploration - latency_penalty


class ModelSelector:
    """
    UCB1 bandit for dynamic NIM model selection.
    Tracks per-model accuracy and latency, automatically routing
    to whichever model has the best risk-adjusted performance.
    """

    def __init__(self, models: list[str] | None = None) -> None:
        self._stats: dict[str, ModelStats] = {
            m: ModelStats(model=m) for m in (models or CANDIDATE_MODELS)
        }
        self._total_calls = 0
        self._lock = asyncio.Lock()

    def select_model(self) -> str:
        """Return the model with the highest UCB1 score."""
        N = self._total_calls
        best = max(
            self._stats.values(),
            key=lambda s: s.ucb1_score(N),
        )
        return best.model

    async def call_with_selection(
        self,
        messages: list[dict],
        task_type: str = "general",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> tuple[dict, str]:
        """
        Select model via UCB1, call NIM, record outcome.
        Returns (parsed_json_result, selected_model_name).
        """
        model = self.select_model()
        t0 = time.monotonic()
        try:
            result = await nim_json(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            )
            latency = time.monotonic() - t0
            async with self._lock:
                s = self._stats[model]
                s.n_calls      += 1
                s.total_latency += latency
                self._total_calls += 1
            log.debug("ModelSelector: %s | latency=%.0fms", model, latency * 1000)
            return result, model
        except Exception as exc:
            latency = time.monotonic() - t0
            async with self._lock:
                s = self._stats[model]
                s.n_calls   += 1
                s.failures  += 1
                s.total_latency += latency
                self._total_calls += 1
            log.warning("Model %s failed: %s — penalizing", model, exc)
            # Fallback to most capable model
            fallback = CANDIDATE_MODELS[0]
            result = await nim_json(messages, model=fallback, temperature=temperature)
            return result, fallback

    def record_trade_outcome(self, model: str, win: bool, pnl_pct: float) -> None:
        """Reward signal: win=+1, loss=-1, scaled by pnl magnitude."""
        if model not in self._stats:
            return
        reward = (1.0 if win else -1.0) * min(abs(pnl_pct) * 10, 1.0)
        self._stats[model].total_reward += reward
        log.debug("Model %s reward: %.3f (win=%s pnl=%.2f%%)", model, reward, win, pnl_pct * 100)

    def get_leaderboard(self) -> list[dict]:
        return sorted(
            [
                {
                    "model":       s.model,
                    "calls":       s.n_calls,
                    "avg_reward":  round(s.avg_reward, 4),
                    "avg_latency_ms": round(s.avg_latency_ms, 1),
                    "ucb1":        round(s.ucb1_score(self._total_calls), 4),
                    "failures":    s.failures,
                }
                for s in self._stats.values()
            ],
            key=lambda x: x["ucb1"],
            reverse=True,
        )
