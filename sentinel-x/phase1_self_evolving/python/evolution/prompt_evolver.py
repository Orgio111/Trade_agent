"""
Autonomous Prompt Evolver — Phase 1 Self-Evolving Core.

Rewrites agent prompts WITHOUT human intervention using a
meta-LLM loop over accumulated failure diagnoses.

Architecture:
  PromptEvolver
    ├── EvolutionHistory  — stores prompt versions + performance
    ├── FailureAggregator — batches diagnoses per agent
    └── MetaLLM rewriter  — NIM generates improved prompt
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from evolution.failure_classifier import FailureDiagnosis, FailureType
from core.nim_client import nim_json

log = logging.getLogger(__name__)

# Minimum failures before triggering evolution
EVOLUTION_THRESHOLD  = 5
# Maximum prompt versions to keep per agent
MAX_PROMPT_VERSIONS  = 10
# Confidence threshold for applying a new prompt (avoid regressing)
MIN_EVOLUTION_CONF   = 0.70


@dataclass
class PromptVersion:
    version:       int
    prompt:        str
    created_at:    datetime
    win_rate:      float = 0.0
    trade_count:   int   = 0
    avg_pnl_pct:   float = 0.0
    active:        bool  = True


@dataclass
class AgentEvolutionState:
    agent_name:     str
    current_prompt: str
    versions:       list[PromptVersion] = field(default_factory=list)
    pending_failures: deque = field(default_factory=lambda: deque(maxlen=50))
    total_evolutions: int   = 0


_META_EVOLVER_SYSTEM = """You are a meta-AI that improves trading agent prompts.
Given an agent's current prompt and a batch of failure diagnoses, rewrite the prompt
to prevent these failures in the future.

Rules:
1. ONLY add specificity — never remove existing valid logic
2. Add concrete failure-prevention examples (few-shot)
3. Add explicit conditions that BLOCK the failure mode
4. Keep the response format requirement identical
5. Return JSON:
{
  "evolved_prompt": "<full new system prompt>",
  "changes_summary": "<3 bullet points of what changed>",
  "confidence": <0-1 that this evolution will improve performance>,
  "test_condition": "<specific condition to verify this helped>"
}
Be surgical. Prompt bloat kills reasoning quality."""


class PromptEvolver:
    """
    Autonomously evolves agent prompts based on accumulated failure diagnoses.
    Implements a version control system with rollback capability.
    """

    def __init__(self, initial_prompts: dict[str, str]) -> None:
        self._states: dict[str, AgentEvolutionState] = {
            name: AgentEvolutionState(
                agent_name=name,
                current_prompt=prompt,
                versions=[PromptVersion(
                    version=0,
                    prompt=prompt,
                    created_at=datetime.utcnow(),
                    active=True,
                )],
            )
            for name, prompt in initial_prompts.items()
        }
        self._evolution_lock = asyncio.Lock()

    def record_failure(self, diagnosis: FailureDiagnosis) -> None:
        """Queue a failure diagnosis for the affected agents."""
        for agent in diagnosis.affected_agents:
            if agent in self._states:
                self._states[agent].pending_failures.append(diagnosis)
                log.debug("Failure queued for %s: %s (severity=%.2f)",
                          agent, diagnosis.failure_type.value, diagnosis.severity)

    def record_outcome(self, agent: str, win: bool, pnl_pct: float) -> None:
        """Track trade outcomes for the current prompt version."""
        if agent not in self._states:
            return
        state = self._states[agent]
        if state.versions:
            v = state.versions[-1]
            n = v.trade_count
            v.win_rate   = (v.win_rate * n + (1.0 if win else 0.0)) / (n + 1)
            v.avg_pnl_pct = (v.avg_pnl_pct * n + pnl_pct) / (n + 1)
            v.trade_count += 1

    async def maybe_evolve(self, agent: str) -> PromptVersion | None:
        """
        Check if the agent has enough failures to trigger evolution.
        Returns the new PromptVersion if evolved, None otherwise.
        """
        if agent not in self._states:
            return None

        state = self._states[agent]
        failures = list(state.pending_failures)

        if len(failures) < EVOLUTION_THRESHOLD:
            return None

        # Filter for high-severity, high-confidence diagnoses
        actionable = [f for f in failures if f.confidence > 0.6 and f.severity > 0.4]
        if len(actionable) < 3:
            return None

        async with self._evolution_lock:
            return await self._evolve_prompt(agent, actionable)

    async def _evolve_prompt(
        self,
        agent: str,
        failures: list[FailureDiagnosis],
    ) -> PromptVersion | None:
        state = self._states[agent]

        # Aggregate failure patterns
        type_counts: dict[str, int] = defaultdict(int)
        for f in failures:
            type_counts[f.failure_type.value] += 1

        top_failure = max(type_counts, key=type_counts.get)  # type: ignore[arg-type]
        failure_summary = "\n".join(
            f"  [{f.failure_type.value}] (severity={f.severity:.2f}): {f.root_cause}\n"
            f"    Recommended: {f.recommended_action}"
            for f in failures[:5]  # send top 5 to NIM
        )

        context = (
            f"Agent: {agent}\n"
            f"Evolution #: {state.total_evolutions + 1}\n"
            f"Current win rate: {state.versions[-1].win_rate:.1%}\n"
            f"Dominant failure: {top_failure} ({type_counts[top_failure]} occurrences)\n\n"
            f"Current prompt:\n{state.current_prompt}\n\n"
            f"Failure diagnoses:\n{failure_summary}"
        )

        result = await nim_json(
            [
                {"role": "system", "content": _META_EVOLVER_SYSTEM},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        evolution_confidence = float(result.get("confidence", 0.5))
        if evolution_confidence < MIN_EVOLUTION_CONF:
            log.warning(
                "Evolution confidence %.2f < %.2f threshold — not applying for %s",
                evolution_confidence, MIN_EVOLUTION_CONF, agent,
            )
            return None

        new_prompt = result.get("evolved_prompt", "")
        if not new_prompt or len(new_prompt) < 100:
            log.error("Evolver returned empty/short prompt for %s — skipping", agent)
            return None

        new_version = PromptVersion(
            version=state.total_evolutions + 1,
            prompt=new_prompt,
            created_at=datetime.utcnow(),
        )
        state.versions.append(new_version)
        state.current_prompt = new_prompt
        state.total_evolutions += 1
        state.pending_failures.clear()

        # Prune old versions
        if len(state.versions) > MAX_PROMPT_VERSIONS:
            state.versions = state.versions[-MAX_PROMPT_VERSIONS:]

        log.info(
            "PROMPT EVOLVED: %s v%d | confidence=%.2f | change: %s",
            agent, new_version.version, evolution_confidence,
            result.get("changes_summary", "")[:80],
        )
        return new_version

    def rollback(self, agent: str, to_version: int) -> bool:
        """Roll back to a previous prompt version."""
        if agent not in self._states:
            return False
        state = self._states[agent]
        for v in state.versions:
            if v.version == to_version:
                state.current_prompt = v.prompt
                log.warning("Rolled back %s to v%d", agent, to_version)
                return True
        return False

    def get_prompt(self, agent: str) -> str:
        return self._states[agent].current_prompt if agent in self._states else ""

    def get_evolution_report(self) -> dict:
        return {
            name: {
                "total_evolutions": s.total_evolutions,
                "current_version":  s.versions[-1].version if s.versions else 0,
                "win_rate":         s.versions[-1].win_rate if s.versions else 0,
                "pending_failures": len(s.pending_failures),
            }
            for name, s in self._states.items()
        }
