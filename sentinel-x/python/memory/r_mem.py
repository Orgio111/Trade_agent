"""
Reasoning-Driven Hierarchical Memory (R-Mem).
The Supervisor Agent consults R-Mem to resolve conflicts between Bull/Bear agents
by retrieving the most relevant historical reasoning chains from FAISS.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from core.nim_client import nim_chat, nim_embed, nim_json
from memory.vector_store import FAISSStrategyMemory, StrategyMemory

log = logging.getLogger(__name__)

_RMEM_QUERY_SYSTEM = """You are a trading historian with access to past trade reasoning chains.
Given a current market context and similar historical setups, extract the key lessons.
Focus on: which signals were predictive, what caused failures, and how similar setups resolved.
Be concise — max 3 bullet points."""

_PROMPT_AMENDMENT_SYSTEM = """You are a meta-learning agent. Based on the failure analysis of a trade,
you must improve the few-shot examples in the analyst prompt.
Return JSON:
{
  "agent": "<technical|fundamental|sentiment|quant>",
  "failure_type": "shallow_reasoning" | "model_drift" | "regime_change" | "execution_error",
  "new_few_shot_example": "<formatted example that would have caught this failure>",
  "prompt_amendment": "<specific instruction to add to the agent's system prompt>"
}"""


class RMem:
    """
    Reasoning-Driven Hierarchical Memory — bridges FAISS retrieval and NIM reasoning.
    Provides context to the Supervisor and amends agent prompts post-trade.
    """

    def __init__(self, store: FAISSStrategyMemory) -> None:
        self._store = store
        self._prompt_amendments: dict[str, list[str]] = {
            "technical": [],
            "fundamental": [],
            "sentiment": [],
            "quant": [],
        }

    async def retrieve_context(
        self,
        reasoning_chain: str,
        symbol: str,
        k: int = 5,
    ) -> str:
        """
        Embed the current reasoning chain, retrieve k similar past setups,
        and synthesize a contextual summary via NIM.
        """
        embeddings = await nim_embed([reasoning_chain])
        query_vec = embeddings[0]
        import numpy as np
        similar = self._store.search(np.array(query_vec), k=k)

        if not similar:
            return "No similar historical setups found in strategy memory."

        history_txt = "\n\n".join(
            f"[{m.timestamp.date()} | {m.symbol} | {m.outcome} | PnL={m.pnl_pct:.1%}]\n"
            f"Reasoning: {m.reasoning_chain[:300]}..."
            for m in similar
        )

        summary = await nim_chat(
            [
                {"role": "system", "content": _RMEM_QUERY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Current setup ({symbol}):\n{reasoning_chain[:400]}\n\n"
                        f"Similar historical setups:\n{history_txt}\n\n"
                        "Summarize the key lessons in 3 bullet points."
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=256,
        )
        return summary

    async def record_outcome(
        self,
        session_id: str,
        symbol: str,
        reasoning_chain: str,
        outcome: str,
        pnl_pct: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed and store the trade outcome in FAISS."""
        embeddings = await nim_embed([reasoning_chain])
        import numpy as np
        vec = np.array(embeddings[0])

        memory = StrategyMemory(
            session_id=session_id,
            symbol=symbol,
            timestamp=datetime.utcnow(),
            reasoning_chain=reasoning_chain,
            outcome=outcome,
            pnl_pct=pnl_pct,
            embedding=vec,
            metadata=metadata or {},
        )
        self._store.add(memory)

    async def perform_post_mortem(
        self,
        session_id: str,
        symbol: str,
        reasoning_chain: str,
        outcome: str,
        pnl_pct: float,
    ) -> dict[str, str]:
        """
        After a LOSS, analyze the reasoning chain to identify failure type
        and amend agent prompts (Recursive Prompt Engineering).
        """
        if outcome == "WIN":
            log.info("Post-mortem skipped for winning trade %s", session_id[:8])
            return {}

        result = await nim_json(
            [
                {"role": "system", "content": _PROMPT_AMENDMENT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Symbol: {symbol} | PnL: {pnl_pct:.2%} | Outcome: {outcome}\n"
                        f"Full reasoning chain:\n{reasoning_chain}\n\n"
                        "Analyze the failure and provide the prompt amendment."
                    ),
                },
            ],
            temperature=0.1,
        )

        agent = result.get("agent", "technical")
        amendment = result.get("prompt_amendment", "")
        failure_type = result.get("failure_type", "unknown")
        new_example  = result.get("new_few_shot_example", "")

        if amendment and agent in self._prompt_amendments:
            self._prompt_amendments[agent].append(amendment)
            log.warning(
                "Prompt amended for '%s' agent — failure: %s (session=%s)",
                agent, failure_type, session_id[:8],
            )

        return {
            "agent": agent,
            "failure_type": failure_type,
            "amendment": amendment,
            "new_example": new_example,
        }

    def get_amendments(self, agent: str) -> list[str]:
        return self._prompt_amendments.get(agent, [])
