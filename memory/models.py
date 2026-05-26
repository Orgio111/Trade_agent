"""Data models for the TurboVec semantic memory system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core.models import Side


class MemoryType(str, Enum):
    """Category of memory entry — used for filtering during retrieval."""
    MARKET_STATE = "market_state"
    EXECUTED_TRADE = "executed_trade"
    PROFITABLE_SETUP = "profitable_setup"
    FAILED_SETUP = "failed_setup"
    RISK_EVENT = "risk_event"
    AGENT_DECISION = "agent_decision"
    NEWS_EVENT = "news_event"
    MACRO_EVENT = "macro_event"


class RetrievalPurpose(str, Enum):
    """Why we are retrieving — influences ranking weights."""
    TRADE_VALIDATION = "trade_validation"      # can we find similar profitable setups?
    RISK_ASSESSMENT = "risk_assessment"        # what went wrong in similar conditions?
    SIGNAL_ENHANCEMENT = "signal_enhancement"  # boost confidence when pattern matches
    GENERAL_CONTEXT = "general_context"        # broad historical awareness


@dataclass
class MemoryEntry:
    """
    A single memory entry stored in the TurboVec engine.

    The ``vector`` is a compressed embedding of the market state / event.
    ``metadata`` holds structured details (prices, signals, outcome, etc.).
    """
    memory_id: str                           # UUID
    memory_type: MemoryType
    symbol: str
    timestamp: datetime
    vector: list[float] | None = None        # embedding; None before indexing
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional outcome fields (filled post-trade)
    side: Side | None = None
    pnl_pct: float | None = None
    win: bool | None = None
    consensus_score: float | None = None
    confidence_score: float | None = None


@dataclass
class SearchResult:
    """A single result from a TurboVec similarity search."""
    memory: MemoryEntry
    similarity: float                        # cosine similarity [0, 1]
    rank: int


@dataclass
class RetrievalContext:
    """
    Structured context injected into agent prompts.

    Carries both the raw search results and pre-formatted text
    that can be directly appended to a system prompt.
    """
    purpose: RetrievalPurpose
    symbol: str
    query_vector: list[float] | None = None
    results: list[SearchResult] = field(default_factory=list)
    n_similar_profitable: int = 0            # count of positive-outcome neighbors
    n_similar_failed: int = 0                # count of negative-outcome neighbors
    avg_win_pnl: float | None = None         # avg PnL % of profitable neighbors
    avg_loss_pnl: float | None = None
    memory_age_hours: float | None = None    # how far back the memories span

    def format_for_prompt(self, max_results: int = 5) -> str:
        """Format as a text block that can be injected into any agent prompt."""
        if not self.results:
            return ""

        lines = [
            "━━━ Historical Memory Context (TurboVec) ━━━",
            f"Purpose: {self.purpose.value}",
            f"Matched {len(self.results)} similar historical states:",
        ]

        for r in self.results[:max_results]:
            mem = r.memory
            ts = mem.timestamp.strftime("%Y-%m-%d %H:%M")
            outcome = ""
            if mem.win is not None:
                outcome = f" | {'✅ WIN' if mem.win else '❌ LOSS'} ({mem.pnl_pct:+.2%})"
            elif mem.memory_type == MemoryType.RISK_EVENT:
                outcome = " | ⚠️ RISK EVENT"
            lines.append(
                f"  [{r.rank}] {ts} {mem.symbol} "
                f"sim={r.similarity:.2f}{outcome}"
            )

        if self.n_similar_profitable > 0 or self.n_similar_failed > 0:
            lines.append(
                f"\n  → {self.n_similar_profitable} profitable / {self.n_similar_failed} failed "
                f"(avg win: {self.avg_win_pnl:+.2%}, avg loss: {self.avg_loss_pnl:+.2%})"
            )

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return "\n".join(lines)
