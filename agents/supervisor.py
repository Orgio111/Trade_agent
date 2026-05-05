"""
Supervisor (Portfolio Manager): LangGraph-based stateful orchestration.
Implements the Supervisor Pattern — coordinates all sub-agents,
synthesizes debate logs + risk reports, issues final BUY/SELL/HOLD orders.
"""
from __future__ import annotations

import asyncio
import logging
import operator
import time
from datetime import datetime
from typing import Annotated, Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph  # type: ignore[import]

from agents.council import CouncilAgent
from agents.execution import ExecutionAgent
from agents.fundamental import FundamentalAgent
from agents.memory_agent import MemoryAgent
from agents.risk_engine import RiskEngine
from agents.sentiment import SentimentAgent
from agents.technical import TechnicalAgent
from core.config import get_settings
from core.kill_switch import KillSwitch
from core.models import (
    CouncilDecision,
    Order,
    PortfolioState,
    RiskReport,
    Side,
    TechnicalSignal,
)
from core.nim_client import nim_chat
from core.observability import (
    AGENT_LATENCY,
    PNL_GAUGE,
    DRAWDOWN_GAUGE,
)

logger = logging.getLogger(__name__)


# ── LangGraph state schema ─────────────────────────────────────────────────────
class TradingState(TypedDict):
    symbol: str
    closes: list[float]
    highs: list[float]
    lows: list[float]
    volumes: list[float]
    current_price: float
    portfolio: PortfolioState
    technical_signal: TechnicalSignal | None
    council_decision: CouncilDecision | None
    risk_report: RiskReport | None
    order: Order | None
    logs: Annotated[list[str], operator.add]
    abort: bool


# ── Node functions ─────────────────────────────────────────────────────────────
async def node_technical(state: TradingState) -> dict:
    agent = TechnicalAgent()
    sig = await agent.analyze(
        state["symbol"],
        np.array(state["closes"]),
        np.array(state["highs"]),
        np.array(state["lows"]),
        np.array(state["volumes"]),
    )
    return {"technical_signal": sig, "logs": [f"Technical: {sig.trend.value} conf={sig.confidence:.2f}"]}


async def node_fundamental(state: TradingState) -> dict:
    agent = FundamentalAgent()
    sig = await agent.analyze(state["symbol"])
    return {"logs": [f"Fundamental: {sig.trend.value} conf={sig.confidence:.2f}"]}


async def node_sentiment(state: TradingState) -> dict:
    agent = SentimentAgent()
    sig = await agent.analyze(state["symbol"])
    return {"logs": [f"Sentiment: {sig.trend.value} conf={sig.confidence:.2f}"]}


async def node_council(state: TradingState) -> dict:
    agent = CouncilAgent()
    decision = await agent.deliberate(
        state["symbol"],
        technical=state.get("technical_signal"),
    )
    return {
        "council_decision": decision,
        "logs": [f"Council: {decision.final_side.value} consensus={decision.consensus_score:.2f}"],
    }


async def node_risk(state: TradingState) -> dict:
    cfg = get_settings()
    decision = state["council_decision"]
    if decision is None:
        return {"abort": True, "logs": ["Risk: no council decision — aborting"]}

    closes = np.array(state["closes"])
    returns = np.diff(np.log(closes + 1e-10))
    portfolio = state["portfolio"]
    win_rate = 0.55
    avg_win = 0.015
    avg_loss = 0.010

    engine = RiskEngine()
    report = await engine.evaluate(
        decision,
        returns,
        state["current_price"],
        portfolio.equity,
        win_rate,
        avg_win,
        avg_loss,
    )
    abort = not report.approved
    msg = f"Risk: {'APPROVED' if report.approved else 'REJECTED — ' + (report.rejection_reason or '')}"
    return {"risk_report": report, "abort": abort, "logs": [msg]}


async def node_supervisor_decision(state: TradingState) -> dict:
    cfg = get_settings()
    decision = state["council_decision"]
    risk = state["risk_report"]
    if not decision or not risk or state.get("abort"):
        return {"abort": True, "logs": ["Supervisor: no valid signal — HOLD"]}

    # Final NIM-powered rationale synthesis
    debate_summary = "\n".join(
        f"  [{a.agent_name}] {a.position.value}: {a.argument}"
        for a in decision.debate_log
    )
    prompt = (
        f"Symbol: {decision.symbol}\n"
        f"Council final: {decision.final_side.value} (consensus={decision.consensus_score:.2f})\n"
        f"Debate:\n{debate_summary}\n"
        f"Risk: VaR99={risk.var_99:.2%}, size_usd={risk.position_size_usd:.0f}, "
        f"stop={risk.stop_loss_price:.4f}, TP={risk.take_profit_price:.4f}\n"
        "In 2 sentences, explain the final trading rationale."
    )
    rationale = await nim_chat(
        [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=200
    )
    decision.rationale = rationale
    return {"council_decision": decision, "logs": [f"Supervisor rationale: {rationale[:80]}..."]}


async def node_execute(state: TradingState) -> dict:
    if state.get("abort"):
        return {"logs": ["Execution: skipped (abort=True)"]}
    decision = state["council_decision"]
    risk = state["risk_report"]
    if not decision or not risk:
        return {"logs": ["Execution: missing decision or risk"]}

    agent = ExecutionAgent()
    order = await agent.execute(decision, risk, np.array(state["closes"]))
    return {"order": order, "logs": [f"Execution: {order.status.value} @ {order.avg_fill_price:.4f}"]}


def should_abort(state: TradingState) -> str:
    return "abort" if state.get("abort") else "continue"


# ── Graph assembly ─────────────────────────────────────────────────────────────
def build_trading_graph() -> Any:
    g = StateGraph(TradingState)

    g.add_node("technical", node_technical)
    g.add_node("fundamental", node_fundamental)
    g.add_node("sentiment", node_sentiment)
    g.add_node("council", node_council)
    g.add_node("risk", node_risk)
    g.add_node("supervisor", node_supervisor_decision)
    g.add_node("execute", node_execute)

    g.set_entry_point("technical")
    # Fan-out: technical → fundamental & sentiment in parallel
    g.add_edge("technical", "fundamental")
    g.add_edge("technical", "sentiment")
    # Fan-in: both feed council
    g.add_edge("fundamental", "council")
    g.add_edge("sentiment", "council")

    g.add_edge("council", "risk")
    g.add_conditional_edges(
        "risk",
        should_abort,
        {"abort": END, "continue": "supervisor"},
    )
    g.add_conditional_edges(
        "supervisor",
        should_abort,
        {"abort": END, "continue": "execute"},
    )
    g.add_edge("execute", END)

    return g.compile()


# ── Supervisor orchestrator class ──────────────────────────────────────────────
class PortfolioSupervisor:
    """Top-level orchestrator. Feeds market data into the LangGraph pipeline."""

    def __init__(self, initial_equity: float) -> None:
        cfg = get_settings()
        self._graph = build_trading_graph()
        self._kill_switch = KillSwitch(initial_equity)
        self._portfolio = PortfolioState(
            equity=initial_equity,
            cash=initial_equity,
            peak_equity=initial_equity,
        )
        self._memory = MemoryAgent()

    async def start(self) -> None:
        await self._memory.connect()
        logger.info("PortfolioSupervisor started — equity=%.2f", self._portfolio.equity)

    async def run_cycle(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> Order | None:
        if self._kill_switch.is_active:
            logger.warning("Kill switch active — skipping cycle for %s", symbol)
            return None

        state: TradingState = {
            "symbol": symbol,
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "volumes": volumes,
            "current_price": closes[-1] if closes else 0.0,
            "portfolio": self._portfolio,
            "technical_signal": None,
            "council_decision": None,
            "risk_report": None,
            "order": None,
            "logs": [],
            "abort": False,
        }

        with AGENT_LATENCY.labels(agent="supervisor").time():
            result = await self._graph.ainvoke(state)

        for log_line in result.get("logs", []):
            logger.info("[%s] %s", symbol, log_line)

        order = result.get("order")
        if order and result.get("council_decision") and result.get("risk_report"):
            await self._memory.record_open(
                order,
                result["council_decision"].model_dump(mode="json"),
                result["risk_report"].model_dump(mode="json"),
            )

        # Update portfolio state (simplified — real impl tracks open positions)
        self._portfolio.timestamp = datetime.utcnow()
        PNL_GAUGE.labels(symbol=symbol).set(self._portfolio.daily_pnl)
        DRAWDOWN_GAUGE.set(self._portfolio.current_drawdown_pct)

        triggered = self._kill_switch.update(self._portfolio)
        if triggered:
            self._portfolio.kill_switch_active = True
            await self._kill_switch.broadcast_halt()

        return order
