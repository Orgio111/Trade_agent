"""
Supervisor Agent — LangGraph StateGraph with R-Mem integration.
Orchestrates the full 5-agent council debate, applies STL Protocol,
and issues final trade decisions routed to the Go Gateway.
"""
from __future__ import annotations

import asyncio
import json
import logging
import operator
import uuid
from datetime import datetime
from typing import Annotated, Any, TypedDict

import httpx
import numpy as np
from langgraph.graph import END, StateGraph  # type: ignore[import]

from agents.quant import QuantAgent
from council.stl_protocol import STLResult, run_stl_protocol
from core.nim_client import nim_chat, nim_json
from core.scheduler import TaskType, route
from memory.r_mem import RMem
from memory.vector_store import FAISSStrategyMemory

log = logging.getLogger(__name__)

# ── Bull / Bear prompts (refined via R-Mem amendments) ───────────────────────
_BULL_SYSTEM = """You are the BULL analyst. Build the strongest BUY case from the data.
Back every claim with a specific data point. Output JSON:
{
  "position": "BUY",
  "argument": "<evidence-based 2-3 sentences>",
  "quantitative_score": <0-1 conviction>,
  "supporting_factors": ["<factor with data>", ...],
  "risk_factors": ["<bear risk>", ...]
}"""

_BEAR_SYSTEM = """You are the BEAR analyst. Build the strongest SELL case from the data.
Back every claim with a specific data point. Output JSON:
{
  "position": "SELL",
  "argument": "<evidence-based 2-3 sentences>",
  "quantitative_score": <0-1 conviction>,
  "supporting_factors": ["<factor with data>", ...],
  "risk_factors": ["<bull risk>", ...]
}"""

_SUPERVISOR_RATIONALE_SYSTEM = """You are the Portfolio Manager at a $100M+ hedge fund.
You have the full debate log, STL scores, risk report, and R-Mem context.
Produce a 2-3 sentence natural-language trade rationale that:
1. States the directional thesis
2. Quantifies the key risk metric (VaR99 or ATR stop)
3. Names the single most compelling signal and the biggest risk
Be concise, institutional-grade. No filler."""


# ── LangGraph state schema ─────────────────────────────────────────────────────
class SentinelState(TypedDict):
    symbol:         str
    session_id:     str
    closes:         list[float]
    highs:          list[float]
    lows:           list[float]
    volumes:        list[float]
    current_price:  float

    # Agent outputs
    bull_arg:       dict | None
    bear_arg:       dict | None
    fundamental:    dict | None
    sentiment_sig:  dict | None
    quant_analysis: dict | None

    # Council outputs
    r_mem_context:  str
    stl:            STLResult | None
    reasoning_chain: str
    rationale:      str

    # Decision
    blocked:        bool
    abort:          bool
    logs:           Annotated[list[str], operator.add]


# ── Graph node implementations ─────────────────────────────────────────────────
def _market_context(state: SentinelState) -> str:
    closes = np.array(state["closes"])
    returns = np.diff(np.log(closes + 1e-10))
    rv = float(returns[-20:].std() * np.sqrt(252)) if len(returns) >= 20 else 0.0
    return (
        f"Symbol: {state['symbol']} | Price: {state['current_price']:.4f} | "
        f"RV(20d): {rv:.1%} | Bars: {len(closes)}"
    )


async def node_r_mem(state: SentinelState, r_mem: RMem) -> dict:
    """Retrieve similar historical setups from FAISS before debate starts."""
    context_query = f"{state['symbol']} price={state['current_price']:.2f}"
    context = await r_mem.retrieve_context(context_query, state["symbol"])
    return {"r_mem_context": context, "logs": [f"R-Mem: retrieved context ({len(context)} chars)"]}


async def node_bull(state: SentinelState, r_mem: RMem) -> dict:
    ctx = _market_context(state)
    amendments = r_mem.get_amendments("bull")
    extra = ("\nLearned refinements:\n" + "\n".join(f"- {a}" for a in amendments[-3:])) if amendments else ""
    result = await route(
        TaskType.LLM,
        gpu_fn=lambda: nim_json(
            [{"role": "system", "content": _BULL_SYSTEM + extra},
             {"role": "user", "content": ctx}],
            temperature=0.4,
        ),
        cpu_fn=lambda: asyncio.coroutine(lambda: {
            "position": "BUY", "argument": "CPU fallback bull argument",
            "quantitative_score": 0.5, "supporting_factors": [], "risk_factors": []
        })(),
    )
    return {"bull_arg": result, "logs": [f"Bull: score={result.get('quantitative_score', 0):.2f}"]}


async def node_bear(state: SentinelState, r_mem: RMem) -> dict:
    ctx = _market_context(state)
    result = await route(
        TaskType.LLM,
        gpu_fn=lambda: nim_json(
            [{"role": "system", "content": _BEAR_SYSTEM},
             {"role": "user", "content": ctx}],
            temperature=0.4,
        ),
        cpu_fn=lambda: asyncio.coroutine(lambda: {
            "position": "SELL", "argument": "CPU fallback bear argument",
            "quantitative_score": 0.5, "supporting_factors": [], "risk_factors": []
        })(),
    )
    return {"bear_arg": result, "logs": [f"Bear: score={result.get('quantitative_score', 0):.2f}"]}


async def node_quant(state: SentinelState) -> dict:
    agent = QuantAgent()
    result = await agent.analyze(
        state["symbol"],
        np.array(state["closes"]),
        np.array(state["highs"]),
        np.array(state["lows"]),
    )
    return {
        "quant_analysis": result,
        "logs": [f"Quant: regime={result.get('regime')} hurst={result.get('hurst', 0):.3f}"],
    }


async def node_stl(state: SentinelState) -> dict:
    """Run the full 5-agent STL Protocol calibration."""
    stl = await run_stl_protocol(
        symbol=state["symbol"],
        bull_arg=state.get("bull_arg") or {},
        bear_arg=state.get("bear_arg") or {},
        fundamental=state.get("fundamental") or {},
        sentiment=state.get("sentiment_sig") or {},
        quant=state.get("quant_analysis") or {},
        r_mem_context=state.get("r_mem_context", ""),
    )

    reasoning_chain = json.dumps({
        "session_id": state["session_id"],
        "symbol":     state["symbol"],
        "bull_arg":   state.get("bull_arg"),
        "bear_arg":   state.get("bear_arg"),
        "quant":      state.get("quant_analysis"),
        "stl":        {
            "bull_score": stl.bull_score,
            "bear_score": stl.bear_score,
            "confidence_pct": stl.confidence_pct,
            "final_side": stl.final_side,
            "contradictions": stl.contradictions,
        },
    }, default=str)

    return {
        "stl": stl,
        "reasoning_chain": reasoning_chain,
        "blocked": stl.blocked,
        "abort": stl.blocked,
        "logs": [
            f"STL: {stl.final_side} conf={stl.confidence_pct:.1f}% "
            f"{'BLOCKED: ' + stl.block_reason if stl.blocked else 'APPROVED'}"
        ],
    }


async def node_rationale(state: SentinelState) -> dict:
    stl = state.get("stl")
    prompt = (
        f"Symbol: {state['symbol']} | Side: {stl.final_side if stl else 'HOLD'} | "
        f"Confidence: {stl.confidence_pct:.1f}% | "
        f"Bull score: {stl.bull_score:.2f} | Bear score: {stl.bear_score:.2f}\n"
        f"Key debate:\n{stl.rationale if stl else ''}\n"
        f"R-Mem context:\n{state.get('r_mem_context', '')[:300]}"
    )
    rationale = await nim_chat(
        [{"role": "system", "content": _SUPERVISOR_RATIONALE_SYSTEM},
         {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=200,
    )
    return {"rationale": rationale, "logs": [f"Supervisor: {rationale[:80]}..."]}


async def node_dispatch(state: SentinelState, gateway_url: str) -> dict:
    """POST the trade signal to the Go Gateway."""
    stl = state["stl"]
    if not stl or state.get("abort"):
        return {"logs": ["Dispatch: skipped (blocked or abort)"]}

    closes_arr = np.array(state["closes"])
    returns    = np.diff(np.log(closes_arr + 1e-10)).tolist()

    payload = {
        "session_id":      state["session_id"],
        "symbol":          state["symbol"],
        "side":            stl.final_side,
        "current_price":   state["current_price"],
        "consensus_score": stl.consensus_score,
        "confidence_pct":  stl.confidence_pct,
        "atr_14":          float(state.get("quant_analysis", {}).get("gk_vol", 0) or 0),
        "rationale":       state.get("rationale", ""),
        "return_series":   returns[-252:],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{gateway_url}/v1/trade", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return {"logs": [f"Dispatch: approved={data.get('approved')} → gateway"]}
    except Exception as exc:
        log.error("Gateway dispatch failed: %s", exc)
        return {"logs": [f"Dispatch: FAILED — {exc}"]}


def _should_abort(state: SentinelState) -> str:
    return "abort" if state.get("abort") or state.get("blocked") else "continue"


# ── Graph builder ─────────────────────────────────────────────────────────────
def build_sentinel_graph(r_mem: RMem, gateway_url: str) -> Any:
    g = StateGraph(SentinelState)

    g.add_node("r_mem",    lambda s: node_r_mem(s, r_mem))
    g.add_node("bull",     lambda s: node_bull(s, r_mem))
    g.add_node("bear",     lambda s: node_bear(s, r_mem))
    g.add_node("quant",    node_quant)
    g.add_node("stl",      node_stl)
    g.add_node("rationale",node_rationale)
    g.add_node("dispatch", lambda s: node_dispatch(s, gateway_url))

    g.set_entry_point("r_mem")
    # Parallel fan-out after R-Mem context retrieval
    g.add_edge("r_mem", "bull")
    g.add_edge("r_mem", "bear")
    g.add_edge("r_mem", "quant")
    # Fan-in to STL
    g.add_edge("bull",  "stl")
    g.add_edge("bear",  "stl")
    g.add_edge("quant", "stl")

    g.add_conditional_edges("stl", _should_abort, {"abort": END, "continue": "rationale"})
    g.add_edge("rationale", "dispatch")
    g.add_edge("dispatch",  END)

    return g.compile()


# ── Supervisor orchestrator ───────────────────────────────────────────────────
class SentinelSupervisor:
    def __init__(self, gateway_url: str = "http://localhost:8080") -> None:
        self._store   = FAISSStrategyMemory()
        self._r_mem   = RMem(self._store)
        self._graph   = build_sentinel_graph(self._r_mem, gateway_url)

    async def run(
        self,
        symbol: str,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> SentinelState:
        state: SentinelState = {
            "symbol":         symbol,
            "session_id":     str(uuid.uuid4()),
            "closes":         closes,
            "highs":          highs,
            "lows":           lows,
            "volumes":        volumes,
            "current_price":  closes[-1] if closes else 0.0,
            "bull_arg":       None,
            "bear_arg":       None,
            "fundamental":    None,
            "sentiment_sig":  None,
            "quant_analysis": None,
            "r_mem_context":  "",
            "stl":            None,
            "reasoning_chain": "",
            "rationale":      "",
            "blocked":        False,
            "abort":          False,
            "logs":           [],
        }

        result = await self._graph.ainvoke(state)

        for line in result.get("logs", []):
            log.info("[%s] %s", symbol, line)

        return result

    async def record_outcome(
        self,
        session_id: str,
        symbol: str,
        reasoning_chain: str,
        outcome: str,
        pnl_pct: float,
    ) -> dict:
        await self._r_mem.record_outcome(session_id, symbol, reasoning_chain, outcome, pnl_pct)
        return await self._r_mem.perform_post_mortem(
            session_id, symbol, reasoning_chain, outcome, pnl_pct
        )
