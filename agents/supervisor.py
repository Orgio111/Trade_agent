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
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph  # type: ignore[import]

from agents.council import CouncilAgent
from agents.execution import ExecutionAgent
from agents.features import FeatureExtractor, get_feature_extractor
from agents.fundamental import FundamentalAgent
from agents.memory_agent import MemoryAgent
from agents.paper_account import PaperAccount
from agents.quant_sentinel import QuantSentinelAgent
from agents.risk_engine import RiskEngine
from agents.sentiment import SentimentAgent
from agents.technical import TechnicalAgent

from core.config import get_settings
from core.kill_switch import KillSwitch
from core.models import (
    CouncilDecision,
    FeatureSignal,
    Order,
    PortfolioState,
    RiskReport,
    Side,
    TechnicalSignal,
)
from core.llm import llm_chat
from core.observability import (
    AGENT_LATENCY,
    PNL_GAUGE,
    DRAWDOWN_GAUGE,
)
from core.scheduler import TaskType, get_circuit_breaker_state, route
from core.sentinelx_bridge import (
    get_portfolio_heat as rust_portfolio_heat,
    subscribe_kill_switch as rust_subscribe_kill_switch,
)
from core.sentinelx_gateway import submit_trade as gateway_submit_trade
from memory.pipeline import SemanticMemoryPipeline, get_memory_pipeline
from memory.models import RetrievalPurpose


# ── Trade lifecycle: open position tracking ───────────────────────────────────
@dataclass
class OpenPosition:
    """Track an open position for close-detection in run_cycle()."""
    order_id: str
    session_id: str
    side: Side
    entry_price: float
    quantity: float
    opened_at: datetime = field(default_factory=datetime.utcnow)

logger = logging.getLogger(__name__)

# ── Global MLOps references (set by supervisor on init) ──────────────────────
_mlops_pipeline: Any | None = None
_mlops_scheduler: Any | None = None

# ── Rust kill switch subscription task ────────────────────────────────────────
_rust_ks_task: asyncio.Task | None = None
_rust_ks_active: bool = False


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
    feature_signal: FeatureSignal | None
    council_decision: CouncilDecision | None
    risk_report: RiskReport | None
    order: Order | None
    logs: Annotated[list[str], operator.add]
    abort: bool


# ── Global memory pipeline reference (set by supervisor on init) ─────────────
_memory_pipeline: SemanticMemoryPipeline | None = None


async def _get_memory_context(state: TradingState) -> str:
    """Retrieve TurboVec memory context for the current market state."""
    pipe = _memory_pipeline
    if not pipe:
        return ""
    try:
        ctx = await pipe.retrieve(
            symbol=state["symbol"],
            closes=np.array(state["closes"]),
            volumes=np.array(state["volumes"]) if state.get("volumes") else None,
            technical=state.get("technical_signal"),
            purpose=RetrievalPurpose.TRADE_VALIDATION,
        )
        return ctx.format_for_prompt() if ctx.results else ""
    except Exception as exc:
        logger.debug("Memory retrieval failed for %s: %s", state["symbol"], exc)
        return ""


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


async def node_features(state: TradingState) -> dict:
    extractor = get_feature_extractor()
    sig = extractor.compute(
        state["symbol"],
        current_price=state.get("current_price"),
    )
    return {"feature_signal": sig, "logs": [f"Features: OFI={sig.ofi or 0:+.3f} CVD={sig.cvd or 0:.0f} FR={sig.funding_rate or 0:+.5f} OI={sig.open_interest or 0:.0f}"]}


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
    # Retrieve memory context for this market state
    memory_ctx = await _get_memory_context(state)
    # Pass OHLC data so council can run QuantSentinel analysis
    closes_arr = np.array(state["closes"]) if state.get("closes") else None
    highs_arr = np.array(state["highs"]) if state.get("highs") else None
    lows_arr = np.array(state["lows"]) if state.get("lows") else None
    decision = await agent.deliberate(
        state["symbol"],
        closes=closes_arr,
        highs=highs_arr,
        lows=lows_arr,
        technical=state.get("technical_signal"),
        features=state.get("feature_signal"),
        memory_context=memory_ctx,
    )
    log = f"Council: {decision.final_side.value} consensus={decision.consensus_score:.2f}"
    if memory_ctx:
        log += " [memory-augmented]"
    return {
        "council_decision": decision,
        "logs": [log],
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

    # Build feature context if available
    feat_ctx = ""
    if decision.features:
        f = decision.features
        feat_ctx = (
            f"Features: OFI={f.ofi:+.3f}, CVD={f.cvd:.0f}, "
            f"Funding={f.funding_rate:+.5f}% (ann.), OI={f.open_interest:.0f}, "
            f"OI-price_corr={f.oi_price_delta_corr or 0:+.2f}, "
            f"trend={f.trend.value} conf={f.confidence:.2f}\n"
        )

    prompt = (
        f"Symbol: {decision.symbol}\n"
        f"Council final: {decision.final_side.value} (consensus={decision.consensus_score:.2f})\n"
        f"Debate:\n{debate_summary}\n"
        f"{feat_ctx}"
        f"Risk: VaR99={risk.var_99:.2%}, size_usd={risk.position_size_usd:.0f}, "
        f"stop={risk.stop_loss_price:.4f}, TP={risk.take_profit_price:.4f}\n"
        "In 2 sentences, explain the final trading rationale."
    )
    rationale = await llm_chat(
        [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=200
    )
    decision.rationale = rationale
    return {"council_decision": decision, "logs": [f"Supervisor rationale: {rationale[:80]}..."],}    


async def node_execute(state: TradingState) -> dict:
    if state.get("abort"):
        return {"logs": ["Execution: skipped (abort=True)"]}
    decision = state["council_decision"]
    risk = state["risk_report"]
    if not decision or not risk:
        return {"logs": ["Execution: missing decision or risk"]}

    # ── Try Go Gateway first ────────────────────────────────────────────────
    if get_settings().sentinelx_gateway_url:
        gateway_result = await gateway_submit_trade(
            session_id=decision.session_id,
            symbol=decision.symbol,
            side=decision.final_side.value,
            current_price=state["current_price"],
            consensus_score=decision.consensus_score,
            atr_14=decision.technical.atr_14 if decision.technical else None,
            return_series=state["closes"][-100:] if len(state["closes"]) > 0 else None,
            rationale=decision.rationale,
        )
        if gateway_result.approved:
            from core.models import Order, OrderStatus
            order = Order(
                symbol=decision.symbol,
                side=decision.final_side,
                quantity=risk.position_size_units,
                price=risk.stop_loss_price,
                order_id=gateway_result.order_id or "",
                status=OrderStatus.FILLED,
                avg_fill_price=risk.stop_loss_price,
                session_id=decision.session_id,
            )
            return {
                "order": order,
                "logs": [f"Gateway execution: {order.status.value} id={order.order_id}"],
            }
        else:
            log_msg = f"Gateway rejected trade: {gateway_result.reason}"
            logger.warning("Gateway execution rejected for %s: %s", decision.symbol, gateway_result.reason)
            # Fall through to local execution

    # ── Local execution engine ───────────────────────────────────────────────
    agent = _execution_agent or ExecutionAgent()
    order = await agent.execute(decision, risk, np.array(state["closes"]))

    # ── Store trade state in TurboVec memory ───────────────────────────────────
    pipe = _memory_pipeline
    if pipe and order:
        try:
            await pipe.store_market_state(
                symbol=decision.symbol,
                closes=np.array(state["closes"]),
                volumes=np.array(state["volumes"]) if state.get("volumes") else None,
                technical=decision.technical,
            )
        except Exception as exc:
            logger.debug("Failed to store market state memory: %s", exc)

    return {"order": order, "logs": [f"Execution: {order.status.value} @ {order.avg_fill_price:.4f}"]}


# ── Global execution agent reference (set on init) ─────────────────────────
_execution_agent: ExecutionAgent | None = None


def should_abort(state: TradingState) -> str:
    return "abort" if state.get("abort") else "continue"


# ── Graph assembly ─────────────────────────────────────────────────────────────
def build_trading_graph() -> Any:
    g = StateGraph(TradingState)

    g.add_node("technical", node_technical)
    g.add_node("features", node_features)
    g.add_node("fundamental", node_fundamental)
    g.add_node("sentiment", node_sentiment)
    g.add_node("council", node_council)
    g.add_node("risk", node_risk)
    g.add_node("supervisor", node_supervisor_decision)
    g.add_node("execute", node_execute)

    g.set_entry_point("technical")
    # Fan-out: technical → features, fundamental, sentiment in parallel
    g.add_edge("technical", "features")
    g.add_edge("technical", "fundamental")
    g.add_edge("technical", "sentiment")
    # Fan-in: all three feed council
    g.add_edge("features", "council")
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
    """Top-level orchestrator. Feeds market data into the LangGraph pipeline.

    Parameters
    ----------
    initial_equity:
        Starting capital for the live portfolio.
    paper_account:
        Optional :class:`PaperAccount` for paper trading. When provided, it is
        passed to the :class:`ExecutionAgent` and its snapshot is pushed to the
        dashboard for paper-mode visibility.
    """

    def __init__(self, initial_equity: float, paper_account: PaperAccount | None = None) -> None:
        global _memory_pipeline
        cfg = get_settings()
        self._graph = build_trading_graph()
        self._kill_switch = KillSwitch(initial_equity)
        self._portfolio = PortfolioState(
            equity=initial_equity,
            cash=initial_equity,
            peak_equity=initial_equity,
        )
        self._memory = MemoryAgent()
        self._open_positions: dict[str, OpenPosition] = {}

        # Paper trading account
        self._paper_account = paper_account

        # Initialize TurboVec semantic memory pipeline
        if cfg.memory_enabled:
            _memory_pipeline = get_memory_pipeline()
            logger.info("TurboVec semantic memory pipeline initialized")

        # Initialize feature extractor (lazy — started in self.start())
        if cfg.features_enabled:
            self._feature_extractor = get_feature_extractor()
            logger.info("FeatureExtractor initialized")
        else:
            self._feature_extractor = None

        # Pass paper account to the execution agent
        global _execution_agent
        self._execution_agent = ExecutionAgent(paper_account=paper_account)
        _execution_agent = self._execution_agent

        # Initialize MLOps pipeline (lazy — started in self.start())
        self._mlops_drift: Any = None
        self._mlops_pipeline: Any = None
        self._mlops_scheduler: Any = None
        if cfg.mlops_enabled:
            from mlops.drift_detector import DriftDetector
            from mlops.pipeline import MLOpsPipeline
            from mlops.scheduler import MLOpsScheduler
            self._mlops_drift = DriftDetector("ppo_execution")
            self._mlops_pipeline = MLOpsPipeline(self._mlops_drift)
            self._mlops_scheduler = MLOpsScheduler(self._mlops_drift, self._mlops_pipeline)
            logger.info("MLOps pipeline initialized (check_interval=%ds)", cfg.mlops_check_interval_s)

    async def start(self) -> None:
        global _memory_pipeline
        try:
            await self._memory.connect()
        except Exception as exc:
            logger.warning("Memory agent connection failed (continuing): %s", exc)
        if _memory_pipeline:
            try:
                await _memory_pipeline.start()
            except Exception as exc:
                logger.warning("Memory pipeline start failed (continuing): %s", exc)
        if self._feature_extractor:
            try:
                await self._feature_extractor.start()
            except Exception as exc:
                logger.warning("Feature extractor start failed (continuing): %s", exc)
        # Start MLOps scheduler
        if self._mlops_scheduler:
            try:
                await self._mlops_scheduler.start()
            except Exception as exc:
                logger.warning("MLOps scheduler start failed (continuing): %s", exc)

        # ── Subscribe to Rust kill switch events ───────────────────────────
        global _rust_ks_task, _rust_ks_active
        try:
            stream = await rust_subscribe_kill_switch("python-supervisor")
            if stream is not None:
                async def _ks_listener():
                    global _rust_ks_active
                    try:
                        async for event in stream:
                            if event.active:
                                _rust_ks_active = True
                                logger.critical(
                                    "Rust kill switch TRIGGERED: %s (dd=%.1f%%)",
                                    event.reason, event.drawdown_pct * 100,
                                )
                                # Also trigger Python-side kill switch
                                if hasattr(self, '_kill_switch'):
                                    self._kill_switch._active = True
                    except Exception as exc:
                        logger.warning("Rust kill switch listener ended: %s", exc)
                        _rust_ks_active = False

                _rust_ks_task = asyncio.create_task(_ks_listener())
                logger.info("Subscribed to Rust kill switch events")
        except Exception as exc:
            logger.debug("Rust kill switch subscription failed: %s", exc)

        logger.info("PortfolioSupervisor started — equity=%.2f", self._portfolio.equity)

    async def stop(self) -> None:
        """Graceful shutdown: cancel background tasks and close connections."""
        global _rust_ks_task
        if _rust_ks_task is not None:
            _rust_ks_task.cancel()
            try:
                await _rust_ks_task
            except asyncio.CancelledError:
                pass
            _rust_ks_task = None
            logger.info("Rust kill switch listener cancelled")
        try:
            from core.sentinelx_bridge import close as close_bridge
            await close_bridge()
            logger.debug("Sentinel-X bridge closed")
        except Exception:
            pass
        logger.info("PortfolioSupervisor stopped")

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
            "feature_signal": None,
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
        decision = result.get("council_decision")
        risk_report = result.get("risk_report")

        # ── Trade close detection ─────────────────────────────────────────────
        existing = self._open_positions.get(symbol)
        if existing and decision:
            should_close = (
                (order is not None and order.side != existing.side)
                or (decision.final_side == Side.HOLD and order is None)
            )
            if should_close:
                outcome = await self._memory.record_close(existing.order_id, closes[-1])
                if outcome:
                    logger.info(
                        "Closed %s %s PnL=%.2f (%s)",
                        symbol, existing.side.value, outcome.pnl or 0,
                        "WIN" if outcome.win else "LOSS",
                    )
                    pipe = _memory_pipeline
                    if pipe:
                        try:
                            vols = np.array(volumes) if volumes else None
                            await pipe.store_trade_outcome(
                                decision, outcome, np.array(closes), vols
                            )
                        except Exception as exc:
                            logger.debug("Failed to store trade outcome memory: %s", exc)
                del self._open_positions[symbol]
                order = None  # prevent immediate flip

        # ── Trade open ────────────────────────────────────────────────────────
        if order and decision and risk_report and symbol not in self._open_positions:
            await self._memory.record_open(
                order,
                decision.model_dump(mode="json"),
                risk_report.model_dump(mode="json"),
            )
            self._open_positions[symbol] = OpenPosition(
                order_id=order.order_id,
                session_id=order.session_id,
                side=order.side,
                entry_price=order.avg_fill_price or closes[-1],
                quantity=order.quantity,
            )

        # Update portfolio state
        self._portfolio.timestamp = datetime.utcnow()
        PNL_GAUGE.labels(symbol=symbol).set(self._portfolio.daily_pnl)
        DRAWDOWN_GAUGE.set(self._portfolio.current_drawdown_pct)

        # ── Query Rust portfolio heat ──────────────────────────────────────
        rust_heat: float | None = None
        try:
            heat = await rust_portfolio_heat(
                symbol=symbol,
                current_price=closes[-1] if closes else 0.0,
                portfolio_equity=self._portfolio.equity,
                returns=np.diff(np.log(np.array(closes) + 1e-10)) if len(closes) > 5 else None,
            )
            if heat is not None:
                rust_heat = heat
        except Exception as exc:
            logger.debug("Rust portfolio heat query failed: %s", exc)

        # ── Push paper account snapshot to dashboard ────────────────────────
        if self._paper_account:
            self._paper_account.mark_to_market(symbol, closes[-1] if closes else 0.0)
            try:
                from web.server import get_dashboard_state

                ds = get_dashboard_state()
                await ds.update_paper_state(self._paper_account.snapshot())
            except Exception as exc:
                logger.debug("Dashboard paper update failed: %s", exc)

        # ── Push data to dashboard state ───────────────────────────────────────
        try:
            from web.server import get_dashboard_state
            ds = get_dashboard_state()
            await ds.update_portfolio(self._portfolio)
            await ds.update_price(symbol, state["current_price"])
            if decision:
                await ds.update_council_decision(symbol, decision)
            if risk_report:
                risk_dict = {
                    "var_95": risk_report.var_95,
                    "var_99": risk_report.var_99,
                    "cvar_99": risk_report.cvar_99,
                    "kelly_fractional": risk_report.kelly_fractional,
                    "position_size_usd": risk_report.position_size_usd,
                    "stop_loss_price": risk_report.stop_loss_price,
                    "take_profit_price": risk_report.take_profit_price,
                }
                await ds.update_risk(symbol, risk_dict)
            if order:
                await ds.add_order(order)
            if result.get("feature_signal"):
                await ds.update_features(symbol, result["feature_signal"])
            # Update agent signals from the graph logs
            for log_line in result.get("logs", []):
                if ":" in log_line:
                    agent_name = log_line.split(":")[0].strip().lower()
                    await ds.update_agent_signal(agent_name, {
                        "status": log_line,
                        "side": decision.final_side.value if decision else "HOLD",
                        "confidence": decision.consensus_score if decision else 0.0,
                    })

            # ── Push Sentinel-X scheduler & Rust metrics ──────────────────────
            await ds.update_sentinelx_state({
                "circuit_breaker": get_circuit_breaker_state(),
                "rust_ks_active": _rust_ks_active,
                "rust_portfolio_heat": rust_heat,
            })
        except Exception as exc:
            logger.debug("Dashboard state update failed: %s", exc)

        triggered = self._kill_switch.update(self._portfolio)
        if triggered:
            self._portfolio.kill_switch_active = True
            await self._kill_switch.broadcast_halt()

        return order
