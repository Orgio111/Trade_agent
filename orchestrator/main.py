"""
QUANTEX AI Orchestrator v1.0 — Full system with swarm intelligence,
RL training, vector memory, risk engine, and data pipeline.
"""
import os
import json
import asyncio
import time as time_module
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()  # Load .env file for API keys (GROQ, NVIDIA, OPENROUTER)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .nim_client import NIMOrchestrator
from .nim_enhanced import EnhancedNIMOrchestrator, NeMoDistiller
from .inference_integration import InferenceIntegration
from .strategy import StrategyEngine, Signal
from .agents import (
    MarketAnalystAgent, RiskGuardianAgent, SentimentAgent,
    DeepSeekAnalysisAgent, AgentOpinion,
)
from .backtest import BacktestEngine, DataLoader, BacktestResult
from .paper_account import PaperAccount, OrderRequest
from .position_manager import PositionManager, TradePlan, build_entry_ladder, build_exit_structure
from .feature_engine import FeatureEngine
from .ml_signals import MLSignalEngine
from .database import Database, TradeRecord, PositionRecord, StrategyPerfRecord
from .metrics import (
    metrics_endpoint, record_trade, update_portfolio_metrics,
    record_agent_latency, update_ml_metrics,
    update_adaptive_routing_metrics, record_chain_adaptation,
)
from .swarm.debate_system import AgentSwarm, ScalpingAgent, SwingAgent, RegimeAgent, AnomalyAgent
from .memory.vector_memory import TradingMemorySystem, AgentCredibilityTracker
from .risk.risk_engine import RiskEngine, PanicMode, AntiOvertradeSystem, RiskParams
from .risk.risk_engine_v2 import EnhancedRiskEngine, EnhancedRiskParams, KillSwitch, KellySizer
from .rl.trading_env import TradingEnvironment
from .rl.strategy_evolver import StrategyEvolver, OptunaOptimizer
from .data_pipeline import DataPipeline
from .timesfm_forecaster import TimesFMForecaster
from .market_structure import MarketStructureEngine, FakeBreakoutDetector
from .microstructure import SpoofingDetector, HiddenLiquidityDetector, DeltaCVDTracker, LiquidationCascadePredictor, OrderBookImbalanceAnalyzer
from .execution import ExecutionOrchestrator, TWAPExecutor, VWAPExecutor, IcebergExecutor, SmartOrderRouter, SlippageEstimator, get_execution_plan, ExecutionPlan
from .agent_routing import AgentModelRouter, TaskRouter, AGENT_MODEL_MAP, AGENT_PROVIDER_CHAINS, AGENT_TASK_TYPE_OVERRIDES

# Trinity Layer A: Brain registry + NATS runner
from .brains import BRAIN_REGISTRY, BaseBrain
from .brains.base_brain import BrainRunner, BrainNATSPublisher


# ── Per-brain intervals (seconds) — used to set runner cadence ──
BRAIN_INTERVALS: dict[str, float] = {
    "timesfm": 60.0,              # Heavy model inference
    "freqai": 15.0,               # ML model
    "llm_regime": 15.0,           # LLM-based classification
    "microstructure": 10.0,       # Fast tick-level analysis
    "finbert": 30.0,              # NLP news sentiment
    "finrl": 30.0,                # RL position sizing
    "onchain": 60.0,              # Slow on-chain data
    "statarb": 15.0,              # Statistical arbitrage
    "orderflow_nautilus": 5.0,    # Ultra-fast orderbook
    "polymarket_alpha": 60.0,     # Prediction market data
    "custom_nn": 30.0,            # Neural network inference
}

# ── Global State ──────────────────────────────────────────

nim: NIMOrchestrator | None = None
nim_enhanced: EnhancedNIMOrchestrator | None = None
nim_integration: InferenceIntegration | None = None
strategy_engine: StrategyEngine | None = None
market_analyst: MarketAnalystAgent | None = None
deepseek_analyst: DeepSeekAnalysisAgent | None = None
risk_guardian: RiskGuardianAgent | None = None
sentiment_agent: SentimentAgent | None = None
paper_account: PaperAccount | None = None
position_manager: PositionManager | None = None
backtest_engine: BacktestEngine | None = None
feature_engine: FeatureEngine | None = None
ml_engine: MLSignalEngine | None = None
database: Database | None = None

# v1.0 components
swarm: AgentSwarm | None = None
memory: TradingMemorySystem | None = None
credibility_tracker: AgentCredibilityTracker | None = None
risk_engine: RiskEngine | None = None
panic_mode: PanicMode | None = None
anti_overtrade: AntiOvertradeSystem | None = None
data_orchestrator: DataPipeline | None = None
nemo_distiller: NeMoDistiller | None = None

# Scalping & Swing agents (used directly)
scalping_agent: ScalpingAgent | None = None
swing_agent: SwingAgent | None = None
regime_agent: RegimeAgent | None = None
anomaly_agent: AnomalyAgent | None = None

# Institutional v2 components
enhanced_risk: EnhancedRiskEngine | None = None
market_structure: MarketStructureEngine | None = None
fake_breakout: FakeBreakoutDetector | None = None
spoofing_detector: SpoofingDetector | None = None
hidden_liquidity: HiddenLiquidityDetector | None = None
delta_tracker: DeltaCVDTracker | None = None
liq_cascade: LiquidationCascadePredictor | None = None
ob_imbalance: OrderBookImbalanceAnalyzer | None = None
execution_orch: ExecutionOrchestrator | None = None
agent_router: AgentModelRouter | None = None
task_router: TaskRouter | None = None
timesfm_forecaster: TimesFMForecaster | None = None

# Trinity Layer A: Brain runner instances
brain_publisher: BrainNATSPublisher | None = None
brain_runners: dict[str, BrainRunner] = {}
brain_runner_tasks: dict[str, asyncio.Task] = {}

active_connections: list[WebSocket] = []


async def _compute_signal(
    symbol: str = "BTCUSDT",
    source: str = "strategy",
) -> dict:
    """Fetch real Binance data and compute a trading signal.

    Shared helper used by both the REST /api/v1/signal endpoint
    and the WebSocket signal broadcast task.
    Falls back to mock data if Binance API is unavailable.
    """
    import numpy as np
    import pandas as pd

    # Try real Binance data first, fallback to mock
    interval = "1h"
    end = datetime.now()
    start = end - timedelta(hours=100)  # ~100 hourly candles
    try:
        df = await DataLoader.from_binance_api(
            symbol=symbol, interval=interval, start_time=start, end_time=end
        )
        source_label = "Binance API"
    except Exception:
        df = DataLoader.generate_mock_data(periods=100, start_price=50000.0)
        source_label = "Mock (Binance unavailable)"
    if df.empty:
        df = DataLoader.generate_mock_data(periods=100, start_price=50000.0)
        source_label = "Mock (empty response from Binance)"

    if source == "timesfm" and timesfm_forecaster:
        fc = timesfm_forecaster.forecast_dataframe(df, horizon=24)
        return {
            "symbol": symbol, "source": "timesfm", "data_source": source_label,
            "signal": fc.direction, "confidence": fc.confidence,
            "entry_price": fc.last_price,
            "reason": (fc.error or f"TimesFM expected return {fc.expected_return:+.4%} over {fc.horizon} steps"),
            "metadata": {
                "expected_return": fc.expected_return,
                "point_forecast": fc.point_forecast[:24],
                "model": fc.model,
                "available": fc.available,
            },
        }
    if source == "ml" and ml_engine:
        signal = ml_engine.predict_signal(df)
    elif source == "swarm" and swarm:
        context = {
            "symbol": symbol, "price": float(df["close"].iloc[-1]),
            "df": df, "regime": "unknown",
            "portfolio": paper_account.get_portfolio() if paper_account else {},
        }
        decision = await swarm.run_swarm_debate(context)
        return {
            "symbol": symbol, "source": "swarm", "data_source": source_label,
            "signal": decision.direction, "confidence": decision.confidence,
            "entry_price": df["close"].iloc[-1], "reason": decision.reasoning[:200],
            "debate_log": decision.debate_log[:10],
        }
    elif strategy_engine:
        signal = strategy_engine.generate_signal(df)
    else:
        return {"error": "No signal source available"}

    return {
        "symbol": symbol, "source": source, "data_source": source_label,
        "signal": signal.direction, "confidence": signal.confidence,
        "entry_price": signal.entry_price, "stop_loss": signal.stop_loss,
        "take_profits": signal.take_profits, "reason": signal.reason,
        "metadata": signal.metadata,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nim, nim_enhanced, nim_integration, strategy_engine, market_analyst, deepseek_analyst
    global risk_guardian, sentiment_agent, scalping_agent, swing_agent
    global regime_agent, anomaly_agent
    global paper_account, position_manager, backtest_engine, feature_engine
    global ml_engine, database
    global swarm, memory, credibility_tracker, risk_engine, panic_mode
    global anti_overtrade, data_orchestrator, nemo_distiller
    global enhanced_risk, market_structure, fake_breakout
    global spoofing_detector, hidden_liquidity, delta_tracker, liq_cascade, ob_imbalance
    global execution_orch, agent_router, task_router
    global timesfm_forecaster
    global brain_publisher, brain_runners, brain_runner_tasks

    # Phase 1-3 components (nim_enhanced kept for NeMoDistiller backward compat)
    nim = InferenceIntegration(budget_tier=os.getenv("INFERENCE_BUDGET_TIER", "free"))
    nim_enhanced = EnhancedNIMOrchestrator()
    nim_integration = nim  # Alias for clarity — same InferenceIntegration instance
    strategy_engine = StrategyEngine()

    # Agents now use the new InferenceIntegration (backward compatible interface)
    market_analyst = MarketAnalystAgent(nim_integration)
    deepseek_analyst = DeepSeekAnalysisAgent(nim_integration)
    risk_guardian = RiskGuardianAgent(nim_integration)
    sentiment_agent = SentimentAgent()

    # Individual agents
    scalping_agent = ScalpingAgent(nim_integration)
    swing_agent = SwingAgent(nim_integration)
    regime_agent = RegimeAgent(nim_integration)
    anomaly_agent = AnomalyAgent(nim_integration)

    # Trading components
    paper_account = PaperAccount(
        initial_balance=float(os.getenv("INITIAL_BALANCE", "1000.0")),
    )
    position_manager = PositionManager()
    backtest_engine = BacktestEngine(
        initial_balance=float(os.getenv("INITIAL_BALANCE", "1000.0")),
    )
    feature_engine = FeatureEngine()

    # ML
    ml_engine = MLSignalEngine()
    ml_engine.load_model()

    # Database
    database = Database()
    try:
        await database.connect()
        print("✅ PostgreSQL connected")
    except Exception as e:
        print(f"⚠️  PostgreSQL unavailable: {e}")

    # v1.0: Swarm Intelligence
    memory = TradingMemorySystem()
    credibility_tracker = AgentCredibilityTracker(memory)
    swarm = AgentSwarm(nim_integration)
    risk_engine = RiskEngine()
    panic_mode = PanicMode()
    anti_overtrade = AntiOvertradeSystem()
    data_orchestrator = DataPipeline()
    nemo_distiller = NeMoDistiller()

    # Institutional v2 components
    enhanced_risk = EnhancedRiskEngine()
    market_structure = MarketStructureEngine()
    fake_breakout = FakeBreakoutDetector()
    spoofing_detector = SpoofingDetector()
    hidden_liquidity = HiddenLiquidityDetector()
    delta_tracker = DeltaCVDTracker()
    liq_cascade = LiquidationCascadePredictor()
    ob_imbalance = OrderBookImbalanceAnalyzer()
    execution_orch = ExecutionOrchestrator()
    agent_router = AgentModelRouter()
    task_router = TaskRouter()

    # TimesFM forecaster (model weights load lazily on first forecast)
    timesfm_forecaster = TimesFMForecaster()

    # ── Trinity Layer A: Brain Runners ────────────────────────────────
    # Each brain runs independently, publishing signals to NATS JetStream "signals.raw"
    # The Go orchestrator (Layer B) aggregates these into a weighted signal.
    brain_publisher = BrainNATSPublisher()
    brain_runners = {}
    brain_runner_tasks = {}

    try:
        await brain_publisher.connect()
        print(f"  ✅ NATS JetStream connected — publishing {len(BRAIN_REGISTRY)} brains to signals.raw")

        for brain_id, brain_class in BRAIN_REGISTRY.items():
            try:
                brain_instance: BaseBrain = brain_class()
                interval = BRAIN_INTERVALS.get(brain_id, 15.0)
                runner = BrainRunner(brain_instance, brain_publisher, interval_secs=interval)
                brain_runners[brain_id] = runner
                # Start as background task
                task = asyncio.create_task(runner.run(symbol="BTCUSDT"), name=f"brain:{brain_id}")
                brain_runner_tasks[brain_id] = task
                print(f"    ✅ Brain [{brain_id}] started — interval={interval}s")
            except Exception as e:
                print(f"    ❌ Brain [{brain_id}] failed to start: {e}")

    except Exception as e:
        print(f"  ⚠️  NATS unavailable — brains will not publish: {e}")
        print(f"     {len(BRAIN_REGISTRY)} brains registered but not connected to NATS")

    print("  Institutional v2: EnhancedRisk, MarketStructure, Execution, Microstructure, AgentRouting")

    # Start data feeds
    asyncio.create_task(data_orchestrator.start())

    #    # WebSocket broadcast: inference routing data
    async def _ws_routing_broadcast():
        while True:
            await asyncio.sleep(5)
            if nim_integration and active_connections:
                try:
                    # Reuse the same logic as the REST endpoint
                    provider_health = await nim_integration.get_provider_health()
                    agent_chains = {}
                    for agent_id, chain in sorted(AGENT_PROVIDER_CHAINS.items()):
                        if agent_id == "default":
                            continue
                        adapted = nim_integration.agent_router.get_optimal_provider_chain(agent_id, chain)
                        agent_chains[agent_id] = {
                            "base_chain": chain,
                            "adapted_chain": adapted,
                            "task_override": AGENT_TASK_TYPE_OVERRIDES.get(agent_id),
                            "adapted": adapted != chain,
                        }
                    agent_chains["default"] = {
                        "base_chain": AGENT_PROVIDER_CHAINS.get("default", []),
                        "adapted_chain": None,
                        "task_override": None,
                        "adapted": False,
                    }
                    payload = {
                        "type": "inference_routing",
                        "provider_health": provider_health,
                        "agent_chains": agent_chains,
                        "adaptive_routing": nim_integration.agent_router.get_adaptive_routing_summary(),
                        "agent_mappings": nim_integration.get_agent_model_summary(),
                        "cost_usage": {
                            "today": nim_integration.get_todays_usage(),
                            "cache": nim_integration.get_cache_stats(),
                            "over_budget": nim_integration.is_over_budget(),
                        },
                    }
                    await broadcast(payload)
                except Exception:
                    pass

    asyncio.create_task(_ws_routing_broadcast())

    # WebSocket broadcast: portfolio data (replaces HTTP polling)
    async def _ws_portfolio_broadcast():
        while True:
            await asyncio.sleep(5)
            if paper_account and active_connections:
                try:
                    p = paper_account.get_portfolio()
                    payload = {"type": "portfolio", **p}
                    await broadcast(payload)
                except Exception:
                    pass

    asyncio.create_task(_ws_portfolio_broadcast())

    # WebSocket broadcast: live signal data (replaces HTTP polling)
    async def _ws_signal_broadcast():
        while True:
            await asyncio.sleep(15)
            if strategy_engine and active_connections:
                try:
                    result = await _compute_signal(symbol="BTCUSDT", source="ml")
                    if "error" not in result:
                        payload = {"type": "signal", **result}
                        await broadcast(payload)
                except Exception:
                    pass

    asyncio.create_task(_ws_signal_broadcast())

    # Metrics loop
    update_portfolio_metrics(
        balance=paper_account.balance, equity=paper_account.balance,
        drawdown=0.0, open_positions=0, consec_losses=0, total_pnl=0.0,
    )

    async def _metrics_loop():
        last_adapted_state: dict[str, list[str]] = {}
        while True:
            await asyncio.sleep(5)
            if paper_account:
                p = paper_account.get_portfolio()
                update_portfolio_metrics(
                    balance=p["balance"], equity=p["equity"],
                    drawdown=p["drawdown"], open_positions=p["open_positions"],
                    consec_losses=p["consecutive_losses"], total_pnl=p["total_pnl"],
                )

            # Adaptive routing metrics (sync router → Prometheus)
            if nim_integration and nim_integration.agent_router:
                update_adaptive_routing_metrics(nim_integration.agent_router)

                # Detect chain adaptation events
                for agent_id, base_chain in AGENT_PROVIDER_CHAINS.items():
                    if agent_id == "default":
                        continue
                    current = nim_integration.agent_router.get_optimal_provider_chain(
                        agent_id, base_chain
                    )
                    prev = last_adapted_state.get(agent_id)
                    if prev is not None and current != prev:
                        record_chain_adaptation(agent_id)
                    last_adapted_state[agent_id] = current

    asyncio.create_task(_metrics_loop())

    print("✅ QUANTEX AI Orchestrator v1.0 initialized")
    print(f"   Paper balance: ${paper_account.balance:.2f}")
    print(f"   ML model: {'loaded' if ml_engine._model else 'not trained'}")
    print(f"   Memory: {'Qdrant stub' if memory else 'unavailable'}")
    print("   Data feeds: BinanceWS + Funding + OpenInterest")
    print(f"   Inference: multi-provider router (Groq + NVIDIA NIM + OpenRouter)")
    print(f"   Budget tier: {os.getenv('INFERENCE_BUDGET_TIER', 'free')}")
    yield

    # Cleanup: stop all brain runners and disconnect NATS
    if brain_runners:
        print(f"  Stopping {len(brain_runners)} brain runners...")
        for brain_id, runner in brain_runners.items():
            runner.stop()
        # Cancel runner tasks
        for brain_id, task in brain_runner_tasks.items():
            task.cancel()
        # Give tasks a moment to finish
        if brain_runner_tasks:
            await asyncio.sleep(0.5)
    if brain_publisher:
        await brain_publisher.close()
        print("  ✅ NATS publisher disconnected")

    if database:
        await database.disconnect()
    if data_orchestrator:
        await data_orchestrator.stop()


app = FastAPI(
    title="QUANTEX AI Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",       # Next.js dev server
        "http://localhost:3001",       # Grafana (if needed)
        "http://frontend:3000",        # Docker Compose service
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await websocket.send_json({"status": "ok", "echo": message})
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast(event: dict):
    for conn in active_connections:
        try:
            await conn.send_json(event)
        except Exception:
            pass


# ── Health ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    db_status = "unknown"
    if database:
        db_health = await database.health()
        db_status = db_health["status"]
    return {
        "status": "ok",
        "service": "quantex-orchestrator",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "paper",
        "swarm": swarm is not None,
        "memory": memory is not None,
        "data_feeds": len(data_orchestrator._tasks) if data_orchestrator else 0,
        "ml": {"trained": ml_engine._model is not None if ml_engine else False},
        "database": db_status,
        "balance": paper_account.balance if paper_account else 0,
        "brains": {
            "registered": len(BRAIN_REGISTRY),
            "runners_active": len(brain_runners),
            "nats_connected": (
                brain_publisher is not None
                and brain_publisher.nc is not None
                and brain_publisher.nc.is_connected
            ),
            "brain_ids": list(BRAIN_REGISTRY.keys()),
            "intervals_sec": {k: BRAIN_INTERVALS.get(k, 15.0) for k in BRAIN_REGISTRY},
        },
    }


# ── Status ─────────────────────────────────────────────────

@app.get("/api/v1/status")
async def system_status():
    db_status = "unknown"
    if database:
        db_health = await database.health()
        db_status = db_health["status"]

    status = {
        "version": "1.0.0",
        "mode": "paper",
        "services": {
            "nim": nim is not None,
            "nim_enhanced": nim_enhanced is not None,
            "nim_integration": nim_integration is not None,
            "strategy": strategy_engine is not None,
            "swarm": swarm is not None,
            "memory": memory is not None,
            "risk_engine": risk_engine is not None,
            "panic_mode": panic_mode is not None,
            "paper_account": paper_account is not None,
            "backtest_engine": backtest_engine is not None,
            "ml_engine": ml_engine is not None,
            "data_pipeline": data_orchestrator is not None,
            "database": db_status,
        },
        "websocket_clients": len(active_connections),
    }

    if paper_account:
        portfolio = paper_account.get_portfolio()
        status["portfolio"] = portfolio
        update_portfolio_metrics(
            balance=portfolio["balance"], equity=portfolio["equity"],
            drawdown=portfolio["drawdown"], open_positions=portfolio["open_positions"],
            consec_losses=portfolio["consecutive_losses"], total_pnl=portfolio["total_pnl"],
        )

    if ml_engine and ml_engine._model:
        status["ml"] = {"trained": True, "train_count": ml_engine._train_count}

    return status


# ── API: Portfolio ──────────────────────────────────────────────────

@app.get("/api/v1/portfolio")
async def get_portfolio():
    if not paper_account:
        return {"error": "Not initialized"}
    return paper_account.get_portfolio()


@app.get("/api/v1/positions")
async def get_positions():
    if not paper_account:
        return {"error": "Not initialized"}
    return {"positions": paper_account.get_positions()}


@app.get("/api/v1/trades")
async def get_trades(limit: int = Query(50, ge=1, le=200)):
    if not paper_account:
        return {"error": "Not initialized"}
    return {"trades": paper_account.get_trades(limit=limit)}


@app.post("/api/v1/account/reset")
async def reset_account(balance: float = Query(1000.0)):
    if not paper_account:
        return {"error": "Not initialized"}
    paper_account.reset(balance)
    return {"status": "ok", "balance": paper_account.balance}


# ── API: Price ──────────────────────────────────────────────────────

@app.post("/api/v1/price")
async def update_price(data: dict):
    if not paper_account:
        return {"error": "Not initialized"}
    symbol = data.get("symbol", "BTCUSDT")
    bid = data.get("bid", data.get("price", 0))
    ask = data.get("ask", data.get("price", 0))
    last = data.get("price", bid)
    paper_account.update_price(symbol, bid, ask, last)
    return {"status": "ok"}


# ── API: Signal ─────────────────────────────────────────────────────

@app.get("/api/v1/signal")
async def get_signal(symbol: str = "BTCUSDT", source: str = "strategy"):
    return await _compute_signal(symbol, source)


# ── API: Swarm ──────────────────────────────────────────────────────

@app.post("/api/v1/swarm/debate")
async def swarm_debate(context: dict):
    if not swarm:
        return {"error": "Swarm not initialized"}
    decision = await swarm.run_swarm_debate(context)
    return {
        "direction": decision.direction,
        "confidence": decision.confidence,
        "vote_breakdown": decision.vote_breakdown,
        "risk_verdict": decision.risk_verdict,
        "dissents": decision.dissents,
        "debate_log": decision.debate_log,
    }


@app.get("/api/v1/swarm/agents")
async def swarm_agents():
    if not swarm:
        return {"error": "Swarm not initialized"}
    return {"agents": list(swarm.agents.keys())}


# ── API: Agents ─────────────────────────────────────────────────────

@app.post("/api/v1/agent/deepseek-analyze")
async def deepseek_analyze(context: dict):
    if not deepseek_analyst:
        return {"error": "Not initialized"}
    opinion = await deepseek_analyst.analyze(
        symbol=context.get("symbol", "BTCUSDT"),
        price=context.get("price", 50000),
        indicators=context.get("indicators", {}),
        regime=context.get("regime", "unknown"),
        sentiment=context.get("sentiment"),
        memory_context=context.get("memory_context"),
    )
    return {
        "agent": opinion.agent_id, "signal": opinion.signal,
        "confidence": opinion.confidence, "reasoning": opinion.reasoning,
        "metadata": opinion.metadata,
    }


@app.post("/api/v1/agent/risk-check")
async def risk_check(signal_data: dict):
    if not risk_guardian:
        return {"error": "Not initialized"}
    sig = Signal(
        direction=signal_data.get("direction", "hold"),
        confidence=signal_data.get("confidence", 0.5),
        entry_price=signal_data.get("entry_price"),
        stop_loss=signal_data.get("stop_loss"),
        take_profits=signal_data.get("take_profits", []),
    )
    portfolio = paper_account.get_portfolio() if paper_account else {}
    opinion = await risk_guardian.evaluate(sig, portfolio)
    return {"approved": opinion.signal != "hold", "reason": opinion.reasoning, "size_modifier": opinion.confidence}


@app.get("/api/v1/agent/sentiment")
async def get_sentiment():
    if not sentiment_agent:
        return {"error": "Not initialized"}
    opinion = await sentiment_agent.analyze()
    return {"signal": opinion.signal, "confidence": opinion.confidence,
            "reasoning": opinion.reasoning, "metadata": opinion.metadata}


# ── API: Risk Engine ────────────────────────────────────────────────

@app.post("/api/v1/risk/check")
async def risk_check_all(trade: dict):
    if not risk_engine or not paper_account:
        return {"error": "Risk engine not initialized"}
    portfolio = paper_account.get_portfolio()
    result = risk_engine.check_all_gates(trade, portfolio)
    return {
        "approved": result.approved,
        "severity": result.severity,
        "reason": result.reason,
        "size_multiplier": result.size_multiplier,
        "max_leverage": result.max_leverage,
    }


@app.get("/api/v1/risk/panic")
async def check_panic():
    if not panic_mode or not paper_account:
        return {"error": "Not initialized"}
    market_conditions: dict = {}
    if data_orchestrator:
        funding = data_orchestrator.get_current_funding("BTCUSDT")
        if funding:
            market_conditions["funding_rate"] = funding.get("funding_rate", 0)
    portfolio = paper_account.get_portfolio()
    return panic_mode.assess(market_conditions, portfolio)


@app.get("/api/v1/risk/overtrade")
async def check_overtrade():
    if not anti_overtrade:
        return {"error": "Not initialized"}
    return anti_overtrade.check()


# ── API: Memory ─────────────────────────────────────────────────────

@app.post("/api/v1/memory/store")
async def store_memory(data: dict):
    if not memory:
        return {"error": "Memory not initialized"}
    memory.store_trade_memory(
        data.get("trade", {}), data.get("outcome", {})
    )
    return {"status": "ok"}


@app.post("/api/v1/memory/recall")
async def recall_memory(context: dict):
    if not memory:
        return {"error": "Memory not initialized"}
    similar = memory.recall_similar_setups(context)
    stats = memory.get_pattern_stats(similar)
    return {"similar_trades": similar, "stats": stats}


# ── API: RL Training ────────────────────────────────────────────────

@app.post("/api/v1/rl/train")
async def rl_train(params: dict = {}):
    if not ml_engine:
        return {"error": "ML engine not initialized"}
    import numpy as np
    import pandas as pd
    days = params.get("days", 90)
    # Try real Binance data first, fallback to mock
    symbol = params.get("symbol", "BTCUSDT")
    interval = params.get("interval", "1h")
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = await DataLoader.from_binance_api(symbol=symbol, interval=interval, start_time=start, end_time=end)
        source = "Binance API"
    except Exception:
        data = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
        source = "Mock (Binance unavailable)"
    if data.empty:
        return {"error": "No data available"}
    env = TradingEnvironment(data, initial_balance=1000.0)
    obs = env.reset()
    total_reward = 0.0
    steps = min(params.get("steps", 500), len(data))
    for _ in range(steps):
        action = np.random.randint(0, 6)  # Random actions for now
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return {
        "status": "completed", "steps": steps, "total_reward": round(total_reward, 4),
        "final_balance": round(info.get("balance", 0), 4),
        "trades": len(env.trades), "data_source": source,
    }


@app.post("/api/v1/rl/evolve")
async def rl_evolve(params: dict = {}):
    """Run genetic strategy evolution."""
    def fitness_fn(strat):
        # Simplified fitness based on parameter quality
        score = 0.0
        p = strat.params
        if p.get("ema_fast", 9) < p.get("ema_slow", 21):
            score += 0.3
        if 60 > p.get("rsi_oversold", 30) < p.get("rsi_overbought", 70) > 60:
            score += 0.2
        if 1.0 < p.get("atr_mult_sl", 1.5) < 3.0:
            score += 0.2
        if p.get("min_confidence", 0.4) > 0.35:
            score += 0.3
        return score

    evolver = StrategyEvolver(fitness_fn, population_size=20)
    evolver.initialize_population()
    population = evolver.evolve(generations=params.get("generations", 10))
    best = evolver.get_best()
    return {
        "status": "completed",
        "generations": evolver.generation,
        "best_fitness": round(best.fitness, 4) if best else 0,
        "best_params": best.params if best else {},
        "population_size": len(population),
    }


# ── API: Backtest ───────────────────────────────────────────────────

@app.post("/api/v1/backtest/run")
async def run_backtest(params: dict):
    if not backtest_engine or not strategy_engine:
        return {"error": "Not initialized"}
    symbol = params.get("symbol", "BTCUSDT")
    interval = params.get("interval", "1h")
    days = params.get("days", 30)
    leverage = params.get("leverage", 3)
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = await DataLoader.from_binance_api(symbol=symbol, interval=interval, start_time=start, end_time=end)
        using_mock = False
    except Exception:
        data = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
        using_mock = True
    if data.empty:
        return {"error": "No data"}
    result = await backtest_engine.run(strategy_engine, data, symbol=symbol, leverage=leverage)
    return {
        "symbol": symbol, "interval": interval, "days": days, "leverage": leverage,
        "using_mock_data": using_mock, "candles_analyzed": len(data),
        "result": {
            "total_trades": result.total_trades,
            "win_rate": round(result.win_rate, 4),
            "total_pnl": round(result.total_pnl, 4),
            "max_drawdown_pct": round(result.max_drawdown_pct, 4),
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "profit_factor": result.profit_factor,
            "expectancy": round(result.expectancy, 4),
        },
        "summary": result.summary(),
    }


# ── API: ML ─────────────────────────────────────────────────────────

@app.post("/api/v1/ml/train")
async def ml_train(params: dict = {}):
    if not ml_engine:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd
    days = params.get("days", 90)
    # Try real Binance data first, fallback to mock
    symbol = params.get("symbol", "BTCUSDT")
    interval = params.get("interval", "1h")
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        data = await DataLoader.from_binance_api(symbol=symbol, interval=interval, start_time=start, end_time=end)
        source = "Binance API"
    except Exception:
        data = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
        source = "Mock (Binance unavailable)"
    if data.empty:
        return {"error": "No data available"}
    t0 = time_module.time()
    result = ml_engine.train(data, force=params.get("force", False))
    duration = time_module.time() - t0
    if result.get("status") == "trained":
        update_ml_metrics(confidence=0.5, direction="long",
                          accuracy=result.get("test_accuracy", 0.5), age_hours=0.0)
    return {**result, "duration_seconds": round(duration, 2), "data_source": source, "candles": len(data)}


@app.post("/api/v1/ml/predict")
async def ml_predict(params: dict = {}):
    if not ml_engine:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd
    symbol = params.get("symbol", "BTCUSDT")
    days = params.get("days", 14)
    interval = params.get("interval", "1h")
    # Try real Binance data first, fallback to mock
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = await DataLoader.from_binance_api(symbol=symbol, interval=interval, start_time=start, end_time=end)
        source = "Binance API"
    except Exception:
        df = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
        source = "Mock (Binance unavailable)"
    if df.empty:
        return {"error": "No data available"}
    raw = params.get("raw", False)
    signal = ml_engine.predict(df) if raw else ml_engine.predict_signal(df)
    needs_train = ml_engine.needs_retraining(df)
    return {
        "symbol": symbol,
        "data_source": source,
        "candles": len(df),
        "signal": signal if raw else {
            "direction": signal.direction, "confidence": signal.confidence,
            "entry_price": signal.entry_price, "stop_loss": signal.stop_loss,
            "take_profits": signal.take_profits, "reason": signal.reason,
            "metadata": signal.metadata,
        },
        "needs_retraining": needs_train,
    }


@app.get("/api/v1/ml/status")
async def ml_status():
    if not ml_engine:
        return {"error": "Not initialized"}
    return {
        "trained": ml_engine._model is not None,
        "train_count": ml_engine._train_count,
        "last_train_time": ml_engine._last_train_time.isoformat() if ml_engine._last_train_time else None,
        "model_age_hours": round((datetime.now() - ml_engine._last_train_time).total_seconds() / 3600, 1) if ml_engine._last_train_time else None,
    }


# ── API: TimesFM Forecasting ────────────────────────────────────────

@app.post("/api/v2/forecast/timesfm")
async def timesfm_forecast(params: dict = {}):
    """Forecast future price with Google's TimesFM foundation model.

    Body params:
      - prices: optional list[float] of historical closes (overrides fetch)
      - symbol: market symbol when fetching data (default BTCUSDT)
      - interval: candle interval (default 1h)
      - days: lookback window in days (default 30)
      - horizon: number of steps to forecast (default 24)
    """
    if not timesfm_forecaster:
        return {"error": "Not initialized"}

    horizon = int(params.get("horizon", 24))
    prices = params.get("prices")

    if prices:
        result = timesfm_forecaster.forecast(prices, horizon=horizon)
        return {"data_source": "client-provided", **result.to_dict()}

    symbol = params.get("symbol", "BTCUSDT")
    interval = params.get("interval", "1h")
    days = int(params.get("days", 30))
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = await DataLoader.from_binance_api(symbol=symbol, interval=interval, start_time=start, end_time=end)
        source = "Binance API"
    except Exception:
        df = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
        source = "Mock (Binance unavailable)"
    if df.empty:
        return {"error": "No data available"}

    result = timesfm_forecaster.forecast_dataframe(df, horizon=horizon)
    return {"symbol": symbol, "data_source": source, "candles": len(df), **result.to_dict()}


@app.get("/api/v2/forecast/timesfm/status")
async def timesfm_status():
    if not timesfm_forecaster:
        return {"error": "Not initialized"}
    return {
        "checkpoint": timesfm_forecaster.checkpoint,
        "max_context": timesfm_forecaster.max_context,
        "max_horizon": timesfm_forecaster.max_horizon,
        "loaded": timesfm_forecaster._model is not None,
        "load_error": timesfm_forecaster._load_error,
    }


# ── API: Trade ──────────────────────────────────────────────────────

@app.post("/api/v1/trade")
async def execute_trade(trade: dict):
    if not paper_account:
        return {"error": "Not initialized"}
    request = OrderRequest(
        symbol=trade.get("symbol", "BTCUSDT"),
        side=trade.get("side", "buy"),
        order_type=trade.get("order_type", "market"),
        quantity=trade.get("quantity", 0.001),
        price=trade.get("price"),
        reduce_only=trade.get("reduce_only", False),
        stop_loss=trade.get("stop_loss"),
        take_profits=trade.get("take_profits"),
    )
    order = paper_account.place_order(request)
    portfolio = paper_account.get_portfolio()
    return {
        "order_id": order.id, "status": order.status,
        "avg_fill_price": order.avg_fill_price, "filled_quantity": order.filled_quantity,
        "fees": order.fees, "portfolio": portfolio,
    }


@app.post("/api/v1/positions/open")
async def open_position(plan: dict):
    if not position_manager or not paper_account:
        return {"error": "Not initialized"}
    trade_plan = TradePlan(
        symbol=plan["symbol"], direction=plan["direction"],
        entry_price=plan["entry_price"], quantity=plan.get("quantity", 0.001),
        leverage=plan.get("leverage", 3), confidence=plan.get("confidence", 0.6),
        stop_loss=plan["stop_loss"], take_profits=plan.get("take_profits", []),
        reason=plan.get("reason", ""), regime=plan.get("regime", "unknown"),
    )
    order = paper_account.place_order(OrderRequest(
        symbol=trade_plan.symbol, side="buy" if trade_plan.direction == "long" else "sell",
        order_type="market", quantity=trade_plan.quantity, price=trade_plan.entry_price,
        stop_loss=trade_plan.stop_loss, take_profits=trade_plan.take_profits,
    ))
    if order.status == "rejected":
        return {"error": f"Order rejected: {order.status}"}
    position = position_manager.open_position(trade_plan)
    return {
        "position_id": position.id, "order_id": order.id,
        "status": order.status, "entry_price": trade_plan.entry_price,
        "stop_loss": trade_plan.stop_loss, "take_profits": trade_plan.take_profits,
    }


@app.post("/api/v1/positions/{position_id}/close")
async def close_position(position_id: str):
    if not position_manager:
        return {"error": "Not initialized"}
    result = position_manager.close_position(position_id)
    if not result:
        return {"error": "Position not found"}
    return result


@app.get("/api/v1/positions/managed")
async def get_managed_positions():
    if not position_manager:
        return {"error": "Not initialized"}
    return {
        "open": [{
            "id": p.id, "symbol": p.symbol, "direction": p.direction,
            "entry_price": round(p.entry_price, 2), "current_price": round(p.current_price, 2),
            "quantity": round(p.quantity, 6), "unrealized_pnl": round(p.unrealized_pnl, 4),
            "stop_loss": round(p.stop_loss, 2), "status": p.status,
        } for p in position_manager.get_all_positions()],
        "closed": [{
            "id": p.id, "symbol": p.symbol, "pnl": round(p.realized_pnl, 4), "exit_reason": p.exit_reason,
        } for p in position_manager.get_closed_positions()],
    }


# ── API: Database ───────────────────────────────────────────────────

@app.get("/api/v1/db/stats")
async def db_stats(days: int = Query(30, ge=1, le=365)):
    if not database:
        return {"error": "Not initialized"}
    try:
        return await database.get_trade_stats(days=days)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/db/trades")
async def db_trades(limit: int = Query(50, ge=1, le=200)):
    if not database:
        return {"error": "Not initialized"}
    try:
        return {"trades": await database.get_recent_trades(limit=limit)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/db/daily-pnl")
async def db_daily_pnl(days: int = Query(30, ge=1, le=365)):
    if not database:
        return {"error": "Not initialized"}
    try:
        return {"daily_pnl": await database.get_daily_pnl(days=days)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/db/strategies")
async def db_strategies():
    if not database:
        return {"error": "Not initialized"}
    try:
        return {"strategies": await database.get_active_strategies()}
    except Exception as e:
        return {"error": str(e)}


# ── API: Market Structure v2 ────────────────────────────────────────

@app.get("/api/v2/market-structure")
async def get_market_structure(symbol: str = "BTCUSDT"):
    if not market_structure or not paper_account:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd

    # Try real Binance data first, fallback to mock
    end = datetime.now()
    start = end - timedelta(hours=200)
    try:
        df = await DataLoader.from_binance_api(
            symbol=symbol, interval="1h", start_time=start, end_time=end
        )
        data_source = "Binance API"
    except Exception:
        periods = 200
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="1h")
        base = 50000 + np.random.normal(0, 200, periods)
        df = pd.DataFrame({
            "open": base + np.random.normal(0, 100, periods),
            "high": base + 200 + abs(np.random.normal(0, 100, periods)),
            "low": base - 200 - abs(np.random.normal(0, 100, periods)),
            "close": base,
            "volume": np.random.exponential(100, periods),
        }, index=dates)
        data_source = "Mock (Binance unavailable)"

    result = market_structure.compute_all(df)
    signal = market_structure.get_market_structure_signal(df, df.iloc[-1])

    return {
        "symbol": symbol,
        "data_source": data_source,
        "signal": signal,
        "structure": {
            "swing_highs": int(result["swing_high"].sum()),
            "swing_lows": int(result["swing_low"].sum()),
            "bos_up": int(result["bos_up"].sum()),
            "bos_down": int(result["bos_down"].sum()),
            "choch": int(result["choch_up"].sum() + result["choch_down"].sum()),
            "liq_sweeps": int(result["liq_sweep_high"].sum() + result["liq_sweep_low"].sum()),
            "order_blocks": int((result["ob_bullish"] > 0).sum() + (result["ob_bearish"] > 0).sum()),
            "fvg_gaps": int((result["fvg_bullish_top"] > 0).sum() + (result["fvg_bearish_top"] > 0).sum()),
            "wyckoff_phase": result["wyckoff_phase"].iloc[-1],
        },
    }


@app.post("/api/v2/market-structure/fake-breakout")
async def check_fake_breakout(data: dict):
    if not fake_breakout:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd
    days = data.get("days", 1)
    symbol = data.get("symbol", "BTCUSDT")
    base = data.get("price", 50000)
    level = data.get("level", base)
    direction = data.get("direction", "up")

    # Try real Binance data first, fallback to mock
    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = await DataLoader.from_binance_api(
            symbol=symbol, interval="1h", start_time=start, end_time=end
        )
        data_source = "Binance API"
    except Exception:
        periods = max(days * 24, 20)
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="1h")
        rng = np.random.RandomState()
        df = pd.DataFrame({
            "open": base + rng.normal(0, 50, periods),
            "high": base + 100 + abs(rng.normal(0, 80, periods)),
            "low": base - 50 - abs(rng.normal(0, 50, periods)),
            "close": base + rng.normal(0, 60, periods),
            "volume": np.random.exponential(100, periods),
        }, index=dates)
        data_source = "Mock (Binance unavailable)"

    result = fake_breakout.detect(df, level, direction)
    result["data_source"] = data_source
    result["candles_analyzed"] = len(df)
    return result


# ── API: Enhanced Risk v2 ────────────────────────────────────────────

@app.post("/api/v2/risk/check")
async def enhanced_risk_check(trade: dict):
    if not enhanced_risk or not paper_account:
        return {"error": "Not initialized"}
    portfolio = paper_account.get_portfolio()
    positions = paper_account.get_positions()
    portfolio["positions"] = positions
    result = enhanced_risk.pre_trade_check(trade, portfolio)
    return result


@app.post("/api/v2/risk/kelly")
async def kelly_calculate(data: dict):
    if not enhanced_risk:
        return {"error": "Not initialized"}
    size = enhanced_risk.calculate_position_size(
        balance=data.get("balance", 1000.0),
        price=data.get("price", 50000),
        atr=data.get("atr", 0),
        confidence=data.get("confidence", 0.5),
        win_rate=data.get("win_rate", 0.5),
        avg_win=data.get("avg_win", 100),
        avg_loss=data.get("avg_loss", 100),
        regime=data.get("regime", "weak_trend"),
        symbol=data.get("symbol", "BTCUSDT"),
    )
    return size


@app.get("/api/v2/risk/report")
async def risk_report():
    if not enhanced_risk:
        return {"error": "Not initialized"}
    return enhanced_risk.get_risk_report()


@app.get("/api/v2/risk/kill-switch")
async def kill_switch_status():
    if not enhanced_risk:
        return {"error": "Not initialized"}
    return {
        "kill_switch": {
            "activated": enhanced_risk.kill_switch.status.activated,
            "triggered_by": enhanced_risk.kill_switch.status.triggered_by,
            "cooldown_until": enhanced_risk.kill_switch.status.cooldown_until,
        },
        "streaks": {
            "consecutive_wins": enhanced_risk.martingale._consecutive_wins,
            "consecutive_losses": enhanced_risk.martingale._consecutive_losses,
            "current_multiplier": enhanced_risk.martingale.get_multiplier(),
        },
        "time_decay": {
            "multiplier": enhanced_risk.time_decay.get_multiplier(),
        },
    }


@app.post("/api/v2/risk/record-trade")
async def record_trade_result(data: dict):
    if not enhanced_risk:
        return {"error": "Not initialized"}
    enhanced_risk.record_trade_result(
        pnl=data.get("pnl", 0),
        symbol=data.get("symbol", "BTCUSDT"),
        size=data.get("size", 0),
    )
    return {"status": "ok"}


# ── API: Execution v2 ────────────────────────────────────────────────

@app.post("/api/v2/execution/plan")
async def create_execution_plan(order: dict):
    plan = get_execution_plan(order)
    return {
        "symbol": plan.symbol,
        "side": plan.side,
        "total_quantity": plan.total_quantity,
        "algorithm": plan.algorithm,
        "num_slices": len(plan.slices),
        "slices": plan.slices[:20],
        "estimated_slippage_bps": plan.estimated_slippage_bps,
        "estimated_total_cost": plan.estimated_total_cost,
        "duration_seconds": plan.duration_seconds,
        "urgency": plan.urgency,
        "reasoning": plan.reasoning,
    }


@app.post("/api/v2/execution/slippage")
async def estimate_slippage(data: dict):
    estimator = SlippageEstimator()
    result = estimator.estimate(
        quantity=data.get("quantity", 0.1),
        price=data.get("price", 50000),
        volume_24h=data.get("volume_24h", 10_000_000),
        volatility_bps=data.get("volatility_bps", 50),
        spread_bps=data.get("spread_bps", 3),
    )
    return result


# ── API: Microstructure v2 ──────────────────────────────────────────

@app.post("/api/v2/microstructure/delta")
async def analyze_delta(data: dict):
    if not delta_tracker:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd
    symbol = data.get("symbol", "BTCUSDT")
    price = data.get("price", 50000)
    lookback = data.get("lookback", 100)

    # Try to load real price ticks from Binance klines
    try:
        end = datetime.now()
        start = end - timedelta(hours=24)
        df = await DataLoader.from_binance_api(
            symbol=symbol, interval="1m", start_time=start, end_time=end, limit=lookback
        )
        if not df.empty:
            for idx in range(len(df)):
                candle = df.iloc[idx]
                # Approximate delta as close-open difference normalized
                delta = (candle["close"] - candle["open"]) / candle["close"] * candle["volume"]
                delta_tracker.record_tick(float(candle["close"]), float(delta))
        else:
            raise ValueError("Empty DataFrame")
    except Exception:
        # Fallback to simulated delta ticks
        for _ in range(lookback):
            delta_tracker.record_tick(
                price + np.random.normal(0, 10),
                np.random.normal(0, 0.5),
            )

    div = delta_tracker.analyze_divergence(lookback=lookback)
    signal = delta_tracker.get_delta_signal()
    return {
        "cvd": delta_tracker.get_cvd(),
        "divergence": div,
        "signal": signal,
    }


@app.post("/api/v2/microstructure/spoofing")
async def check_spoofing(data: dict):
    if not spoofing_detector:
        return {"error": "Not initialized"}
    import numpy as np
    symbol = data.get("symbol", "BTCUSDT")
    base_price = data.get("price", 50000)

    # Try to fetch real order book from Binance
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
            )
            resp.raise_for_status()
            depth = resp.json()
            bids = [[float(p), float(q)] for p, q in depth.get("bids", [])[:10]]
            asks = [[float(p), float(q)] for p, q in depth.get("asks", [])[:10]]
            for _ in range(5):
                spoofing_detector.record_snapshot(bids, asks)
    except Exception:
        # Fallback: simulate order book snapshots
        for _ in range(20):
            bids = [[base_price - i*10, np.random.exponential(5)] for i in range(10)]
            asks = [[base_price + i*10, np.random.exponential(5)] for i in range(10)]
            spoofing_detector.record_snapshot(bids, asks)

    result = spoofing_detector.analyze()
    result["symbol"] = symbol
    return result


@app.post("/api/v2/microstructure/liquidation-cascade")
async def predict_cascade(data: dict):
    if not liq_cascade:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd
    symbol = data.get("symbol", "BTCUSDT")
    price = data.get("price", 50000)

    # Try to fetch real Binance klines for price acceleration detection
    fetched_price = price
    price_1h_ago = data.get("price_1h_ago", price * 0.99)
    price_4h_ago = data.get("price_4h_ago", price * 1.01)
    try:
        end = datetime.now()
        start = end - timedelta(hours=48)
        df = await DataLoader.from_binance_api(
            symbol=symbol, interval="1h", start_time=start, end_time=end
        )
        if not df.empty and len(df) >= 5:
            fetched_price = float(df["close"].iloc[-1])
            price_1h_ago = float(df["close"].iloc[-2]) if len(df) >= 2 else fetched_price
            price_4h_ago = float(df["close"].iloc[-5]) if len(df) >= 5 else fetched_price
    except Exception:
        pass  # Keep the defaults from data params

    # Simulate recent liquidations (no real public API for this)
    for _ in range(np.random.randint(5, 30)):
        liq_cascade.record_liquidation(
            symbol=symbol,
            side=np.random.choice(["buy", "sell"]),
            quantity=np.random.exponential(5),
            price=price + np.random.normal(0, 100),
            usd_value=np.random.exponential(500_000),
        )

    market_data = {
        "funding_rate": data.get("funding_rate", 0.0001),
        "open_interest": data.get("open_interest", 10_000_000_000),
        "open_interest_24h_ago": data.get("oi_24h_ago", 10_500_000_000),
        "price": price,
        "price_1h_ago": price_1h_ago,
        "price_4h_ago": price_4h_ago,
    }
    result = liq_cascade.predict(market_data)
    result["market_price"] = round(price, 2)
    result["symbol"] = symbol
    return result


@app.post("/api/v2/microstructure/orderbook")
async def analyze_orderbook(data: dict):
    import numpy as np
    symbol = data.get("symbol", "BTCUSDT")
    base_price = data.get("price", 50000)
    depth = data.get("depth", 10)

    # Try to fetch real order book from Binance
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={min(depth, 100)}"
            )
            resp.raise_for_status()
            depth_data = resp.json()
            bids = [[float(p), float(q)] for p, q in depth_data.get("bids", [])[:depth]]
            asks = [[float(p), float(q)] for p, q in depth_data.get("asks", [])[:depth]]
    except Exception:
        # Fallback: simulated order book
        bids = [[base_price - i*5, np.random.exponential(5)] for i in range(depth)]
        asks = [[base_price + i*5, np.random.exponential(5)] for i in range(depth)]

    result = OrderBookImbalanceAnalyzer.analyze(bids, asks, depth_levels=depth)
    result["symbol"] = symbol
    result["current_price"] = (bids[0][0] + asks[0][0]) / 2 if bids and asks else base_price
    return result


# ── API: Agent Routing v2 ───────────────────────────────────────────

@app.get("/api/v2/agents/routing")
async def get_agent_routing():
    if not agent_router:
        return {"error": "Not initialized"}
    return {
        "agent_mappings": agent_router.get_agent_model_summary(),
        "free_models": agent_router.get_free_model_list(),
    }


@app.get("/api/v2/inference/routing")
async def get_inference_routing():
    """
    Full inference routing overview — shows every agent's live routing config.

    Returns:
        - provider_health: Live health status of Groq, NVIDIA NIM, OpenRouter
        - agent_chains: Static provider chain per agent (from AGENT_PROVIDER_CHAINS)
        - task_overrides: Task type overrides per agent (from AGENT_TASK_TYPE_OVERRIDES)
        - adaptive_routing: Real observed P50 latencies and adapted chains
        - agent_mappings: Agent→model assignments
        - cost_usage: Today's token/cost usage and cache efficiency
    """
    if not nim_integration:
        return {"error": "Inference integration not initialized"}

    # Import configs for display
    from .agent_routing import (
        AGENT_PROVIDER_CHAINS,
        AGENT_TASK_TYPE_OVERRIDES,
    )

    provider_health = await nim_integration.get_provider_health()

    # Build per-agent chain display with adapted versions
    agent_chains = {}
    for agent_id, chain in sorted(AGENT_PROVIDER_CHAINS.items()):
        if agent_id == "default":
            continue
        adapted = nim_integration.agent_router.get_optimal_provider_chain(agent_id, chain)
        agent_chains[agent_id] = {
            "base_chain": chain,
            "adapted_chain": adapted,
            "task_override": AGENT_TASK_TYPE_OVERRIDES.get(agent_id),
            "adapted": adapted != chain,
        }

    # Default chain
    agent_chains["default"] = {
        "base_chain": AGENT_PROVIDER_CHAINS.get("default", []),
        "adapted_chain": None,
        "task_override": None,
        "adapted": False,
    }

    return {
        "provider_health": provider_health,
        "agent_chains": agent_chains,
        "adaptive_routing": nim_integration.agent_router.get_adaptive_routing_summary(),
        "agent_mappings": nim_integration.get_agent_model_summary(),
        "cost_usage": {
            "today": nim_integration.get_todays_usage(),
            "cache": nim_integration.get_cache_stats(),
            "over_budget": nim_integration.is_over_budget(),
        },
    }


@app.get("/api/v2/inference/adaptive-routing")
async def get_adaptive_routing(
    agent_id: str = Query(None, description="Filter to a single agent (e.g. 'scalping_agent')"),
):
    """
    Adaptive routing performance data — per-agent per-provider latency tracking.

    Shows the EMA-smoothed latency observations, sample counts, and whether
    the adaptive router has enough data to reorder each agent's provider chain.

    Returns the raw output of AgentModelRouter.get_adaptive_routing_summary():
        - {agent_id: {provider: {ema_latency_ms, samples, success_rate,
                                 _adaptive_ready, _adapted_chain, ...}}}

    Query params:
        - agent_id: (optional) Filter to a single agent
    """
    if not nim_integration:
        return {"error": "Inference integration not initialized"}

    return nim_integration.agent_router.get_adaptive_routing_summary(agent_id=agent_id)


@app.post("/api/v2/agents/resolve")
async def resolve_task(data: dict):
    if not task_router:
        return {"error": "Not initialized"}
    task_type = data.get("task_type", "analysis")
    latency_budget = data.get("latency_budget_ms")
    result = task_router.resolve(task_type, latency_budget)
    return result


@app.get("/api/v2/agents/recommend")
async def recommend_task(symbol: str = "BTCUSDT", volatility: float = 0.02,
                         timeframe: str = "1h", confidence: float = 0.5,
                         regime: str = "weak_trend"):
    if not task_router:
        return {"error": "Not initialized"}
    context = {
        "volatility": volatility,
        "timeframe": timeframe,
        "confidence": confidence,
        "regime": regime,
    }
    task_type = task_router.get_task_recommendation(context)
    resolved = task_router.resolve(task_type)
    return {
        "recommended_task": task_type,
        "reasoning": f"Based on vol={volatility}, tf={timeframe}, conf={confidence}, regime={regime}",
        "resolved": resolved,
    }


# ── API: Layer A Brain Runners ────────────────────────────────────────

@app.get("/api/v1/brains")
async def get_brains_status():
    """Status of all registered brain runners and their latest signals."""
    nats_connected = brain_publisher is not None and brain_publisher.nc is not None and brain_publisher.nc.is_connected
    return {
        "total_registered": len(BRAIN_REGISTRY),
        "runners_active": len(brain_runners),
        "nats_connected": nats_connected,
        "brains": {
            brain_id: {
                "class": brain_class.__name__,
                "runner_active": brain_id in brain_runners,
                "task_alive": (
                    brain_runner_tasks.get(brain_id) is not None
                    and not brain_runner_tasks[brain_id].done()
                ) if brain_id in brain_runner_tasks else False,
                "interval": BRAIN_INTERVALS.get(brain_id, 15.0),
            }
            for brain_id, brain_class in sorted(BRAIN_REGISTRY.items())
        },
    }


# ── Metrics ─────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ORCHESTRATOR_PORT", "8001"))
    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
