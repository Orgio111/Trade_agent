"""
QUANTEX AI Orchestrator v1.0 — Full system with swarm intelligence,
RL training, vector memory, risk engine, and data pipeline.
"""
import os
import json
import asyncio
import time as time_module
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from contextlib import asynccontextmanager

from .nim_client import NIMOrchestrator
from .nim_enhanced import EnhancedNIMOrchestrator, NeMoDistiller
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
)
from .swarm.debate_system import AgentSwarm, ScalpingAgent, SwingAgent, RegimeAgent, AnomalyAgent
from .memory.vector_memory import TradingMemorySystem, AgentCredibilityTracker
from .risk.risk_engine import RiskEngine, PanicMode, AntiOvertradeSystem, RiskParams
from .risk.risk_engine_v2 import EnhancedRiskEngine, EnhancedRiskParams, KillSwitch, KellySizer
from .rl.trading_env import TradingEnvironment
from .rl.strategy_evolver import StrategyEvolver, OptunaOptimizer
from .data.pipeline import MarketDataOrchestrator
from .market_structure import MarketStructureEngine, FakeBreakoutDetector
from .microstructure import SpoofingDetector, HiddenLiquidityDetector, DeltaCVDTracker, LiquidationCascadePredictor, OrderBookImbalanceAnalyzer
from .execution import ExecutionOrchestrator, TWAPExecutor, VWAPExecutor, IcebergExecutor, SmartOrderRouter, SlippageEstimator, get_execution_plan, ExecutionPlan
from .agent_routing import AgentModelRouter, TaskRouter, AGENT_MODEL_MAP


# ── Global State ──────────────────────────────────────────

nim: NIMOrchestrator | None = None
nim_enhanced: EnhancedNIMOrchestrator | None = None
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
data_orchestrator: MarketDataOrchestrator | None = None
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

active_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nim, nim_enhanced, strategy_engine, market_analyst, deepseek_analyst
    global risk_guardian, sentiment_agent, scalping_agent, swing_agent
    global regime_agent, anomaly_agent
    global paper_account, position_manager, backtest_engine, feature_engine
    global ml_engine, database
    global swarm, memory, credibility_tracker, risk_engine, panic_mode
    global anti_overtrade, data_orchestrator, nemo_distiller
    global enhanced_risk, market_structure, fake_breakout
    global spoofing_detector, hidden_liquidity, delta_tracker, liq_cascade, ob_imbalance
    global execution_orch, agent_router, task_router

    # Phase 1-3 components
    nim = NIMOrchestrator()
    nim_enhanced = EnhancedNIMOrchestrator()
    strategy_engine = StrategyEngine()
    market_analyst = MarketAnalystAgent(nim)
    deepseek_analyst = DeepSeekAnalysisAgent(nim)
    risk_guardian = RiskGuardianAgent(nim)
    sentiment_agent = SentimentAgent()

    # Individual agents
    scalping_agent = ScalpingAgent(nim)
    swing_agent = SwingAgent(nim)
    regime_agent = RegimeAgent(nim)
    anomaly_agent = AnomalyAgent(nim)

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
    memory = TradingMemorySystem(use_stub=True)
    credibility_tracker = AgentCredibilityTracker(memory)
    swarm = AgentSwarm(nim)
    risk_engine = RiskEngine()
    panic_mode = PanicMode()
    anti_overtrade = AntiOvertradeSystem()
    data_orchestrator = MarketDataOrchestrator()
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

    print("  Institutional v2: EnhancedRisk, MarketStructure, Execution, Microstructure, AgentRouting")

    # Start data feeds
    asyncio.create_task(data_orchestrator.start_all_feeds())

    # Metrics loop
    update_portfolio_metrics(
        balance=paper_account.balance, equity=paper_account.balance,
        drawdown=0.0, open_positions=0, consec_losses=0, total_pnl=0.0,
    )

    async def _metrics_loop():
        while True:
            await asyncio.sleep(5)
            if paper_account:
                p = paper_account.get_portfolio()
                update_portfolio_metrics(
                    balance=p["balance"], equity=p["equity"],
                    drawdown=p["drawdown"], open_positions=p["open_positions"],
                    consec_losses=p["consecutive_losses"], total_pnl=p["total_pnl"],
                )

    asyncio.create_task(_metrics_loop())

    print("✅ QUANTEX AI Orchestrator v1.0 initialized")
    print(f"   Paper balance: ${paper_account.balance:.2f}")
    print(f"   ML model: {'loaded' if ml_engine._model else 'not trained'}")
    print(f"   Memory: {'Qdrant stub' if memory else 'unavailable'}")
    print(f"   Data feeds: running {len(data_orchestrator.feeds)} feeds")
    yield

    # Cleanup
    if database:
        await database.disconnect()
    await data_orchestrator.stop_all_feeds()


app = FastAPI(
    title="QUANTEX AI Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
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
        "data_feeds": len(data_orchestrator.feeds) if data_orchestrator else 0,
        "ml": {"trained": ml_engine._model is not None if ml_engine else False},
        "database": db_status,
        "balance": paper_account.balance if paper_account else 0,
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
    import numpy as np
    import pandas as pd
    rng = np.random.RandomState()
    periods = 100
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="1min")
    base = 50000 + rng.normal(0, 200)
    df = pd.DataFrame({
        "open": rng.normal(base, 100, periods),
        "high": rng.normal(base + 200, 100, periods),
        "low": rng.normal(base - 200, 100, periods),
        "close": rng.normal(base, 100, periods),
        "volume": rng.normal(100, 20, periods),
    }, index=dates)
    df["high"] = df[["open", "close"]].max(axis=1) + abs(rng.normal(0, 50, periods))
    df["low"] = df[["open", "close"]].min(axis=1) - abs(rng.normal(0, 50, periods))

    if source == "ml" and ml_engine:
        signal = ml_engine.predict_signal(df)
    elif source == "swarm" and swarm:
        # Use swarm with full debate
        context = {
            "symbol": symbol, "price": float(df["close"].iloc[-1]),
            "df": df, "regime": "unknown",
            "portfolio": paper_account.get_portfolio() if paper_account else {},
        }
        decision = await swarm.run_swarm_debate(context)
        return {
            "symbol": symbol, "source": "swarm",
            "signal": decision.direction, "confidence": decision.confidence,
            "entry_price": df["close"].iloc[-1], "reason": decision.reasoning[:200],
            "debate_log": decision.debate_log[:10],
        }
    elif strategy_engine:
        signal = strategy_engine.generate_signal(df)
    else:
        return {"error": "No signal source available"}

    return {
        "symbol": symbol, "source": source,
        "signal": signal.direction, "confidence": signal.confidence,
        "entry_price": signal.entry_price, "stop_loss": signal.stop_loss,
        "take_profits": signal.take_profits, "reason": signal.reason,
        "metadata": signal.metadata,
    }


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
    market_conditions = data_orchestrator.get_latest_sentiment() if data_orchestrator else {}
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
    data = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
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
        "trades": len(env.trades),
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
    data = DataLoader.generate_mock_data(periods=days * 24, start_price=50000.0)
    t0 = time_module.time()
    result = ml_engine.train(data, force=params.get("force", False))
    duration = time_module.time() - t0
    if result.get("status") == "trained":
        update_ml_metrics(confidence=0.5, direction="long",
                          accuracy=result.get("test_accuracy", 0.5), age_hours=0.0)
    return {**result, "duration_seconds": round(duration, 2)}


@app.post("/api/v1/ml/predict")
async def ml_predict(params: dict = {}):
    if not ml_engine:
        return {"error": "Not initialized"}
    import numpy as np
    import pandas as pd
    symbol = params.get("symbol", "BTCUSDT")
    df = DataLoader.generate_mock_data(periods=200, start_price=50000.0)
    raw = params.get("raw", False)
    signal = ml_engine.predict(df) if raw else ml_engine.predict_signal(df)
    needs_train = ml_engine.needs_retraining(df)
    return {
        "symbol": symbol,
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
    rng = np.random.RandomState()
    periods = 200
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="1h")
    base = 50000 + rng.normal(0, 200, periods)
    df = pd.DataFrame({
        "open": base + rng.normal(0, 100, periods),
        "high": base + 200 + abs(rng.normal(0, 100, periods)),
        "low": base - 200 - abs(rng.normal(0, 100, periods)),
        "close": base,
        "volume": np.random.exponential(100, periods),
    }, index=dates)

    result = market_structure.compute_all(df)
    signal = market_structure.get_market_structure_signal(df, df.iloc[-1])

    return {
        "symbol": symbol,
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
    periods = 20
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="1h")
    rng = np.random.RandomState()
    base = data.get("price", 50000)
    df = pd.DataFrame({
        "open": base + rng.normal(0, 50, periods),
        "high": base + 100 + abs(rng.normal(0, 80, periods)),
        "low": base - 50 - abs(rng.normal(0, 50, periods)),
        "close": base + rng.normal(0, 60, periods),
        "volume": np.random.exponential(100, periods),
    }, index=dates)

    result = fake_breakout.detect(df, data.get("level", base), data.get("direction", "up"))
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
    # Simulate delta ticks
    import numpy as np
    price = data.get("price", 50000)
    for _ in range(data.get("ticks", 100)):
        delta_tracker.record_tick(
            price + np.random.normal(0, 10),
            np.random.normal(0, 0.5),
        )
    div = delta_tracker.analyze_divergence(lookback=data.get("lookback", 100))
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
    # Simulate order book snapshots
    base_price = data.get("price", 50000)
    for _ in range(20):
        bids = [[base_price - i*10, np.random.exponential(5)] for i in range(10)]
        asks = [[base_price + i*10, np.random.exponential(5)] for i in range(10)]
        spoofing_detector.record_snapshot(bids, asks)

    result = spoofing_detector.analyze()
    return result


@app.post("/api/v2/microstructure/liquidation-cascade")
async def predict_cascade(data: dict):
    if not liq_cascade:
        return {"error": "Not initialized"}
    import numpy as np
    # Simulate recent liquidations
    for _ in range(np.random.randint(5, 30)):
        liq_cascade.record_liquidation(
            symbol=data.get("symbol", "BTCUSDT"),
            side=np.random.choice(["buy", "sell"]),
            quantity=np.random.exponential(5),
            price=data.get("price", 50000) + np.random.normal(0, 100),
            usd_value=np.random.exponential(500_000),
        )

    market_data = {
        "funding_rate": data.get("funding_rate", 0.0001),
        "open_interest": data.get("open_interest", 10_000_000_000),
        "open_interest_24h_ago": data.get("oi_24h_ago", 10_500_000_000),
        "price": data.get("price", 50000),
        "price_1h_ago": data.get("price_1h_ago", 50500),
        "price_4h_ago": data.get("price_4h_ago", 51000),
    }
    result = liq_cascade.predict(market_data)
    return result


@app.post("/api/v2/microstructure/orderbook")
async def analyze_orderbook(data: dict):
    import numpy as np
    base_price = data.get("price", 50000)
    depth = data.get("depth", 10)
    bids = [[base_price - i*5, np.random.exponential(5)] for i in range(depth)]
    asks = [[base_price + i*5, np.random.exponential(5)] for i in range(depth)]

    result = OrderBookImbalanceAnalyzer.analyze(bids, asks, depth_levels=depth)
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


# ── Metrics ─────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    return await metrics_endpoint()


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ORCHESTRATOR_PORT", "8001"))
    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
