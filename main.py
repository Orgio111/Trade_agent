"""Main orchestrator - FastAPI + WebSocket for real-time trading."""

import asyncio
import json
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml

from models import Candle, Signal, AccountState, ExecutionResult
from state import IncrementalState
from router import OllamaRouter
from risk import RiskEngine
from memory import LocalMemory
from execution import BinanceClient, PaperClient, OrderRequest, OrderSide, OrderType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global components
state: Optional[IncrementalState] = None
router: Optional[OllamaRouter] = None
risk_engine: Optional[RiskEngine] = None
memory: Optional[LocalMemory] = None
execution_client = None
account_state: Optional[AccountState] = None
paper_mode = True


# System prompts for different modes
SCALP_SYSTEM_PROMPT = """You are a high-frequency scalping trader.
Analyze the 1-minute candle data and current market state.
Respond with ONLY valid JSON:
{
  "action": "BUY|SELL|HOLD",
  "confidence": 0.0-1.0,
  "size_pct": 0.0-1.0,
  "entry_price": float or null,
  "stop_loss": float or null,
  "take_profit": float or null,
  "reasoning": "brief explanation",
  "regime": "trend_up|trend_down|range|volatile|transition"
}"""

REASONING_SYSTEM_PROMPT = """You are a quantitative trading analyst.
Analyze the market data and provide a trading decision.
Consider: regime, key levels, momentum, volume, risk/reward.
Respond with ONLY valid JSON (same format as above)."""

DEEP_SYSTEM_PROMPT = """You are a macro trading strategist.
Perform deep analysis of market structure, regime, and macro factors.
Provide comprehensive trading thesis with specific levels.
Respond with ONLY valid JSON (same format)."""


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def initialize_components():
    """Initialize all system components."""
    global state, router, risk_engine, memory, execution_client, account_state, paper_mode
    
    config = load_config()
    paper_mode = config.get("system", {}).get("paper_mode", True)
    
    logger.info("Initializing Local Trading AI System...")
    
    # Initialize components
    state = IncrementalState("config.yaml")
    router = OllamaRouter("config.yaml")
    await router.initialize()
    
    risk_engine = RiskEngine("config.yaml")
    memory = LocalMemory("config.yaml")
    
    # Execution client
    if paper_mode:
        from execution import PaperClient
        execution_client = PaperClient(initial_balance=10000.0)
        logger.info("Running in PAPER TRADING mode")
    else:
        # Load API keys from environment
        import os
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        testnet = config.get("system", {}).get("testnet", True)
        
        if not api_key or not api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET required for live trading")
        
        execution_client = BinanceClient(api_key, api_secret, testnet=testnet)
        await execution_client.__aenter__()
        logger.info(f"Running in LIVE mode (testnet={testnet})")
    
    # Initial account state
    account_state = AccountState(
        equity=10000.0,
        balance=10000.0,
        unrealized_pnl=0.0,
        daily_pnl=0.0,
        daily_pnl_pct=0.0,
        open_positions=0,
        loss_streak=0,
        max_drawdown_pct=0.0
    )
    
    logger.info("All components initialized successfully")


async def shutdown_components():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
    if execution_client and hasattr(execution_client, '__aexit__'):
        await execution_client.__aexit__(None, None, None)
    if router:
        await router.shutdown()
    logger.info("Shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_components()
    yield
    await shutdown_components()


app = FastAPI(title="Local Trading AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "state": state is not None,
            "router": router is not None,
            "risk": risk_engine is not None,
            "memory": memory is not None,
            "execution": execution_client is not None,
        },
        "state_summary": state.summary() if state else {},
        "router_status": await router.get_status() if router else {},
    }


@app.websocket("/candle")
async def candle_websocket(ws: WebSocket):
    """WebSocket endpoint for receiving 1-minute candles."""
    await ws.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await ws.receive_text()
            candle_data = json.loads(data)
            
            # Parse candle
            candle = Candle.from_dict(candle_data)
            
            # Process through pipeline
            result = await process_candle(candle)
            
            # Send response
            await ws.send_text(json.dumps(result))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.close()


async def process_candle(candle: Candle) -> Dict[str, Any]:
    """Process single candle through full pipeline."""
    start_time = time.time()
    
    # 1. Update incremental state
    features = state.on_new_candle(candle)
    
    # 2. Get context for model
    context = state.get_context_for_model(mode="normal")
    context["urgency"] = "normal"
    
    # 3. Build prompt
    user_prompt = build_user_prompt(candle, features, context)
    
    # 4. Route to appropriate model
    model = router.route(context)
    
    # 5. Generate signal
    signal = await router.generate_signal(context, REASONING_SYSTEM_PROMPT, user_prompt)
    signal.model_used = model
    
    # 6. Risk validation
    risk_decision = risk_engine.validate(signal, account_state)
    
    # 7. Execute if allowed
    execution_result = None
    if risk_decision.allow and risk_decision.adjusted_signal:
        final_signal = risk_decision.adjusted_signal
        
        # Create order request
        order = OrderRequest(
            symbol=candle.symbol,
            side=OrderSide.BUY if final_signal.action == SignalAction.BUY else OrderSide.SELL,
            type=OrderType.LIMIT,
            quantity=final_signal.size_pct * account_state.equity / candle.close,
            price=final_signal.entry_price or candle.close,
            post_only=True,
            reduce_only=False
        )
        
        execution_result = await execution_client.place_order(order)
        
        # Update account state (simplified)
        if execution_result.status == "FILLED":
            account_state.open_positions += 1
    
    # 8. Prepare response
    elapsed = time.time() - start_time
    
    response = {
        "timestamp": int(time.time() * 1000),
        "candle": candle.to_dict(),
        "features": features.to_list() if features else [],
        "signal": signal.model_dump() if signal else None,
        "risk_decision": {
            "allow": risk_decision.allow,
            "reason": risk_decision.reason,
            "max_size": risk_decision.max_size,
        },
        "execution": execution_result.__dict__ if execution_result else None,
        "state": state.summary(),
        "account": account_state.model_dump() if account_state else None,
        "latency_ms": round(elapsed * 1000, 2),
    }
    
    # Log signal
    logger.info(f"Candle processed: {candle.symbol} {candle.close:.2f} | "
                f"Signal: {signal.action if signal else 'N/A'} "
                f"({signal.confidence:.2f} conf, {model}) | "
                f"Risk: {'ALLOW' if risk_decision.allow else 'BLOCK: ' + risk_decision.reason} | "
                f"Latency: {elapsed*1000:.1f}ms")
    
    return response


def build_user_prompt(candle: Candle, features: Any, context: Dict) -> str:
    """Build user prompt for model."""
    return f"""Current Candle: {candle.to_dict()}
Features: {features.to_list() if features else []}
Regime: {context.get('regime', 'unknown')} (confidence: {context.get('regime_confidence', 0):.2f})
Key Levels: Support={context.get('key_levels', {}).get('support', [])} Resistance={context.get('key_levels', {}).get('resistance', [])}
Session: VWAP={context.get('session', {}).get('vwap', 0):.2f} Bias={context.get('session', {}).get('bias', 'neutral')}
Strategy Stats: {context.get('strategy_stats', {})}

Provide trading decision as JSON."""


@app.get("/state")
async def get_state():
    """Get current system state."""
    return {
        "state": state.summary() if state else {},
        "account": account_state.model_dump() if account_state else {},
        "router": await router.get_status() if router else {},
        "risk": risk_engine.get_status() if risk_engine else {},
        "memory": memory.get_memory_stats() if memory else {},
    }


@app.get("/trades")
async def get_trades(limit: int = 20):
    """Get recent trades from memory."""
    if memory:
        return memory.get_recent_trades(limit)
    return []


@app.post("/signal")
async def manual_signal(signal: Signal):
    """Manually submit a signal for testing."""
    risk_decision = risk_engine.validate(signal, account_state)
    return {
        "signal": signal.model_dump(),
        "risk_decision": {
            "allow": risk_decision.allow,
            "reason": risk_decision.reason,
            "adjusted": risk_decision.adjusted_signal.model_dump() if risk_decision.adjusted_signal else None,
        }
    }


# Signal handlers
def handle_shutdown(signum, frame):
    logger.info("Shutdown signal received")
    asyncio.create_task(shutdown_components())
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )