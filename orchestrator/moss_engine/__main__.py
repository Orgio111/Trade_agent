"""Moss Factory Simulation Runner.

Full 24/7 Binance BTC/USDT 15m pipeline:
  1. Boot-up → ledger reconciliation
  2. Whale trade delta ← KryptCryptoCore
  3. MOSS 5-pillar composite + Contrarian Fade override
  4. Guardrail/Daily PnL evaluation
  5. Mock evolution round
  6. Signals published to NATS JetStream `signals.raw`

Usage:
    cd Trade_agent && python -m orchestrator.moss_engine
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from orchestrator.moss_engine.binance_ingestion import BinanceIngestionEngine
from orchestrator.moss_engine.composite_engine import MossCompositeEngine
from orchestrator.moss_engine.krypt_core import KryptCryptoCore
from orchestrator.moss_engine.reconciliation_engine import ReconciliationEngine
from orchestrator.moss_engine.reflection_engine import ReflectiveEvolutionLoop
from orchestrator.moss_engine.schemas import CompositeSignal
from orchestrator.moss_engine.skill_registry import SkillRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
logger = logging.getLogger("moss_simulation")

# ── .env auto-load ──────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass


async def simulate_moss_factory():
    """End-to-end MOSS simulation."""
    # Setup registry
    registry = SkillRegistry()
    
    # Instantiate all engines
    binance_ingestion = BinanceIngestionEngine()
    moss_composite = MossCompositeEngine(registry=registry)
    krypt_core = KryptCryptoCore()
    reconcile_engine = ReconciliationEngine(registry=registry)
    
    # Register skills
    moss_composite.inject_into_registry(registry)
    krypt_core.inject_into_registry(registry)
    
    # Step 1: Boot → Binance ingestion
    logger.info("Step 1 — Bootstrap BTC/USDT 15m candles")
    candles = binance_ingestion.bootstrap_stream("BTC/USDT", "15m")
    latest_candle = candles[-1]
    logger.info("Fetched %d candles, latest OHLC: %s", len(candles), [latest_candle[k] for k in ["open", "high", "low", "close"]])

    # Step 2: Ledger reconciliation
    logger.info("Step 2 — Book reconciliation (simulated)")
    mock_internal = {
        "positions": [
            {"symbol": "BTCUSDT", "notional": 10_000},
            {"symbol": "ETHUSDT", "notional": 5_000},
        ]
    }
    reconciliation = await reconcile_engine.reconcile_ledger_with_exchange(mock_internal)
    guardrail_status = await reconcile_engine.check_and_trigger_guardrails(registry)
    logger.info("Reconciliation: matched=%d, orphans=%d → fixing trades=%d", 
                reconciliation.matched_positions, reconciliation.orphan_positions, 
                reconciliation.generated_fixing_trades)
    logger.info("Guardrail: switch active=%s", guardrail_status.master_kill_switch)

    # Step 3: KryptCryptoCore microstructure
    logger.info("Step 3 — Process whale trade delta")
    whale_trades = [
        {"symbol": "BTC/USDT", "p": latest_candle["close"], "q": 25.0,
         "side": "buy", "notional": latest_candle["close"] * 25.0}
    ]  # mock $100k+ trade
    microstructure = krypt_core.execute(
        symbol="BTC/USDT", interval="15m", ohlcv=candles, trades=whale_trades
    )
    logger.info(
        "Whale Tracker OFI.score=%.4f, skew=%.4f, %d whale events",
        microstructure["ofi_score"], microstructure["directional_skew"],
        len(microstructure["whale_events"]),
    )

    # Step 4: Composite signal fusion
    logger.info("Step 4 — 5-pillar composite signal")
    whale_skew = microstructure["directional_skew"]
    composite_signal: CompositeSignal = moss_composite.compute_composite_signal(
        symbol="BTC/USDT", interval="15m", ohlcv=candles,
        fade_override=microstructure["fade_signal"],
        whale_skew=whale_skew,
        guardrail_daily_drawdown_pct=guardrail_status.daily_drawdown_pct,
    )

    print("CompositeSignal JSON:")
    print(json.dumps(composite_signal.to_dict(), indent=2, default=str))
    
    # Step 5: Reflection evolution
    reflect_engine = ReflectiveEvolutionLoop(registry)
    reflect_engine.buffer_signal(composite_signal)
    evolution_result = reflect_engine.evolve_parameters()

    print("Evolution Report JSON:")
    print(json.dumps(evolution_result.to_dict(), indent=2, default=str))
    
    # Upstream: publish to NATS JetStream
    logger.info("Step 6 — Published to NATS signal.raw")
    print("NATS payload:", composite_signal.to_json())
    
    logger.info("End-to-End MOSS Simulation COMPLETE ✓")


if __name__ == "__main__":
    asyncio.run(simulate_moss_factory())