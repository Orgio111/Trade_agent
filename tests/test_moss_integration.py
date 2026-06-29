"""Moss Signal Factory integration tests.

Tests:
  1. SkillRegistry register/resolve/hotswap
  2. MossCompositeEngine 5-pillar ✓ guardrail
  3. KryptCryptoCore microstructure emit
  4. Reconciliation report emit
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add Trade_agent root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
os.environ["PYTHONPATH"] = str(ROOT_DIR) + ";" + os.environ.get("PYTHONPATH", "")

import pytest
import pytest_asyncio
from orchestrator.moss_engine.schemas import (
    CompositeSignal, SkillMeta, SkillState, WhaleEvent, ReconciliationReport, MomentumScanResult
)
from orchestrator.moss_engine.skill_registry import SkillRegistry
from orchestrator.moss_engine.composite_engine import MossCompositeEngine
from orchestrator.moss_engine.krypt_core import KryptCryptoCore
from orchestrator.moss_engine.reconciliation_engine import ReconciliationEngine


@pytest.fixture
def registry():
    """Test registry with 2 active brains."""
    reg = SkillRegistry()
    composite = MossCompositeEngine(registry=reg)
    krypt = KryptCryptoCore()
    composite.inject_into_registry(reg)
    krypt.inject_into_registry(reg)
    return reg


@pytest_asyncio.fixture
async def reconcile_engine():
    """Reconciliation engine with mock simulation."""
    engine = ReconciliationEngine(registry=SkillRegistry())
    
    async def mock_fetch_balance():
        return {"free": "1000.0", "info": {"positions": [{"symbol": "BTCUSDT", "positionAmt": "0.01"}]}}

    # Patch real exchange
    engine.exchange.fetch_balance = mock_fetch_balance
    return engine


def test_skill_registry_register_resolve(registry):
    """Test SkillRegistry register/resolve."""
    assert registry.registered_count == 2
    
    moss = registry.resolve("moss_composite")
    assert moss is not None
    
    meta: SkillMeta = registry.get_meta("moss_composite")
    assert meta.name == "moss_composite"
    assert meta.version == "1.0.28"


def test_skill_registry_hotswap(registry):
    """Test SkillRegistry hot_swap preserves state."""
    registry.update_state("moss_composite", total_invocations=7, daily_pnl=150.0)
    
    class MockBrain:
        def execute(self, *a):
            return {"mock": True}

    swapped = registry.hot_swap("moss_composite", MockBrain())
    assert swapped
    
    state = registry.get_state("moss_composite")
    assert state.total_invocations == 7


def test_composite_engine_5pillars():
    """MossCompositeEngine emits composite with 5 pillars."""
    engine = MossCompositeEngine()
    ohlcv = [
        {"open": 50000, "high": 51000, "low": 49800, "close": 50500, "volume": 1200},
        {"open": 50500, "high": 50800, "low": 49900, "close": 50100, "volume": 1000},
    ]
    signal: CompositeSignal = engine.compute_composite_signal(
        "BTC/USDT", "1h", ohlcv, guardrail_daily_drawdown_pct=4.0
    )
    assert len(signal.pillars) == 5


def test_krypt_core_microstructure():
    """KryptTrader microstructure emits whale events."""
    core = KryptCryptoCore()
    trades = [{"symbol": "BTC/USDT", "p": 50000, "q": 2.2, "side": "buy"}] * 150
    
    runs = []
    for i in range(20):
        candle = {
            "open": 50000 + i*100,
            "high": 50100 + i*100,
            "low": 49900 + i*100,
            "close": 50050 + i*100,
            "volume": 800 + i*10
        }
        runs.append(candle)

    out = core.execute("BTC/USDT", "15m", runs, trades=trades)
    whales = [WhaleEvent(**w) for w in out["whale_events"]]
    assert len(whales) > 0
    assert whales[0].notional_usd > 100_000


def test_reconciliation_orphan_generate(reconcile_engine):
    """Book reconciliation generates fixing trades."""
    internal_cache = {"positions": [{"symbol": "BTCUSDT", "notional": 10_000}]}
    report: ReconciliationReport = asyncio.run(
        reconcile_engine.reconcile_ledger_with_exchange(internal_cache)
    )
    assert report.generated_fixing_trades >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])