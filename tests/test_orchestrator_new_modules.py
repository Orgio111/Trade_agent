"""Tests for the 5 new orchestrator modules.

Covers: autogen_executor, autogen_trader, enhanced_risk, backtest_engine,
model_loader.
"""

import importlib
import pytest
from unittest.mock import patch, MagicMock

from models import AccountState, Signal


# ── autogen_executor ──────────────────────────────────────────────────


class TestLanggraphSignalConversion:
    """_langgraph_signal_to_model_signal converts LG dict → models.Signal."""

    def test_buy_signal(self):
        from orchestrator.autogen_executor import _langgraph_signal_to_model_signal

        lg = {"action": "BUY", "confidence": 0.82, "entry_reason": ["RSI", "EMA"], "risk_notes": ""}
        sig = _langgraph_signal_to_model_signal(lg, "BTCUSDT")
        assert sig.action == "BUY"
        assert sig.confidence == 0.82
        assert "RSI" in sig.reasoning

    def test_hold_signal(self):
        from orchestrator.autogen_executor import _langgraph_signal_to_model_signal

        lg = {"action": "HOLD", "confidence": 0.1, "entry_reason": [], "risk_notes": ""}
        sig = _langgraph_signal_to_model_signal(lg, "ETHUSDT")
        assert sig.action == "HOLD"

    def test_missing_fields_defaults(self):
        from orchestrator.autogen_executor import _langgraph_signal_to_model_signal

        lg = {"action": "SELL", "confidence": 0.5}
        sig = _langgraph_signal_to_model_signal(lg, "BTCUSDT")
        assert sig.action == "SELL"
        assert sig.confidence == 0.5


class TestAutoGenSignalExecutorConstruction:
    """AutoGenSignalExecutor wires risk + loader correctly."""

    def test_paper_mode(self):
        from orchestrator.autogen_executor import AutoGenSignalExecutor
        from orchestrator.enhanced_risk import EnhancedRiskEngine

        ex = AutoGenSignalExecutor(paper_mode=True, model_warmup=False)
        assert ex.paper_mode is True
        assert isinstance(ex.risk_engine, EnhancedRiskEngine)

    def test_model_loader_has_pinned(self):
        from orchestrator.autogen_executor import AutoGenSignalExecutor

        ex = AutoGenSignalExecutor(paper_mode=True, model_warmup=False)
        assert hasattr(ex.model_loader, "gpu_pinned")
        assert ex.model_loader.keep_alive == "5m"


# ── enhanced_risk ──────────────────────────────────────────────────────


@pytest.fixture
def risk_engine():
    from orchestrator.enhanced_risk import EnhancedRiskEngine
    return EnhancedRiskEngine()


@pytest.fixture
def clean_account():
    return AccountState(
        equity=10000, balance=10000, open_positions=0,
        daily_pnl_pct=0.0, loss_streak=0, max_drawdown_pct=0.0,
    )


class TestEnhancedRiskEngine:
    def test_allow_good_signal(self, risk_engine, clean_account):
        sig = Signal(action="BUY", confidence=0.80, size_pct=0.03)
        dec = risk_engine.validate(sig, clean_account)
        assert dec.allow is True

    def test_reject_low_confidence(self, risk_engine, clean_account):
        sig = Signal(action="BUY", confidence=0.3, size_pct=0.03)
        dec = risk_engine.validate(sig, clean_account)
        assert dec.allow is False
        assert "confidence" in dec.reason.lower()

    def test_reject_daily_drawdown(self, risk_engine):
        state = AccountState(
            equity=10000, balance=10000, open_positions=0,
            daily_pnl_pct=-0.04, loss_streak=0, max_drawdown_pct=0.04,
        )
        sig = Signal(action="BUY", confidence=0.80, size_pct=0.03)
        dec = risk_engine.validate(sig, state)
        assert dec.allow is False
        assert "drawdown" in dec.reason.lower()

    def test_circuit_breaker_on_kill_streak(self, risk_engine):
        state = AccountState(
            equity=10000, balance=10000, open_positions=0,
            daily_pnl_pct=0.0, loss_streak=5, max_drawdown_pct=0.0,
        )
        sig = Signal(action="BUY", confidence=0.80, size_pct=0.03)
        dec = risk_engine.validate(sig, state)
        assert dec.allow is False
        assert "circuit" in dec.reason.lower() or "kill" in dec.reason.lower()

    def test_atr_volatility_block(self, risk_engine, clean_account):
        sig = Signal(action="BUY", confidence=0.80, size_pct=0.03)
        ind = {"atr_pct": 0.06, "atr_mean_pct": 0.02}
        dec = risk_engine.validate(sig, clean_account, indicators=ind)
        assert dec.allow is False
        assert "ATR" in dec.reason

    def test_atr_normal_passes(self, risk_engine, clean_account):
        sig = Signal(action="BUY", confidence=0.80, size_pct=0.03)
        ind = {"atr_pct": 0.015, "atr_mean_pct": 0.02}
        dec = risk_engine.validate(sig, clean_account, indicators=ind)
        assert dec.allow is True

    def test_atr_at_threshold_blocks(self, risk_engine, clean_account):
        """ATR exactly at mean * mult should block (>=)."""
        sig = Signal(action="BUY", confidence=0.80, size_pct=0.03)
        ind = {"atr_pct": 0.05, "atr_mean_pct": 0.02}  # 0.05 == 0.02 * 2.5
        dec = risk_engine.validate(sig, clean_account, indicators=ind)
        assert dec.allow is False

    def test_hold_signal_always_passes(self, risk_engine, clean_account):
        sig = Signal(action="HOLD", confidence=0.1, size_pct=0.0)
        dec = risk_engine.validate(sig, clean_account)
        assert dec.allow is True

    def test_kelly_sizing_tracks_trades(self, risk_engine):
        for i in range(20):
            risk_engine.on_trade_result(0.01 if i % 5 else -0.008, i % 5 != 0)
        status = risk_engine.get_status()
        assert status["kelly_stats"]["total_trades"] == 20
        assert status["kelly_stats"]["win_rate"] > 0

    def test_get_status_keys(self, risk_engine):
        status = risk_engine.get_status()
        assert "circuit_breaker_active" in status
        assert "kelly_stats" in status


# ── model_loader ───────────────────────────────────────────────────────


class TestModelLoader:
    def test_default_construction(self):
        from orchestrator.model_loader import ModelLoader
        ml = ModelLoader()
        assert ml.gpu_pinned == set()
        assert ml.keep_alive is not None

    def test_pinned_set(self):
        from orchestrator.model_loader import ModelLoader
        ml = ModelLoader(gpu_pinned=["phi3:mini"], keep_alive="10m")
        assert "phi3:mini" in ml.gpu_pinned
        assert ml.keep_alive == "10m"

    def test_warm_all_exists(self):
        from orchestrator.model_loader import ModelLoader
        ml = ModelLoader()
        assert callable(getattr(ml, "warm_all", None))

    def test_warmup_model_exists(self):
        from orchestrator.model_loader import ModelLoader
        ml = ModelLoader()
        assert callable(getattr(ml, "warmup_model", None))

    def test_is_loaded_false_initially(self):
        from orchestrator.model_loader import ModelLoader
        ml = ModelLoader()
        assert ml.is_loaded("nonexistent-model") is False

    def test_warmup_times_empty(self):
        from orchestrator.model_loader import ModelLoader
        ml = ModelLoader()
        assert ml.warmup_times == {}


# ── backtest_engine ────────────────────────────────────────────────────


class TestBacktestEngine:
    def test_construction(self):
        from orchestrator.backtest_engine import BacktestEngine
        from unittest.mock import MagicMock
        be = BacktestEngine(executor=MagicMock(), data_path="test.csv")
        assert be.data_path == "test.csv"

    def test_report_dataclass(self):
        from orchestrator.backtest_engine import BacktestReport
        report = BacktestReport()
        assert report.total_cycles == 0


# ── autogen_trader CLI ────────────────────────────────────────────────


class TestCLIParser:
    def test_single_paper(self):
        from orchestrator.autogen_trader import build_parser
        parser = build_parser()
        args = parser.parse_args(["single", "--paper"])
        assert args.command == "single"
        assert args.paper is True

    def test_single_live_mode(self):
        from orchestrator.autogen_trader import build_parser
        parser = build_parser()
        args = parser.parse_args(["single", "--live"])
        assert args.command == "single"
        assert args.paper is False

    def test_daemon_interval(self):
        from orchestrator.autogen_trader import build_parser
        parser = build_parser()
        args = parser.parse_args(["daemon", "--interval", "30"])
        assert args.command == "daemon"
        assert args.interval == 30

    def test_backtest_data(self):
        from orchestrator.autogen_trader import build_parser
        parser = build_parser()
        args = parser.parse_args(["backtest", "--data", "test.csv"])
        assert args.command == "backtest"
        assert args.data == "test.csv"

    def test_status(self):
        from orchestrator.autogen_trader import build_parser
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"


# ── Import smoke test ─────────────────────────────────────────────────


class TestModuleImports:
    @pytest.mark.parametrize("modname", [
        "orchestrator.autogen_executor",
        "orchestrator.autogen_trader",
        "orchestrator.enhanced_risk",
        "orchestrator.backtest_engine",
        "orchestrator.model_loader",
    ])
    def test_import(self, modname):
        m = importlib.import_module(modname)
        assert m is not None
