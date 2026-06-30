"""Tests for RL Memory System."""

import sys
sys.path.append(r"C:\Users\Mns Mns\OneDrive\Documents\GitHub\Trade_agent")

import pytest
from datetime import datetime
from orchestrator.rl_memory.memory_store import (
    TradeOutcome,
    MarketState,
    RLMemoryManager,
    create_memory_manager,
)
from orchestrator.rl_memory.reward_function import (
    compute_reward,
    RewardConfig,
    RewardBreakdown,
    generate_policy_update_prompt,
    get_reward_stats,
)
from orchestrator.rl_memory.rl_node import RLState, init_memory_manager
from orchestrator.rl_memory.langgraph_rl_nodes import (
    create_rl_update_node,
    create_rl_on_close_node,
    create_rl_session_summary_node,
    create_rl_retrieve_similar_node,
    add_rl_nodes_to_graph,
)


# ── TradeOutcome Tests ────────────────────────────────────────


class TestTradeOutcome:
    def test_create_trade(self):
        trade = TradeOutcome(
            trade_id="test_1",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="BUY",
            entry_price=61250.0,
            exit_price=61500.0,
            quantity=0.005,
            pnl_pct=0.0041,
            pnl_abs=1.25,
            confidence=0.85,
            regime="trending_up",
            indicators={"rsi": 62.3, "atr": 580.0},
            agent_reasoning=["RSI oversold", "EMA crossover"],
            risk_decision={"allow": True, "max_size": 0.05},
            execution_latency_ms=6.2,
            status="FILLED",
        )
        assert trade.trade_id == "test_1"
        assert trade.pnl_pct == 0.0041

    def test_to_dict(self):
        trade = TradeOutcome(
            trade_id="test_2",
            timestamp=datetime.now(),
            symbol="ETHUSDT",
            action="SELL",
            entry_price=3000.0,
            exit_price=2950.0,
            quantity=0.1,
            pnl_pct=-0.0167,
            pnl_abs=-50.0,
            confidence=0.72,
            regime="trending_down",
            indicators={},
            agent_reasoning=[],
            risk_decision={},
            execution_latency_ms=5.0,
            status="FILLED",
        )
        d = trade.to_dict()
        assert d["trade_id"] == "test_2"
        assert d["symbol"] == "ETHUSDT"


# ── MarketState Tests ─────────────────────────────────────────


class TestMarketState:
    def test_to_text(self):
        state = MarketState(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            price=61250.0,
            regime="trending_up",
            indicators={"rsi": 62.3, "atr": 580.0},
            recent_pnl=0.0041,
            open_positions=1,
            confidence=0.85,
        )
        text = state.to_text()
        assert "BTCUSDT" in text
        assert "61250.00" in text
        assert "trending_up" in text
        assert "rsi:62.3000" in text


# ── Reward Function Tests ─────────────────────────────────────


class TestRewardFunction:
    def test_compute_reward_buy_profitable(self):
        trade = TradeOutcome(
            trade_id="test_3",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="BUY",
            entry_price=61250.0,
            exit_price=61500.0,
            quantity=0.005,
            pnl_pct=0.0041,
            pnl_abs=1.25,
            confidence=0.85,
            regime="trending_up",
            indicators={"rsi": 62.3},
            agent_reasoning=["RSI oversold"],
            risk_decision={},
            execution_latency_ms=6.2,
            status="FILLED",
        )
        reward, breakdown = compute_reward(trade)
        assert isinstance(reward, float)
        assert reward > 0  # Profitable trade should have positive reward
        assert breakdown.pnl_reward > 0

    def test_compute_reward_sell_loss(self):
        trade = TradeOutcome(
            trade_id="test_4",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="SELL",
            entry_price=61500.0,
            exit_price=61250.0,
            quantity=0.005,
            pnl_pct=-0.0041,
            pnl_abs=-1.25,
            confidence=0.65,
            regime="trending_down",
            indicators={},
            agent_reasoning=[],
            risk_decision={},
            execution_latency_ms=5.0,
            status="FILLED",
        )
        reward, breakdown = compute_reward(trade)
        assert reward < 0  # Loss should have negative reward
        assert breakdown.pnl_reward < 0

    def test_compute_reward_hold(self):
        trade = TradeOutcome(
            trade_id="test_5",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="HOLD",
            entry_price=61250.0,
            exit_price=None,
            quantity=0.0,
            pnl_pct=0.0,
            pnl_abs=0.0,
            confidence=0.1,
            regime="ranging",
            indicators={},
            agent_reasoning=[],
            risk_decision={},
            execution_latency_ms=0.0,
            status="FILLED",
        )
        reward, breakdown = compute_reward(trade)
        # HOLD has small penalty
        assert reward <= 0

    def test_reward_config_custom(self):
        config = RewardConfig(
            pnl_weight=2.0,
            confidence_weight=0.5,
        )
        trade = TradeOutcome(
            trade_id="test_6",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="BUY",
            entry_price=61250.0,
            exit_price=62000.0,
            quantity=0.005,
            pnl_pct=0.012,
            pnl_abs=3.5,
            confidence=0.9,
            regime="trending_up",
            indicators={},
            agent_reasoning=[],
            risk_decision={},
            execution_latency_ms=5.0,
            status="FILLED",
        )
        reward, _ = compute_reward(trade, config)
        # With higher pnl_weight, reward should be larger
        assert reward > 0

    def test_get_reward_stats(self):
        trades = [
            TradeOutcome(trade_id=f"t{i}", timestamp=datetime.now(), symbol="BTCUSDT",
                         action="BUY", entry_price=60000, exit_price=60500, quantity=0.01,
                         pnl_pct=0.008, pnl_abs=4.8, confidence=0.8, regime="trending_up",
                         indicators={}, agent_reasoning=[], risk_decision={}, execution_latency_ms=5, status="FILLED")
            for i in range(5)
        ]
        stats = get_reward_stats(trades)
        assert stats["count"] == 5
        assert "mean" in stats
        assert "sharpe" in stats


# ── Policy Prompt Tests ───────────────────────────────────────


class TestPolicyPrompt:
    def test_generate_prompt(self):
        wins = [
            TradeOutcome(trade_id="w1", timestamp=datetime.now(), symbol="BTCUSDT",
                         action="BUY", entry_price=60000, exit_price=61000, quantity=0.01,
                         pnl_pct=0.016, pnl_abs=9.6, confidence=0.85, regime="trending_up",
                         indicators={"rsi": 55}, agent_reasoning=["breakout"], risk_decision={},
                         execution_latency_ms=5, status="FILLED")
        ]
        losses = [
            TradeOutcome(trade_id="l1", timestamp=datetime.now(), symbol="BTCUSDT",
                         action="SELL", entry_price=60000, exit_price=59500, quantity=0.01,
                         pnl_pct=-0.008, pnl_abs=-4.8, confidence=0.6, regime="ranging",
                         indicators={"rsi": 50}, agent_reasoning=["false breakout"], risk_decision={},
                         execution_latency_ms=5, status="FILLED")
        ]
        prompt = generate_policy_update_prompt(wins, losses)
        assert "Winning Trades" in prompt
        assert "Losing Trades" in prompt
        assert "entry_rules" in prompt
        assert "risk_rules" in prompt


# ── RLState Tests ─────────────────────────────────────────────


class TestRLState:
    def test_add_trade(self):
        rl_state = RLState()
        trade = TradeOutcome(
            trade_id="test_7",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="BUY",
            entry_price=61000,
            exit_price=61500,
            quantity=0.005,
            pnl_pct=0.008,
            pnl_abs=2.5,
            confidence=0.8,
            regime="trending_up",
            indicators={},
            agent_reasoning=[],
            risk_decision={},
            execution_latency_ms=5.0,
            status="FILLED",
        )
        rl_state.add_trade(trade, 0.5)
        assert len(rl_state.episode_trades) == 1
        assert rl_state.session_wins == 1
        assert rl_state.session_total_pnl == 2.5

    def test_session_stats(self):
        rl_state = RLState()
        # Add winning trade
        win = TradeOutcome(trade_id="w", timestamp=datetime.now(), symbol="BTCUSDT",
                           action="BUY", entry_price=60000, exit_price=61000, quantity=0.01,
                           pnl_pct=0.016, pnl_abs=9.6, confidence=0.8, regime="up",
                           indicators={}, agent_reasoning=[], risk_decision={},
                           execution_latency_ms=5, status="FILLED")
        rl_state.add_trade(win, 1.0)
        # Add losing trade
        loss = TradeOutcome(trade_id="l", timestamp=datetime.now(), symbol="BTCUSDT",
                            action="SELL", entry_price=60000, exit_price=59500, quantity=0.01,
                            pnl_pct=-0.008, pnl_abs=-4.8, confidence=0.6, regime="down",
                            indicators={}, agent_reasoning=[], risk_decision={},
                            execution_latency_ms=5, status="FILLED")
        rl_state.add_trade(loss, -0.5)

        stats = rl_state.get_session_stats()
        assert stats["trades"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["win_rate"] == 0.5


# ── RL Nodes Tests ────────────────────────────────────────────


class TestRLNodes:
    def test_create_rl_update_node(self):
        node = create_rl_update_node()
        assert callable(node)

    def test_create_rl_on_close_node(self):
        node = create_rl_on_close_node()
        assert callable(node)

    def test_create_rl_session_summary_node(self):
        node = create_rl_session_summary_node()
        assert callable(node)

    def test_create_rl_retrieve_similar_node(self):
        node = create_rl_retrieve_similar_node()
        assert callable(node)

    def test_rl_update_node_skips_non_filled(self):
        node = create_rl_update_node()
        state = {"execution": {"status": "REJECTED"}}
        result = node(state)
        assert result["rl_skipped"] is True
        assert result["rl_reward"] == 0.0


# ── Nightly Loop Tests ────────────────────────────────────────


class TestNightlyLoop:
    def test_config_creation(self):
        from orchestrator.rl_memory.nightly_loop import NightlyConfig
        config = NightlyConfig()
        assert config.lookback_days == 1
        assert config.min_trades_for_update == 10

    def test_trainer_creation(self):
        from orchestrator.rl_memory.nightly_loop import NightlyConfig, NightlyTrainer
        config = NightlyConfig()
        trainer = NightlyTrainer(config)
        assert trainer.config == config


# ── Memory Manager Factory Test ───────────────────────────────


class TestMemoryManagerFactory:
    def test_create_memory_manager(self):
        # Should not raise even if stores unavailable
        mgr = create_memory_manager()
        assert isinstance(mgr, RLMemoryManager)

    def test_available_stores(self):
        mgr = create_memory_manager()
        # At least check the method exists
        assert hasattr(mgr, "is_available")


# ── Integration Test (Mock) ───────────────────────────────────


class TestIntegrationMock:
    """Integration test using mock memory stores."""

    def test_full_cycle_mock(self):
        """Test the full cycle: trade → reward → memory → retrieval."""
        # This test runs without actual Redis/Postgres/Qdrant
        from orchestrator.rl_memory.memory_store import TradeOutcome, MarketState
        from orchestrator.rl_memory.reward_function import compute_reward
        from orchestrator.rl_memory.rl_node import RLState

        # 1. Create trade
        trade = TradeOutcome(
            trade_id="integ_1",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            action="BUY",
            entry_price=61250.0,
            exit_price=61500.0,
            quantity=0.005,
            pnl_pct=0.0041,
            pnl_abs=1.25,
            confidence=0.85,
            regime="trending_up",
            indicators={"rsi": 62.3},
            agent_reasoning=["breakout"],
            risk_decision={"allow": True},
            execution_latency_ms=6.0,
            status="FILLED",
        )

        # 2. Compute reward
        reward, breakdown = compute_reward(trade)
        assert reward > 0
        assert breakdown.total == reward

        # 3. Add to RL state
        rl_state = RLState()
        rl_state.add_trade(trade, reward)

        # 4. Verify session stats
        stats = rl_state.get_session_stats()
        assert stats["trades"] == 1
        assert stats["wins"] == 1
        assert stats["total_pnl"] == 1.25

        print("✅ Integration mock test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])