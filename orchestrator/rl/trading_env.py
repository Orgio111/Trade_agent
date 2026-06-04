"""
QUANTEX Reinforcement Learning Trading Environment.
Gymnasium-compatible environment for training PPO, SAC, and other RL agents.

Observation Space: 128 features
  [0:6]    OHLCV + returns
  [30:38]  Position + Portfolio
  [38:40]  ML signal (direction, confidence)
  [64:96]  MTF features (32-dim, optional)
  [96:128] Deep encoder (32-dim, GPU LSTM+Transformer, optional)

Action Space: Discrete(4) [hold, long, short, close]
Reward: Hold penalty + PnL-based + position size penalty
"""
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..multi_tf import MultiTimeframeEngine
    from ..deep_encoder import DeepMarketEncoder, MarketSequenceBuffer


class TradingEnvironment:
    """
    Custom trading environment compatible with Gymnasium/Stable-Baselines3.

    Features:
      - 128-dimensional observation space (40 used + 32 MTF + 32 deep encoder + padding)
      - 4 discrete actions (hold, long, short, close)
      - Sharpe-adjusted rewards with asymmetric loss penalty
      - Drawdown tracking and penalty
      - Configurable initial balance, leverage, fees
    """

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10.0,
                 leverage: int = 3, maker_fee: float = 0.0002,
                 taker_fee: float = 0.0004, lookback: int = 50):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.lookback = lookback

        # Observation:
        #   [0:6]    OHLCV + returns
        #   [30:38]  Position + Portfolio
        #   [38:40]  ML signal
        #   [64:96]  MTF features (32 dim)
        #   [96:128] Deep encoder features (32 dim)
        # Total = 128 for flexibility
        self.observation_space_shape = 128

        # Actions: 0=hold, 1=long, 2=short, 3=close
        self.action_space_n = 4

        # State
        self.current_step = lookback
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0.0
        self.position_qty = 0.0
        self.realized_pnl = 0.0
        self.peak_balance = initial_balance
        self.trades: list[dict] = []
        self._total_steps = 0
        self._consecutive_hold_steps = 0  # Track consecutive holds for penalty

        # ML signal feature (optional - injected by GymTradingEnv.set_ml_signal())
        # [ml_direction, ml_confidence] → obs[38], obs[39]
        self._ml_signal = np.array([0.0, 0.0], dtype=np.float32)

        # Multi-Timeframe engine (optional)
        self._mtf_engine: Optional["MultiTimeframeEngine"] = None

        # Deep Market Encoder (optional, GPU-accelerated)
        self._deep_encoder: Optional["DeepMarketEncoder"] = None
        self._seq_buffer: Optional["MarketSequenceBuffer"] = None

    def reset(self) -> np.ndarray:
        """Reset environment to initial state. Returns observation."""
        self.balance = self.initial_balance
        self.current_step = self.lookback
        self.position = 0
        self.entry_price = 0.0
        self.position_qty = 0.0
        self.realized_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.trades = []
        self._total_steps = 0
        self._consecutive_hold_steps = 0

        # Reset deep encoder sequence buffer
        if self._deep_encoder is not None:
            from ..deep_encoder import MarketSequenceBuffer
            self._seq_buffer = MarketSequenceBuffer(lookback=self._deep_encoder.seq_len)
            self._seq_buffer.reset(self.data, self.current_step)

        return self._get_observation()

    def step(self, action: int) -> tuple:
        """
        Execute one step in the environment.

        Args:
            action: 0=hold, 1=long, 2=short, 3=close, 4=scale_in, 5=scale_out

        Returns:
            (observation, reward, done, info)
        """
        self._total_steps += 1
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, {"balance": self.balance}

        prev_balance = self.balance
        current_price = float(self.data.iloc[self.current_step]["close"])

        # Execute action
        if action == 1 and self.position == 0:  # Long
            self.position = 1
            self.entry_price = current_price
            self.position_qty = (self.balance * 0.50) / current_price

        elif action == 2 and self.position == 0:  # Short
            self.position = -1
            self.entry_price = current_price
            self.position_qty = (self.balance * 0.50) / current_price

        elif action == 3 and self.position != 0:  # Close
            self._close_position(current_price)

        # ── Auto-close check (BEFORE unrealized PnL to prevent double-count) ──
        # Check done first (based on balance BEFORE this step's unrealized PnL)
        self.peak_balance = max(self.peak_balance, self.balance)
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        done = (self.balance < self.initial_balance * 0.3) or (self.current_step >= len(self.data) - 1)

        auto_close_penalty = 0.0
        if done and self.position != 0 and self.position_qty > 0:
            # Close at current_price — PnL added ONCE here
            self._close_position(current_price)
            auto_close_penalty = -1.0  # penalty for holding open at end

        # Update unrealized PnL for positions that remain open (not auto-closed)
        if self.position != 0 and self.position_qty > 0:
            unrealized_pnl = self._calculate_pnl(current_price, self.position_qty)
            # 10% unrealized PnL — strong signal without noise collapse
            self.balance = max(self.balance + unrealized_pnl * 0.10, 0.0)

        self.current_step += 1

        # Calculate reward
        pnl_change = self.balance - prev_balance
        reward = self._calculate_reward(pnl_change)

        # ── Action-based reward shaping ────────────────────────────
        # Penalise consecutive holds (agent must trade to avoid this)
        if action == 0:  # HOLD
            self._consecutive_hold_steps += 1
            # Penalty grows: -0.02, -0.04, -0.06, ... capped at -0.30
            hold_penalty = -0.02 * min(self._consecutive_hold_steps, 15)
            reward += hold_penalty
        else:
            self._consecutive_hold_steps = 0
            # No artificial trade bonuses — agent must trade based on PnL alone

        reward += auto_close_penalty

        # Position size penalty — prevent degenerate SCALE_IN spam
        # Cost grows quadratically with leverage exposure
        position_exposure = abs(self.position_qty * self.entry_price) / max(self.balance, 1e-10)
        if position_exposure > 0.8:  # more than 80% of balance exposed
            excess = position_exposure - 0.8
            reward -= excess * 2.0  # linear penalty 2x

        return self._get_observation(), reward, done, {
            "balance": round(self.balance, 4),
            "position": self.position,
            "drawdown": round(dd, 4),
            "trades": len(self.trades),
        }

    def set_ml_signal(self, direction: float, confidence: float):
        """
        Set ML signal for observation[38:40].

        Args:
            direction: 1.0=long, -1.0=short, 0.0=hold
            confidence: 0.0-1.0 confidence score
        """
        self._ml_signal = np.array([float(direction), float(confidence)], dtype=np.float32)

    def _calculate_pnl(self, current_price: float, qty: float) -> float:
        """Calculate PnL for a position."""
        if self.position == 0 or qty <= 0:
            return 0.0
        return (current_price - self.entry_price) * qty * self.leverage * self.position

    def _close_position(self, exit_price: float):
        """Close current position and record trade."""
        pnl = self._calculate_pnl(exit_price, self.position_qty)
        fee = exit_price * self.position_qty * self.taker_fee
        net_pnl = pnl - fee
        self.balance += net_pnl
        self.realized_pnl += net_pnl
        self.trades.append({
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "qty": self.position_qty,
            "pnl": net_pnl,
            "side": "long" if self.position == 1 else "short",
            "step": self.current_step,
        })
        self.position = 0
        self.position_qty = 0.0
        self.entry_price = 0.0

    def _calculate_reward(self, pnl: float) -> float:
        """Sharpe-adjusted reward with asymmetric loss penalty and drawdown penalty."""
        if pnl > 0:
            reward = pnl * 1.0
        else:
            reward = pnl * 1.5  # Asymmetric loss penalty

        # Drawdown penalty
        self.peak_balance = max(self.peak_balance, self.balance)
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        if dd > 0.10:
            reward -= dd * 2.0

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Build 128-dim observation vector."""
        if self.current_step >= len(self.data):
            return np.zeros(self.observation_space_shape, dtype=np.float32)

        recent = self.data.iloc[max(0, self.current_step - self.lookback):self.current_step + 1]
        obs = np.zeros(self.observation_space_shape, dtype=np.float32)

        # OHLCV (normalized)
        current = self.data.iloc[self.current_step]
        base_price = current["close"] if current["close"] > 0 else 1
        obs[0] = current["open"] / base_price
        obs[1] = current["high"] / base_price
        obs[2] = current["low"] / base_price
        obs[3] = 1.0  # Close is reference
        obs[4] = current["volume"] / (current["volume"] + 1)

        # Returns
        if len(recent) >= 5:
            closes = recent["close"].values
            obs[5] = (closes[-1] - closes[-2]) / (closes[-2] + 1e-10)
            obs[6] = (closes[-1] - closes[-5]) / (closes[-5] + 1e-10) if len(closes) >= 5 else 0

        # Position state
        obs[30] = self.position  # -1, 0, 1
        obs[31] = (current["close"] - self.entry_price) / (self.entry_price + 1e-10) if self.entry_price > 0 else 0
        obs[32] = self.position_qty / (self.balance + 1e-10)
        obs[33] = self.realized_pnl / (self.initial_balance + 1e-10)

        # Portfolio state
        obs[34] = self.balance / self.initial_balance
        obs[35] = self.peak_balance / self.initial_balance
        obs[36] = len(self.trades) / 100.0
        obs[37] = (self.peak_balance - self.balance) / (self.peak_balance + 1e-10)

        # ML signal features (injected by GymTradingEnv wrapper)
        obs[38] = self._ml_signal[0]  # ml_direction: 1=long, -1=short, 0=hold
        obs[39] = self._ml_signal[1]  # ml_confidence: 0.0-1.0

        # Multi-Timeframe features (from higher TFs: 1H, 4H, 1D etc.)
        # Injected at obs[64:96] — up to 32 features from 4 timeframes × 8 each
        if self._mtf_engine is not None:
            mtf_vec = self._mtf_engine.get_observation_vector(self.current_step)
            mtf_len = min(len(mtf_vec), 32)
            obs[64:64 + mtf_len] = mtf_vec[:mtf_len]

        # Deep Market Encoder features (GPU-accelerated LSTM + Transformer)
        # Injected at obs[96:128] — 32-dim market embedding
        if self._deep_encoder is not None and self._seq_buffer is not None:
            # Update sequence buffer with current candle
            self._seq_buffer.update(self.data, self.current_step)
            if self._seq_buffer.is_ready:
                seq = self._seq_buffer.get_sequence()
                deep_enc = self._deep_encoder.encode(seq)
                obs[96:128] = deep_enc[:32]

        return obs

    def render(self):
        """Print current state (optional)."""
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        print(f"Step {self.current_step}: Bal=${self.balance:.2f} Pos={self.position} "
              f"DD={dd:.2%} Trades={len(self.trades)}")

    def set_mtf_engine(self, mtf_engine: "MultiTimeframeEngine"):
        """Attach a multi-timeframe engine for richer observations."""
        self._mtf_engine = mtf_engine

    def set_deep_encoder(self, encoder: "DeepMarketEncoder"):
        """Attach a deep market encoder (GPU LSTM+Transformer) for obs[96:128]."""
        from ..deep_encoder import MarketSequenceBuffer
        self._deep_encoder = encoder
        self._seq_buffer = MarketSequenceBuffer(lookback=encoder.seq_len)



def make_env(data: pd.DataFrame, initial_balance: float = 10.0):
    """Factory function for creating trading environments."""
    return TradingEnvironment(data, initial_balance=initial_balance)
