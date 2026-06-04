"""
QUANTEX Reinforcement Learning Trading Environment.
Gymnasium-compatible environment for training PPO, SAC, and other RL agents.

Observation Space: 64 features [OHLCV(5) + returns(2) + position(4) + portfolio(4) + ML signal(2)]
Action Space: Discrete(4) [hold, long, short, close]
Reward: Hold penalty + PnL-based + position size penalty
ML Signal: Injected via set_ml_signal() → obs[38]=direction, obs[39]=confidence
"""
import numpy as np
import pandas as pd


class TradingEnvironment:
    """
    Custom trading environment compatible with Gymnasium/Stable-Baselines3.

    Features:
      - 64-dimensional observation space
      - 6 discrete actions
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

        # Observation: [OHLCV(5) + 20 indicators + position_state(5) + portfolio(4)] = 34
        # Padded to 64 for flexibility
        self.observation_space_shape = 64

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
        prev_price = float(self.data.iloc[self.current_step - 1]["close"])

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
            self._close_position(current_price)        # Note: actions 4 (scale_in) and 5 (scale_out) removed —
        # the 4-action space (hold/long/short/close) prevents degenerate SCALE_IN spam

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
        """Build 64-feature observation vector."""
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

        return obs

    def render(self):
        """Print current state (optional)."""
        dd = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        print(f"Step {self.current_step}: Bal=${self.balance:.2f} Pos={self.position} "
              f"DD={dd:.2%} Trades={len(self.trades)}")


def make_env(data: pd.DataFrame, initial_balance: float = 10.0):
    """Factory function for creating trading environments."""
    return TradingEnvironment(data, initial_balance=initial_balance)
