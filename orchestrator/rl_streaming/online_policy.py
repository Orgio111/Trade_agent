"""
Online RL Policy — Light PPO / Bandit updates from streaming rewards.

Core concept: Each reward event triggers a micro-update to the policy.
Not full PPO (too heavy), but a streaming approximation.

Architecture:
  StreamingRewardEngine → OnlinePolicy.update(advantage, state_vector)
  → Updated weights used for next decision immediately
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OnlinePolicyConfig:
    """Configuration for online policy."""

    # Model architecture
    state_dim: int = 32  # Input state vector dimension
    action_dim: int = 3  # HOLD, LONG, SHORT
    hidden_dim: int = 64

    # Learning
    learning_rate: float = 0.01
    lr_decay: float = 0.9999  # Per update decay
    min_clip_grad_clip: float = 1.0

    # Exploration
    exploration_eps: float = 0.1
    exploration_decay: float = 0.9995
    min_exploration: float = 0.02

    # Entropy bonus (encourage exploration)
    entropy_coef: float = 0.01

    # Baseline (value function)
    value_lr: float = 0.005
    value_loss_coef: float = 0.5

    # Update frequency
    min_updates_before_decay: int = 100

    # Persistence
    save_interval: int = 1000  # Save every N updates
    model_dir: str = "models/online_policy"

    # Device
    device: str = "cpu"  # or "cuda"


class OnlinePolicy:
    """
    Light-weight online policy for streaming RL updates.

    Uses a simple 2-layer MLP with:
    - Policy head: action logits (softmax)
    - Value head: state value estimate

    Updates use REINFORCE with baseline (actor-critic style):
    - Policy gradient: ∇log π(a|s) * advantage
    - Value loss: (V(s) - target)^2
    - Entropy bonus for exploration

    All updates are single-sample (or micro-batch) for ms-level latency.
    """

    def __init__(self, config: OnlinePolicyConfig | None = None):
        self.config = config or OnlinePolicyConfig()
        self._init_weights()
        self._update_count = 0
        self._save_counter = 0

        # Ensure model directory exists
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

        # Try to load existing model
        self._load_latest()

        # Running stats
        self.stats = {
            "updates": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "avg_advantage": 0.0,
        }

    def _init_weights(self):
        """Initialize network weights (Xavier/Glorot)."""
        sd = self.config.state_dim
        hd = self.config.hidden_dim
        ad = self.config.action_dim

        # Xavier initialization
        self.W1 = np.random.randn(sd, hd) * np.sqrt(2.0 / (sd + hd))
        self.b1 = np.zeros(hd)

        self.W2 = np.random.randn(hd, hd) * np.sqrt(2.0 / (hd + hd))
        self.b2 = np.zeros(hd)

        # Policy head
        self.W_pi = np.random.randn(hd, ad) * np.sqrt(2.0 / (hd + ad))
        self.b_pi = np.zeros(ad)

        # Value head
        self.W_v = np.random.randn(hd, 1) * np.sqrt(2.0 / (hd + 1))
        self.b_v = np.zeros(1)

    def forward(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward pass: state → (action_probs, value).

        Args:
            state: (state_dim,) array

        Returns:
            action_probs: (action_dim,) softmax probabilities
            value: scalar state value estimate
        """
        # Ensure correct shape
        state = np.asarray(state, dtype=np.float32).flatten()
        if state.shape[0] != self.config.state_dim:
            # Pad or truncate
            if state.shape[0] < self.config.state_dim:
                state = np.pad(state, (0, self.config.state_dim - state.shape[0]))
            else:
                state = state[: self.config.state_dim]

        # Layer 1
        h1 = np.maximum(0, state @ self.W1 + self.b1)  # ReLU

        # Layer 2
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU

        # Policy head
        logits = h2 @ self.W_pi + self.b_pi
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        action_probs = exp_logits / np.sum(exp_logits)

        # Value head
        value = float(h2 @ self.W_v + self.b_v)

        return action_probs, value

    def act(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        """
        Sample action from policy.

        Returns:
            action: 0=HOLD, 1=LONG, 2=SHORT
            action_prob: probability of selected action
            value: state value estimate
        """
        action_probs, value = self.forward(state)

        if deterministic or np.random.random() > self.config.exploration_eps:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(self.config.action_dim, p=action_probs))

        action_prob = float(action_probs[action])

        # Decay exploration
        if self._update_count > self.config.min_updates_before_decay:
            self.config.exploration_eps = max(
                self.config.min_exploration,
                self.config.exploration_eps * self.config.exploration_decay,
            )

        return action, action_prob, value

    def update(
        self,
        state: np.ndarray,
        action: int,
        advantage: float,
        target_value: float | None = None,
    ) -> dict[str, float]:
        """
        Single-step policy update (REINFORCE with baseline).

        Args:
            state: (state_dim,) state vector
            action: action taken (0, 1, 2)
            advantage: A = R - V(s) (can be pre-computed)
            target_value: optional TD target for value function

        Returns:
            dict with loss components
        """
        state = np.asarray(state, dtype=np.float32).flatten()

        # Forward pass with gradient tracking (manual autograd)
        # Layer 1
        z1 = state @ self.W1 + self.b1
        h1 = np.maximum(0, z1)

        # Layer 2
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)

        # Policy head
        logits = h2 @ self.W_pi + self.b_pi
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        action_probs = exp_logits / np.sum(exp_logits)

        # Value head
        value = float(h2 @ self.W_v + self.b_v)

        # Policy gradient: ∇log π(a|s) * advantage
        # dlogπ/da = one_hot - action_probs
        grad_logits = -action_probs.copy()
        grad_logits[action] += 1.0  # ∇log π = δ_a - π

        # Scale by advantage
        grad_logits *= advantage * self.config.learning_rate

        # Entropy bonus gradient
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        entropy_grad = -self.config.entropy_coef * (np.log(action_probs) + 1)
        grad_logits += entropy_grad * self.config.learning_rate

        # Backprop through policy head
        grad_h2 = grad_logits @ self.W_pi.T
        grad_W_pi = np.outer(h2, grad_logits)
        grad_b_pi = grad_logits

        # Backprop through layer 2
        grad_z2 = grad_h2 * (z2 > 0)  # ReLU gradient
        grad_W2 = np.outer(h1, grad_z2)
        grad_b2 = grad_z2

        # Backprop through layer 1
        grad_h1 = grad_z2 @ self.W2.T
        grad_z1 = grad_h1 * (z1 > 0)
        grad_W1 = np.outer(state, grad_z1)
        grad_b1 = grad_z1

        # Value function loss gradient
        if target_value is not None:
            value_error = value - target_value
            value_loss_grad = self.config.value_loss_coef * value_error * self.config.value_lr
        else:
            # Use advantage as TD error for value
            value_loss_grad = self.config.value_loss_coef * (-advantage) * self.config.value_lr

        grad_h2_v = value_loss_grad * self.W_v.flatten()
        grad_z2_v = grad_h2_v * (z2 > 0)
        grad_W2_v = np.outer(h1, grad_z2_v)
        grad_b2_v = grad_z2_v

        grad_h1_v = grad_z2_v @ self.W2.T
        grad_z1_v = grad_h1_v * (z1 > 0)
        grad_W1_v = np.outer(state, grad_z1_v)
        grad_b1_v = grad_z1_v

        # Combine gradients
        total_grad_W1 = grad_W1 + grad_W1_v
        total_grad_b1 = grad_b1 + grad_b1_v
        total_grad_W2 = grad_W2 + grad_W2_v
        total_grad_b2 = grad_b2 + grad_b2_v

        # Gradient clipping
        for grad in [total_grad_W1, total_grad_b1, total_grad_W2, total_grad_b2,
                     grad_W_pi, grad_b_pi]:
            np.clip(grad, -self.config._grad_clip, self.config._grad_clip, out=grad)

        # Apply updates
        lr = self.config.learning_rate
        if self._update_count > self.config.min_updates_before_decay:
            lr *= self.config.lr_decay

        self.W1 -= lr * total_grad_W1
        self.b1 -= lr * total_grad_b1
        self.W2 -= lr * total_grad_W2
        self.b2 -= lr * total_grad_b2

        self.W_pi -= lr * grad_W_pi
        self.b_pi -= lr * grad_b_pi

        self.W_v -= self.config.value_lr * np.outer(h2, value_loss_grad * (z2 > 0))
        self.b_v -= self.config.value_lr * value_loss_grad * (z2 > 0)

        # Update stats
        self._update_count += 1
        self._save_counter += 1

        policy_loss = -np.log(action_probs[action] + 1e-8) * advantage
        value_loss = (value - (target_value or value)) ** 2

        self.stats["updates"] += 1
        self.stats["policy_loss"] = 0.99 * self.stats["policy_loss"] + 0.01 * policy_loss
        self.stats["value_loss"] = 0.99 * self.stats["value_loss"] + 0.01 * value_loss
        self.stats["entropy"] = 0.99 * self.stats["entropy"] + 0.01 * entropy
        self.stats["avg_advantage"] = 0.99 * self.stats["avg_advantage"] + 0.01 * advantage

        # Periodic save
        if self._save_counter >= self.config.save_interval:
            self.save()
            self._save_counter = 0

        return {
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy),
            "advantage": float(advantage),
            "value": float(value),
            "action_probs": action_probs.tolist(),
        }

    def update_batch(
        self,
        states: list[np.ndarray],
        actions: list[int],
        advantages: list[float],
        target_values: list[float] | None = None,
    ) -> dict[str, float]:
        """Update on a micro-batch."""
        total_loss = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "advantage": 0.0}

        for i, (state, action, advantage) in enumerate(zip(states, actions, advantages)):
            tv = target_values[i] if target_values else None
            loss = self.update(state, action, advantage, tv)
            for k in total_loss:
                total_loss[k] += loss.get(k, 0.0)

        n = len(states)
        return {k: v / n for k, v in total_loss.items()}

    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        """Get full action probability distribution."""
        probs, _ = self.forward(state)
        return probs

    def get_value(self, state: np.ndarray) -> float:
        """Get state value estimate."""
        _, value = self.forward(state)
        return value

    def save(self, path: str | None = None) -> str:
        """Save model to disk."""
        if path is None:
            timestamp = int(time.time())
            path = f"{self.config.model_dir}/policy_{timestamp}.pkl"

        model_data = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W_pi": self.W_pi,
            "b_pi": self.b_pi,
            "W_v": self.W_v,
            "b_v": self.b_v,
            "config": self.config,
            "update_count": self._update_count,
            "stats": self.stats,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        # Also save as "latest"
        latest_path = f"{self.config.model_dir}/policy_latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"[OnlinePolicy] Saved model to {path} (updates={self._update_count})")
        return path

    def _load_latest(self):
        """Load latest model if exists."""
        latest_path = f"{self.config.model_dir}/policy_latest.pkl"
        if os.path.exists(latest_path):
            try:
                with open(latest_path, "rb") as f:
                    model_data = pickle.load(f)

                self.W1 = model_data["W1"]
                self.b1 = model_data["b1"]
                self.W2 = model_data["W2"]
                self.b2 = model_data["b2"]
                self.W_pi = model_data["W_pi"]
                self.b_pi = model_data["b_pi"]
                self.W_v = model_data["W_v"]
                self.b_v = model_data["b_v"]
                self._update_count = model_data.get("update_count", 0)
                self.stats = model_data.get("stats", self.stats)

                logger.info(f"[OnlinePolicy] Loaded model from {latest_path} (updates={self._update_count})")
            except Exception as e:
                logger.warning(f"[OnlinePolicy] Failed to load model: {e}")

    def load(self, path: str):
        """Load model from specific path."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.W1 = model_data["W1"]
        self.b1 = model_data["b1"]
        self.W2 = model_data["W2"]
        self.b2 = model_data["b2"]
        self.W_pi = model_data["W_pi"]
        self.b_pi = model_data["b_pi"]
        self.W_v = model_data["W_v"]
        self.b_v = model_data["b_v"]
        self._update_count = model_data.get("update_count", 0)
        self.stats = model_data.get("stats", self.stats)

        logger.info(f"[OnlinePolicy] Loaded model from {path} (updates={self._update_count})")


class OnlineBanditPolicy:
    """
    Even lighter-weight: Contextual Bandit (LinUCB / Thompson Sampling).

    Use when you want simplest possible online learning.
    Each action maintains its own linear model: reward = θ_a^T * state + noise.
    """

    def __init__(self, state_dim: int = 32, action_dim: int = 3, alpha: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha  # Exploration parameter

        # LinUCB: A_a = I + Σ x x^T, b_a = Σ r x
        self.A = [np.eye(state_dim) for _ in range(action_dim)]
        self.b = [np.zeros(state_dim) for _ in range(action_dim)]
        self.theta = [np.zeros(state_dim) for _ in range(action_dim)]

        self._update_count = 0

    def _update_theta(self, action: int):
        """Solve A θ = b for action's theta."""
        self.theta[action] = np.linalg.solve(self.A[action], self.b[action])

    def act(self, state: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Select action using LinUCB.

        Returns:
            action: selected action
            ucb_scores: UCB score for each action
        """
        state = np.asarray(state, dtype=np.float32).flatten()
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))
        elif state.shape[0] > self.state_dim:
            state = state[:self.state_dim]

        ucb_scores = []
        for a in range(self.action_dim):
            theta_a = self.theta[a]
            mean = theta_a @ state
            # UCB bonus: α * sqrt(x^T A^{-1} x)
            bonus = self.alpha * np.sqrt(state @ np.linalg.solve(self.A[a], state))
            ucb_scores.append(mean + bonus)

        action = int(np.argmax(ucb_scores))
        return action, np.array(ucb_scores)

    def update(self, state: np.ndarray, action: int, reward: float):
        """Update LinUCB matrices."""
        state = np.asarray(state, dtype=np.float32).flatten()
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))
        elif state.shape[0] > self.state_dim:
            state = state[:self.state_dim]

        # Update A and b for chosen action
        self.A[action] += np.outer(state, state)
        self.b[action] += reward * state
        self._update_theta(action)
        self._update_count += 1

    def get_estimates(self, state: np.ndarray) -> np.ndarray:
        """Get expected reward for each action."""
        state = np.asarray(state, dtype=np.float32).flatten()
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))
        elif state.shape[0] > self.state_dim:
            state = state[:self.state_dim]

        return np.array([self.theta[a] @ state for a in range(self.action_dim)])


# Factory functions
def create_online_policy(config: OnlinePolicyConfig | None = None) -> OnlinePolicy:
    """Create online policy instance."""
    return OnlinePolicy(config)


def create_bandit_policy(state_dim: int = 32, action_dim: int = 3, alpha: float = 1.0) -> OnlineBanditPolicy:
    """Create contextual bandit policy."""
    return OnlineBanditPolicy(state_dim, action_dim, alpha)


__all__ = [
    "OnlinePolicyConfig",
    "OnlinePolicy",
    "OnlineBanditPolicy",
    "create_online_policy",
    "create_bandit_policy",
]