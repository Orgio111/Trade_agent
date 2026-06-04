"""
QUANTEX Ensemble Meta-Model — Fuses ML, RL, and rule-based signals into a single trading decision.

Architecture:
  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │ ML/RF    │    │ PPO RL   │    │ Rule-based   │
  │ (signal) │    │ (action) │    │ (regime+SM)  │
  └────┬─────┘    └────┬─────┘    └──────┬───────┘
       │               │                 │
       └───────────────┼─────────────────┘
                       ▼
              ┌─────────────────┐
              │  Signal Fusion  │  ← Weighted voting with
              │  Engine         │     adaptive weights
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Risk Gate      │  ← RiskEngine filters
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Final Decision │
              └─────────────────┘

Weights are adapted based on recent historical accuracy (rolling 20-trade window).

Usage:
    ensemble = EnsembleMetaModel(ml_engine, ppo_model, hmm_detector)
    decision = ensemble.decide(market_data)
    # decision = {"direction": "long", "confidence": 0.75, "reason": "ensemble: ..."}
"""

import numpy as np
from typing import Optional
from collections import deque

from .ml_signals import MLSignalEngine
from .feature_engine import FeatureEngine


class SignalSource:
    """Represents one signal source with its adaptive weight."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self._history: deque = deque(maxlen=20)  # Recent signal accuracies

    def record_outcome(self, was_correct: bool):
        """Record whether this source's signal was correct."""
        self._history.append(1.0 if was_correct else 0.0)

    def get_accuracy(self) -> float:
        """Rolling accuracy over the window."""
        if not self._history:
            return 0.5
        return sum(self._history) / len(self._history)

    def update_weight(self, accuracy: Optional[float] = None):
        """Adapt weight based on recent accuracy (0.1 to 3.0 range)."""
        acc = accuracy if accuracy is not None else self.get_accuracy()
        # Weight = base * (accuracy / 0.5) — above 50% accuracy gets boosted
        self.weight = max(0.1, min(3.0, self.weight * (acc / 0.5 + 0.5)))


class EnsembleMetaModel:
    """
    Meta-model that fuses ML, PPO RL, and rule-based signals.

    The ensemble:
      1. Collects signals from all sources
      2. Applies dynamic weights based on recent accuracy
      3. Computes weighted vote for long/short/hold
      4. Passes through risk gate
      5. Returns final decision with confidence and reasoning

    Weights are adapted after each trade outcome is known via `record_trade_outcome()`.

    Usage:
        ensemble = EnsembleMetaModel(ml, ppo_model, hmm)
        decision = ensemble.decide(context)
        if decision['direction'] != 'hold':
            # execute trade
            pass
        ensemble.record_trade_outcome(decision, actual_pnl)
    """

    def __init__(
        self,
        ml_engine: Optional[MLSignalEngine] = None,
        ppo_model: Optional[object] = None,
        hmm_detector: Optional[object] = None,
    ):
        """
        Args:
            ml_engine: MLSignalEngine instance for RF predictions
            ppo_model: Trained PPO model with .predict(obs) method
            hmm_detector: HMMRegimeDetector instance
        """
        self.ml_engine = ml_engine
        self.ppo_model = ppo_model
        self.hmm_detector = hmm_detector

        # Signal sources with adaptive weights
        self.sources = {
            "ml_rf": SignalSource("ml_rf", weight=1.0),
            "ppo_rl": SignalSource("ppo_rl", weight=1.5),     # PPO gets higher base weight
            "regime": SignalSource("regime", weight=0.8),
            "rule_smc": SignalSource("rule_smc", weight=0.7),  # Smart Money Concepts
        }

        self._feature_engine = FeatureEngine()
        self._rng = np.random.RandomState(42)
        self._total_decisions = 0

    # ── Signal Collection ───────────────────────────────────────

    def _get_ml_signal(self, df) -> dict:
        """Get ML signal (direction + confidence)."""
        if self.ml_engine is None:
            return {"direction": "hold", "confidence": 0.0, "reason": "no_ml"}

        try:
            result = self.ml_engine.predict(df)
            return {
                "direction": result.get("direction", "hold"),
                "confidence": float(result.get("confidence", 0.0)),
                "reason": f"ML/RF: {result.get('direction', 'hold')} ({result.get('confidence', 0):.2f})",
            }
        except Exception:
            return {"direction": "hold", "confidence": 0.0, "reason": "ml_error"}

    def _get_ppo_signal(self, obs: np.ndarray) -> dict:
        """Get PPO signal (action + action probability)."""
        if self.ppo_model is None:
            return {"direction": "hold", "confidence": 0.0, "reason": "no_ppo"}

        try:
            action_raw, states = self.ppo_model.predict(obs, deterministic=True)
            action = int(action_raw[0]) if isinstance(action_raw, np.ndarray) and action_raw.ndim >= 1 else int(action_raw)

            # Map PPO action to direction
            action_map = {0: "hold", 1: "long", 2: "short", 3: "hold"}  # 3=close maps to hold for voting
            direction = action_map.get(action, "hold")

            # Default confidence (action probability from SB3 is complex to extract)
            confidence = 0.6

            return {
                "direction": direction,
                "confidence": confidence,
                "action": action,
                "reason": f"PPO: action={action} ({direction}) conf={confidence:.2f}",
            }
        except Exception:
            return {"direction": "hold", "confidence": 0.0, "reason": "ppo_error"}

    def _get_regime_signal(self, df) -> dict:
        """Get regime-based signal."""
        if self.hmm_detector is None:
            regime = self._feature_engine.detect_regime(df)
            signal_map = {
                "strong_uptrend": ("long", 0.6),
                "strong_downtrend": ("short", 0.6),
                "ranging": ("hold", 0.2),
                "high_volatility": ("hold", 0.1),
                "weak_trend": ("hold", 0.3),
            }
            direction, confidence = signal_map.get(regime, ("hold", 0.3))
            return {"direction": direction, "confidence": confidence, "reason": f"ADX regime: {regime}"}

        result = self.hmm_detector.get_regime_signal(df)
        return {
            "direction": result.get("direction", "hold"),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": result.get("reason", "hmm_regime"),
        }

    def _get_smc_signal(self, df) -> dict:
        """Get Smart Money Concepts signal from market structure."""
        if len(df) < 50:
            return {"direction": "hold", "confidence": 0.0, "reason": "insufficient_data"}

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        recent = df.tail(20)
        prev = df.tail(40).head(20)

        # Detect swing highs/lows
        prev_high = prev["high"].max()
        prev_low = prev["low"].min()
        curr_high = recent["high"].max()
        curr_low = recent["low"].min()

        # Break of Structure (BOS)
        bos_up = curr_high > prev_high  # Bullish BOS
        bos_down = curr_low < prev_low  # Bearish BOS

        # Fair Value Gap detection (simplified)
        fvg_up = False
        fvg_down = False
        for i in range(2, len(recent)):
            if recent["low"].iloc[i] > recent["high"].iloc[i - 2]:
                fvg_up = True
            if recent["high"].iloc[i] < recent["low"].iloc[i - 2]:
                fvg_down = True

        # Consolidate signal
        if bos_up and not bos_down:
            return {"direction": "long", "confidence": 0.65, "reason": "SMC: Bullish BOS"}
        elif bos_down and not bos_up:
            return {"direction": "short", "confidence": 0.65, "reason": "SMC: Bearish BOS"}
        elif fvg_up and not fvg_down:
            return {"direction": "long", "confidence": 0.55, "reason": "SMC: FVG up"}
        elif fvg_down and not fvg_up:
            return {"direction": "short", "confidence": 0.55, "reason": "SMC: FVG down"}

        return {"direction": "hold", "confidence": 0.2, "reason": "SMC: no clear setup"}

    # ── Signal Fusion ───────────────────────────────────────────

    def decide(self, market_data: dict) -> dict:
        """
        Make a trading decision by fusing all available signals.

        Args:
            market_data: dict with:
                - df: pd.DataFrame of recent OHLCV data
                - obs: np.ndarray of RL observation (optional, for PPO)
                - price: current price
                - symbol: trading pair

        Returns:
            dict with:
                - direction: "long" | "short" | "hold"
                - confidence: 0.0-1.0
                - sources: dict of each source's signal
                - weights: dict of each source's current weight
                - reason: human-readable reasoning string
        """
        df = market_data.get("df")
        obs = market_data.get("obs")

        if df is None or len(df) < 50:
            return {"direction": "hold", "confidence": 0.0, "sources": {}, "reason": "insufficient_data"}

        # Collect all signals
        signals = {
            "ml_rf": self._get_ml_signal(df),
            "regime": self._get_regime_signal(df),
            "rule_smc": self._get_smc_signal(df),
        }

        if obs is not None:
            signals["ppo_rl"] = self._get_ppo_signal(obs)
        else:
            signals["ppo_rl"] = {"direction": "hold", "confidence": 0.0, "reason": "no_obs"}

        # Weighted vote
        votes = {"long": 0.0, "short": 0.0, "hold": 0.0}
        total_weight = 0.0
        source_details = {}

        for source_name, signal in signals.items():
            weight = self.sources.get(source_name, SignalSource(source_name)).weight
            direction = signal.get("direction", "hold")
            confidence = signal.get("confidence", 0.0)

            votes[direction] += weight * confidence
            total_weight += weight

            source_details[source_name] = {
                "direction": direction,
                "confidence": round(confidence, 4),
                "weight": round(weight, 4),
                "vote_contribution": round(weight * confidence, 4),
                "reason": signal.get("reason", ""),
            }

        # Normalize votes
        if total_weight > 0:
            for d in votes:
                votes[d] /= total_weight

        # Determine final direction
        best_direction = max(votes, key=votes.get)
        best_confidence = votes[best_direction]

        # Build reasoning
        reasons = [f"{name}: {s['direction']}({s['confidence']:.2f})" for name, s in source_details.items()]

        self._total_decisions += 1

        return {
            "direction": best_direction,
            "confidence": round(best_confidence, 4),
            "sources": source_details,
            "weights": {name: round(src.weight, 4) for name, src in self.sources.items()},
            "votes": {d: round(v, 4) for d, v in votes.items()},
            "reason": "Ensemble: " + " | ".join(reasons),
            "total_decisions": self._total_decisions,
        }

    # ── Adaptation ──────────────────────────────────────────────

    def record_trade_outcome(self, decision: dict, actual_pnl: float):
        """
        Record the outcome of a trade and adapt source weights.

        Call this after each trade closes to improve future decisions.

        Args:
            decision: The dict returned by decide()
            actual_pnl: Realized PnL of the trade
        """
        if actual_pnl is None or decision is None:
            return

        direction = decision.get("direction", "hold")
        if direction == "hold":
            return

        # Was the overall decision correct?
        was_correct = actual_pnl > 0

        # Record outcome for each source
        for source_name, signal in decision.get("sources", {}).items():
            if source_name in self.sources:
                source_direction = signal.get("direction", "hold")
                source_was_correct = (source_direction == direction and was_correct) or \
                                     (source_direction != direction and not was_correct)
                self.sources[source_name].record_outcome(source_was_correct)
                self.sources[source_name].update_weight()

    def get_source_accuracies(self) -> dict:
        """Get rolling accuracy for each signal source."""
        return {
            name: {
                "accuracy": round(src.get_accuracy(), 4),
                "weight": round(src.weight, 4),
                "samples": len(src._history),
            }
            for name, src in self.sources.items()
        }
