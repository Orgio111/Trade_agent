"""
QUANTEX HMM Regime Detection — Hidden Markov Model for market regime classification.

Replaces the simple ADX-only regime detection with probabilistic HMM.
Detects 4 regimes: Bull (trending up), Bear (trending down), Range, High-Vol

Architecture:
  1. Extract observation features (returns, volatility, volume, correlation)
  2. Fit HMM with 4 latent states
  3. Viterbi decoding to infer most likely regime sequence
  4. Smooth with exponential window to reduce flickering
  5. Emit regime signal with confidence score

Usage:
    hmm = HMMRegimeDetector(n_regimes=4)
    hmm.fit(train_data)
    regime, confidence = hmm.predict(current_window)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from hmmlearn import hmm


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime classification.

    Fits a Gaussian HMM on multi-dimensional observation features:
      - Normalized returns (1-period)
      - Normalized returns (5-period)
      - ATR/price (volatility)
      - Volume ratio
      - RSI (normalized)

    Regime labels are assigned post-hoc based on the emission means:
      - Highest mean return → Bull
      - Lowest mean return → Bear  
      - Lowest volatility → Range
      - Highest volatility → High-Vol
    """

    REGIME_MAP = {0: "bull", 1: "bear", 2: "ranging", 3: "high_volatility"}
    REGIME_REVERSE = {v: k for k, v in REGIME_MAP.items()}

    def __init__(self, n_regimes: int = 4, lookback: int = 50, seed: int = 42):
        """
        Args:
            n_regimes: Number of hidden states (default 4: bull, bear, range, high-vol)
            lookback: Minimum data points required before prediction
            seed: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self._model: Optional[hmm.GaussianHMM] = None
        self._regime_labels: Optional[dict[int, str]] = None
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._fitted = False
        self._rng = np.random.RandomState(seed)

    # ── Feature Extraction ────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract 5-dimensional observation features from OHLCV data.

        Features:
          0: 1-period normalized return
          1: 5-period normalized return
          2: Volatility (ATR / close)
          3: Volume ratio (volume / SMA(volume, 20))
          4: Normalized RSI (RSI/100 - 0.5)

        Returns:
            (N, 5) numpy array of features
        """
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        n = len(close)

        # 1. Returns
        returns_1 = np.diff(close, n=1, prepend=close[0]) / (close + 1e-10)
        returns_5 = np.diff(close, n=1, prepend=close[0])  # raw for 5-period
        # Compute 5-period return
        ret_5 = np.zeros(n)
        for i in range(5, n):
            ret_5[i] = (close[i] - close[i - 5]) / (close[i - 5] + 1e-10)

        # 2. Volatility (ATR-like)
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        volatility = atr / (close + 1e-10)

        # 3. Volume ratio
        vol_sma = pd.Series(volume).rolling(20, min_periods=1).mean().values
        vol_ratio = volume / (vol_sma + 1e-10)

        # 4. RSI (simplified)
        delta = np.diff(close, prepend=close[0])
        gain = np.maximum(delta, 0)
        loss = np.maximum(-delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_norm = (rsi / 100.0) - 0.5  # Normalize to [-0.5, 0.5]

        features = np.column_stack([
            returns_1,
            ret_5,
            volatility,
            vol_ratio,
            rsi_norm,
        ])
        return features

    # ── Training ────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, force: bool = False):
        """
        Fit HMM on historical data.

        The model is trained once and can be re-used for all subsequent
        predictions. Call `force=True` to re-train.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            force: Re-train even if already fitted
        """
        if self._fitted and not force:
            return

        features = self._extract_features(df)

        # Normalize features
        self._feature_means = np.mean(features, axis=0)
        self._feature_stds = np.std(features, axis=0) + 1e-10
        features_norm = (features - self._feature_means) / self._feature_stds

        # Fit HMM
        self._model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            random_state=self._rng,
            n_iter=200,
            tol=1e-4,
        )
        self._model.fit(features_norm)

        # Assign regime labels based on emission means
        self._label_regimes()
        self._fitted = True

    def _label_regimes(self):
        """Assign semantic regime labels based on mean return & volatility."""
        if self._model is None:
            return

        means = self._model.means_  # (n_regimes, n_features)
        # Feature 0 = 1-period return, Feature 2 = volatility
        mean_returns = means[:, 0]
        mean_vols = means[:, 2]

        # Sort states by return (highest → bull, lowest → bear)
        return_order = np.argsort(mean_returns)[::-1]

        # Assign labels
        self._regime_labels = {}
        # Highest return → bull (0)
        self._regime_labels[return_order[0]] = "bull"
        # Lowest return → bear (1)
        self._regime_labels[return_order[-1]] = "bear"

        # Remaining: high vol → high_volatility, low vol → ranging
        remaining = [i for i in range(self.n_regimes) if i not in (return_order[0], return_order[-1])]
        if len(remaining) >= 2:
            vol_order = sorted(remaining, key=lambda i: mean_vols[i])
            self._regime_labels[vol_order[0]] = "ranging"     # Lowest vol
            self._regime_labels[vol_order[1]] = "high_volatility"  # Highest vol
        elif len(remaining) == 1:
            self._regime_labels[remaining[0]] = "ranging"

    # ── Prediction ──────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Predict the current market regime.

        Args:
            df: OHLCV DataFrame with sufficient history (>= lookback)

        Returns:
            dict with:
              - regime: str ("bull", "bear", "ranging", "high_volatility")
              - confidence: float (0-1, based on posterior probability)
              - regime_probs: dict of regime → probability for all regimes
              - all_regimes: list of str for the last 50 steps (for plotting)
        """
        if not self._fitted or self._model is None:
            # Fallback to ADX-based detection
            from .feature_engine import FeatureEngine
            regime = FeatureEngine.detect_regime(df)
            return {
                "regime": regime,
                "confidence": 0.3,
                "regime_probs": {regime: 0.3},
                "all_regimes": [regime],
                "model": "adx_fallback",
            }

        features = self._extract_features(df)
        features_norm = (features - self._feature_means) / self._feature_stds

        # Get posterior probabilities
        posterior_probs = self._model.predict_proba(features_norm)
        hidden_states = self._model.predict(features_norm)

        # Current state (most recent)
        current_state = int(hidden_states[-1])
        current_probs = posterior_probs[-1]

        # Map to regime labels
        if self._regime_labels and current_state in self._regime_labels:
            current_regime = self._regime_labels[current_state]
        else:
            current_regime = "ranging"

        # Confidence = posterior probability of the most likely state
        confidence = float(np.max(current_probs))

        # Regime probabilities
        regime_probs = {}
        for state in range(self.n_regimes):
            label = self._regime_labels.get(state, f"state_{state}")
            regime_probs[label] = float(current_probs[state])

        # All regimes for the last N steps (for trend visualization)
        all_regimes = [
            self._regime_labels.get(int(s), "ranging")
            for s in hidden_states[-min(50, len(hidden_states)):]
        ]

        return {
            "regime": current_regime,
            "confidence": round(confidence, 4),
            "regime_probs": regime_probs,
            "all_regimes": all_regimes,
            "model": "hmm",
            "n_regimes": self.n_regimes,
            "n_features": features.shape[1],
        }

    def get_regime_signal(self, df: pd.DataFrame) -> dict:
        """
        Get a trading signal based on regime and transition confidence.

        Returns:
            dict with direction, confidence, reasoning
        """
        result = self.predict(df)
        regime = result["regime"]
        confidence = result["confidence"]

        if regime == "bull" and confidence > 0.6:
            return {
                "direction": "long",
                "confidence": confidence * 0.8,
                "reason": f"HMM regime: {regime} (conf={confidence:.2f})",
            }
        elif regime == "bear" and confidence > 0.6:
            return {
                "direction": "short",
                "confidence": confidence * 0.8,
                "reason": f"HMM regime: {regime} (conf={confidence:.2f})",
            }
        elif regime == "high_volatility":
            return {
                "direction": "hold",
                "confidence": 0.3,
                "reason": f"HMM regime: {regime} — too risky",
            }
        else:
            return {
                "direction": "hold",
                "confidence": 0.2,
                "reason": f"HMM regime: {regime} — no clear direction (conf={confidence:.2f})",
            }

    def is_fitted(self) -> bool:
        return self._fitted

    def get_regime_transition_matrix(self) -> Optional[np.ndarray]:
        """Get the transition probability matrix between regimes."""
        if self._model is not None:
            return self._model.transmat_
        return None
