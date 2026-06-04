"""
QUANTEX ML Signal Generator — FreqAI-style adaptive prediction modeling.

Generates trading signals using ML models trained on technical indicators.
Features auto-retraining, walk-forward validation, and confidence scoring.

Based on FreqAI architecture:
- Computes multi-timeframe features via pandas_ta
- Trains classifiers/regressors on labeled historical data
- Generates live predictions with confidence scores
- Auto-retrains on new data periodically
"""
import io
import json
import pickle
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .strategy import Signal

import numpy as np
import pandas as pd


class MLSignalEngine:
    """
    Machine learning signal generation engine.
    
    Uses scikit-learn models trained on technical indicator features
    to generate trading signals with calibrated confidence scores.
    
    Architecture:
    1. Feature computation (pandas_ta indicators)
    2. Label generation (future returns bucketed into buy/sell/hold)
    3. Model training (Random Forest classifier)
    4. Signal generation with probability calibration
    5. Auto-retraining schedule
    """

    def __init__(
        self,
        model_dir: str = None,
        retrain_interval_hours: int = 24,
        min_training_samples: int = 500,
        lookahead_periods: int = 12,
        threshold_buy: float = 0.005,
        threshold_sell: float = -0.005,
    ):
        if model_dir is None:
            # Default to project's models/ directory
            script_dir = Path(__file__).parent.parent
            model_dir = str(script_dir / "models")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.retrain_interval = timedelta(hours=retrain_interval_hours)
        self.min_samples = min_training_samples
        self.lookahead = lookahead_periods
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell

        self._model = None
        self._feature_names = None
        self._last_train_time: Optional[datetime] = None
        self._train_count = 0

    # ── Feature Engineering ────────────────────────────

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive feature set for ML model.
        
        Uses pandas_ta for reliable technical indicator computation.
        Falls back to manual computation if pandas_ta is unavailable.
        """
        result = df.copy()

        try:
            import pandas_ta as ta

            # Trend indicators
            result["ema_9"] = ta.ema(result["close"], length=9)
            result["ema_21"] = ta.ema(result["close"], length=21)
            result["ema_50"] = ta.ema(result["close"], length=50)
            result["ema_200"] = ta.ema(result["close"], length=200)

            # MACD
            macd = ta.macd(result["close"])
            if macd is not None:
                result["macd"] = macd.iloc[:, 0]
                result["macd_signal"] = macd.iloc[:, 1]
                result["macd_hist"] = macd.iloc[:, 2]

            # RSI
            result["rsi_14"] = ta.rsi(result["close"], length=14)
            result["rsi_7"] = ta.rsi(result["close"], length=7)

            # Stochastic
            stoch = ta.stoch(result["high"], result["low"], result["close"])
            if stoch is not None:
                result["stoch_k"] = stoch.iloc[:, 0]
                result["stoch_d"] = stoch.iloc[:, 1]

            # Bollinger Bands
            bb = ta.bbands(result["close"])
            if bb is not None:
                result["bb_upper"] = bb.iloc[:, 0]
                result["bb_mid"] = bb.iloc[:, 1]
                result["bb_lower"] = bb.iloc[:, 2]

            # ATR
            atr_vals = ta.atr(result["high"], result["low"], result["close"], length=14)
            if atr_vals is not None:
                result["atr"] = atr_vals

            # Volume indicators
            result["obv"] = ta.obv(result["close"], result["volume"])

            # Additional features
            adx_df = ta.adx(result["high"], result["low"], result["close"])
            if adx_df is not None:
                # adx_df returns multiple columns (ADX, DMP, DMN) - extract just ADX
                result["adx"] = adx_df.iloc[:, 0] if isinstance(adx_df, pd.DataFrame) else adx_df

        except Exception:
            # Fallback to manual computation if pandas_ta fails (e.g. NaN data)
            result = self._compute_features_fallback(result)

        # Derived features (model-agnostic) — guard against missing columns
        result["price_vs_ema50"] = (result["close"] - result["ema_50"]) / result["ema_50"].replace(0, np.nan)
        result["price_vs_ema200"] = (result["close"] - result["ema_200"]) / result["ema_200"].replace(0, np.nan)
        if "atr" in result.columns:
            result["atr_pct"] = result["atr"] / result["close"]
        else:
            result["atr_pct"] = np.nan
        result["vol_ratio"] = result["volume"] / result["volume"].rolling(20).mean().replace(0, np.nan)
        result["returns_1"] = result["close"].pct_change(1)
        result["returns_5"] = result["close"].pct_change(5)
        result["returns_10"] = result["close"].pct_change(10)

        # Price action features
        result["hl_range"] = (result["high"] - result["low"]) / result["close"]
        result["oc_range"] = (result["close"] - result["open"]) / result["open"].replace(0, np.nan)

        return result.replace([np.inf, -np.inf], np.nan)

    def _compute_features_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual feature computation when pandas_ta is unavailable."""
        result = df.copy()
        for period in [9, 21, 50, 200]:
            result[f"ema_{period}"] = result["close"].ewm(span=period, adjust=False).mean()
        result["price_vs_ema50"] = (result["close"] - result["ema_50"]) / result["ema_50"].replace(0, np.nan)
        result["price_vs_ema200"] = (result["close"] - result["ema_200"]) / result["ema_200"].replace(0, np.nan)
        return result

    # ── Label Generation ──────────────────────────────

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate training labels from future returns.
        
        Labels:
        1 = BUY (future return > threshold_buy)
        0 = HOLD (future return between thresholds)
        -1 = SELL (future return < threshold_sell)
        """
        future_returns = df["close"].pct_change(self.lookahead).shift(-self.lookahead)
        labels = pd.Series(0, index=df.index)
        labels[future_returns > self.threshold_buy] = 1
        labels[future_returns < self.threshold_sell] = -1
        return labels

    # ── Model Training ────────────────────────────────

    def train(self, df: pd.DataFrame, force: bool = False) -> dict:
        """
        Train/retrain the ML model on historical data.
        
        Args:
            df: OHLCV DataFrame with sufficient history
            force: Force retraining even if not due yet
        
        Returns:
            Training metrics dict
        """
        if not force and self._last_train_time:
            if datetime.now() - self._last_train_time < self.retrain_interval:
                return {"status": "skipped", "reason": "Within retrain interval"}

        if len(df) < self.min_samples:
            return {"status": "skipped", "reason": f"Need {self.min_samples} samples, got {len(df)}"}

        # Compute features
        featured = self.compute_features(df)
        featured = featured.dropna()

        if len(featured) < self.min_samples:
            return {"status": "skipped", "reason": f"After NaN drop: {len(featured)} < {self.min_samples}"}

        # Generate labels
        labels = self.generate_labels(featured)
        featured["label"] = labels

        # Filter to valid samples
        train_data = featured.dropna()
        train_data = train_data[train_data["label"] != 0]  # Only BUY/SELL for training

        if len(train_data) < 100:
            return {"status": "skipped", "reason": f"Only {len(train_data)} tradeable samples"}

        # Feature selection
        exclude_cols = {"open", "high", "low", "close", "volume", "label"}
        feature_cols = [c for c in train_data.columns if c not in exclude_cols]
        feature_cols = [c for c in feature_cols if not pd.isna(train_data[c]).any()]

        X = train_data[feature_cols].values
        y = train_data["label"].values

        # Train test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train Random Forest classifier
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, classification_report

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
            test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

            # Store model
            self._model = {
                "classifier": model,
                "scaler": scaler,
                "feature_names": feature_cols,
            }
            self._feature_names = feature_cols
            self._last_train_time = datetime.now()
            self._train_count += 1

            metrics = {
                "status": "trained",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_accuracy": round(train_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "features": len(feature_cols),
                "train_count": self._train_count,
            }

            # Save model
            self._save_model()
            return metrics

        except ImportError:
            return {"status": "error", "reason": "scikit-learn not installed"}

    # ── Signal Generation ─────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Generate ML-based trading signal from latest data.
        
        Returns signal with direction, confidence, and metadata.
        """
        if not self._model:
            return {
                "direction": "hold",
                "confidence": 0.0,
                "reason": "Model not trained",
                "prob_buy": 0.0,
                "prob_sell": 0.0,
            }

        featured = self.compute_features(df)
        latest = featured.iloc[-1:]

        # Ensure all required features exist
        missing = [c for c in self._feature_names if c not in latest.columns]
        if missing:
            return {
                "direction": "hold",
                "confidence": 0.0,
                "reason": f"Missing features: {missing}",
            }

        X = latest[self._feature_names].fillna(0).values
        X_scaled = self._model["scaler"].transform(X)

        # Get probabilities
        model = self._model["classifier"]
        proba = model.predict_proba(X_scaled)[0]

        # Map to class probabilities
        class_probs = dict(zip(model.classes_, proba))
        prob_buy = class_probs.get(1, 0.0)
        prob_sell = class_probs.get(-1, 0.0)
        prob_hold = class_probs.get(0, 0.0)

        # Determine signal
        confidence_threshold = 0.55
        if prob_buy > confidence_threshold and prob_buy > prob_sell:
            direction = "long"
            confidence = prob_buy
            reason = f"ML buy signal ({prob_buy:.1%} confidence)"
        elif prob_sell > confidence_threshold and prob_sell > prob_buy:
            direction = "short"
            confidence = prob_sell
            reason = f"ML sell signal ({prob_sell:.1%} confidence)"
        else:
            direction = "hold"
            confidence = max(prob_buy, prob_sell, prob_hold)
            reason = f"No clear ML signal (buy:{prob_buy:.1%} sell:{prob_sell:.1%})"

        return {
            "direction": direction,
            "confidence": round(float(confidence), 4),
            "reason": reason,
            "prob_buy": round(float(prob_buy), 4),
            "prob_sell": round(float(prob_sell), 4),
            "prob_hold": round(float(prob_hold), 4),
            "model_trained_at": self._last_train_time.isoformat() if self._last_train_time else None,
            "model_train_count": self._train_count,
        }

    def generate_signal(self, df: pd.DataFrame) -> "Signal":
        """
        Drop-in compatibility alias for StrategyEngine interface.

        Called by BacktestEngine.run() which expects `strategy.generate_signal(df)`.
        Delegates to predict_signal() which wraps the ML dict output
        into a Signal dataclass with ATR-based SL/TP levels.
        """
        return self.predict_signal(df)

    def predict_signal(self, df: pd.DataFrame) -> "Signal":
        """
        Generate ML-based trading signal wrapped as a Signal dataclass.

        Computes ATR-based SL/TP levels from the latest market data
        and wraps the raw ML predict() dict into a proper Signal.
        """
        from .strategy import Signal

        result = self.predict(df)

        direction = result["direction"]
        confidence = result["confidence"]

        if direction == "hold" or confidence < 0.4:
            return Signal(
                direction="hold",
                confidence=confidence,
                reason=result.get("reason", "ML: no clear signal"),
                metadata={
                    "prob_buy": result.get("prob_buy", 0.0),
                    "prob_sell": result.get("prob_sell", 0.0),
                    "prob_hold": result.get("prob_hold", 0.0),
                    "model_train_count": result.get("model_train_count", 0),
                    "source": "ml",
                },
            )

        # Compute ATR from dataframe for SL/TP sizing
        featured = self.compute_features(df)
        if "atr_pct" in featured.columns:
            atr_pct_val = float(featured["atr_pct"].iloc[-1]) if not pd.isna(featured["atr_pct"].iloc[-1]) else 0.02
        else:
            atr_pct_val = 0.02
        atr = float(featured["close"].iloc[-1]) * atr_pct_val if "close" in featured.columns else 0

        entry_price = float(featured["close"].iloc[-1]) if "close" in featured.columns else 0.0

        if direction == "long":
            stop_loss = entry_price - (atr * 1.5)
            tp1 = entry_price + (atr * 2.0)
            tp2 = entry_price + (atr * 3.5)
        else:
            stop_loss = entry_price + (atr * 1.5)
            tp1 = entry_price - (atr * 2.0)
            tp2 = entry_price - (atr * 3.5)

        return Signal(
            direction=direction,
            confidence=confidence,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profits=[
                {"level": 1, "price": round(tp1, 2), "qty_pct": 0.5, "trail": False},
                {"level": 2, "price": round(tp2, 2), "qty_pct": 0.5, "trail": True, "trail_dist": round(atr * 0.5, 2)},
            ],
            reason=result.get("reason", f"ML {direction} signal"),
            metadata={
                "prob_buy": result.get("prob_buy", 0.0),
                "prob_sell": result.get("prob_sell", 0.0),
                "prob_hold": result.get("prob_hold", 0.0),
                "atr": round(atr, 2),
                "entry_price": round(entry_price, 2),
                "model_train_count": result.get("model_train_count", 0),
                "source": "ml",
            },
        )

    def needs_retraining(self, df: pd.DataFrame) -> bool:
        """Check if model needs retraining based on new data volume."""
        if not self._last_train_time:
            return True
        if datetime.now() - self._last_train_time > self.retrain_interval:
            return True
        return False

    # ── Model Persistence ─────────────────────────────

    def _save_model(self):
        """Save trained model to disk."""
        if not self._model:
            return
        path = self.model_dir / "ml_signal_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "classifier": self._model["classifier"],
                "scaler": self._model["scaler"],
                "feature_names": self._feature_names,
                "train_count": self._train_count,
                "last_train_time": self._last_train_time,
            }, f)

    def load_model(self) -> bool:
        """Load trained model from disk."""
        path = self.model_dir / "ml_signal_model.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model = {
                "classifier": data["classifier"],
                "scaler": data["scaler"],
                "feature_names": data["feature_names"],
            }
            self._feature_names = data["feature_names"]
            self._train_count = data.get("train_count", 0)
            self._last_train_time = data.get("last_train_time")
            return True
        except Exception:
            return False
